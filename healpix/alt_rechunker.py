import zarr
from numcodecs import Blosc
import numpy as np
import json
import os
import shutil
from pathlib import Path

def copy_attributes(source, target):
    """Copy all attributes from source to target"""
    for key, value in source.attrs.items():
        target.attrs[key] = value


def infer_dimensions(name, shape):
    """Helper function to infer dimensions from variable name and shape."""
    if len(shape) == 2:  # Most of our variables are (time, cell)
        return ['time', 'cell']
    elif len(shape) == 1:
        if name == 'time':
            return ['time']
        elif name == 'cell':
            return ['cell']
        else:
            return [f'{name}_dim0']
    else:
        return [f'{name}_dim{i}' for i in range(len(shape))]


def rechunk_zarr_archive(source_path, target_path, chunk_sizes=None):
    """
    Rechunk a zarr archive with proper attribute preservation.
    
    Parameters:
    -----------
    source_path : str
        Path to the source zarr archive
    target_path : str
        Path to create the new rechunked zarr archive
    chunk_sizes : dict, optional
        Dictionary mapping variable names to their new chunk sizes.
        If not provided, default chunking will be used.
    """

    compressor = Blosc(cname='zstd', clevel=3, shuffle=2)

    print(f"Opening source zarr: {source_path}")
    source = zarr.open(source_path, mode='r')
    
    # Create target directory if it doesn't exist
    Path(target_path).mkdir(parents=True, exist_ok=True)
    
    # Create new zarr group
    target = zarr.open(target_path, mode='w', zarr_version=2)
    
    # Default chunk sizes if none provided
    if chunk_sizes is None:
        chunk_sizes = {}
    
    # Process each variable in the zarr archive
    for name in source:
        if hasattr(source[name], 'shape') and hasattr(source[name], 'dtype'):
            # Handle arrays
            source_array = source[name]
            shape = source_array.shape
            dtype = source_array.dtype
            
            # Get dimensions from attributes
            dimensions = source_array.attrs.get('_ARRAY_DIMENSIONS', None)
            print(f"Processing array: {name}, shape: {shape}, dimensions: {dimensions}")
            
            # Determine chunks
            if name in chunk_sizes:
                chunks = chunk_sizes[name]
            else:
                # Set default chunks based on dimensions
                if len(shape) == 1:
                    chunks = (min(48, shape[0]),)
                elif len(shape) == 2:
                    chunks = (min(48, shape[0]), min(262144, shape[1]))
                else:
                    # For higher dimensions, use source chunks as default
                    chunks = source_array.chunks
            
            print(f"  Using chunks: {chunks}")
            
            # Create the array with new chunking
            target_array = target.create_array(
                name, 
                shape=shape, 
                dtype=dtype, 
                chunks=chunks,
                compressor=compressor,
                fill_value=source_array.fill_value if hasattr(source_array, 'fill_value') else None
            )
            
            # Copy all attributes
            copy_attributes(source_array, target_array)
            # Ensure _ARRAY_DIMENSIONS is present for xarray compatibility
            if '_ARRAY_DIMENSIONS' in source_array.attrs:
                target_array.attrs['_ARRAY_DIMENSIONS'] = source_array.attrs['_ARRAY_DIMENSIONS']
            else:
                # Infer dimension names
                shape = source_array.shape
                # Try to guess from variable name or use generic names
                if name == "time":
                    dims = ("time",)
                elif name == "cell":
                    dims = ("cell",)
                elif len(shape) == 1:
                    dims = (f"{name}_dim0",)
                else:
                    dims = tuple(f"{name}_dim{i}" for i in range(len(shape)))
                target_array.attrs['_ARRAY_DIMENSIONS'] = dims

            # Copy the data in chunks
            if len(shape) > 0:
                # For 1D arrays
                if len(shape) == 1:
                    chunk_size = chunks[0]
                    for i in range(0, shape[0], chunk_size):
                        end = min(i + chunk_size, shape[0])
                        print(f"  Copying slice {i}:{end}")
                        target_array[i:end] = source_array[i:end]
                
                # For 2D arrays
                elif len(shape) == 2:
                    chunk_size_0 = chunks[0]
                    for i in range(0, shape[0], chunk_size_0):
                        end_i = min(i + chunk_size_0, shape[0])
                        print(f"  Copying slice {i}:{end_i}, :")
                        target_array[i:end_i, :] = source_array[i:end_i, :]
                
                # For higher dimensions, use a simpler approach
                else:
                    print(f"  Copying entire array")
                    target_array[:] = source_array[:]
            
        elif hasattr(source[name], 'create_group'):
            # Handle nested groups
            print(f"Processing group: {name}")
            target_group = target.create_group(name)
            copy_attributes(source[name], target_group)
            # Ensure _ARRAY_DIMENSIONS is present for xarray compatibility
            if '_ARRAY_DIMENSIONS' in sub_source.attrs:
                sub_target_array.attrs['_ARRAY_DIMENSIONS'] = sub_source.attrs['_ARRAY_DIMENSIONS']
            else:
                sub_shape = sub_source.shape
                if subname == "time":
                    dims = ("time",)
                elif subname == "cell":
                    dims = ("cell",)
                elif len(sub_shape) == 1:
                    dims = (f"{subname}_dim0",)
                else:
                    dims = tuple(f"{subname}_dim{i}" for i in range(len(sub_shape)))
                sub_target_array.attrs['_ARRAY_DIMENSIONS'] = dims
            # Recursively process the group
            source_subgroup = source[name]
            
            # Process each item in the subgroup
            for subname in source_subgroup:
                sub_source = source_subgroup[subname]
                if hasattr(sub_source, 'shape') and hasattr(sub_source, 'dtype'):
                    sub_shape = sub_source.shape
                    sub_dtype = sub_source.dtype
                    
                    # Determine chunks for subgroup array
                    subpath = f"{name}/{subname}"
                    if subpath in chunk_sizes:
                        sub_chunks = chunk_sizes[subpath]
                    else:
                        if len(sub_shape) == 1:
                            sub_chunks = (min(48, sub_shape[0]),)
                        elif len(sub_shape) == 2:
                            sub_chunks = (min(48, sub_shape[0]), min(262144, sub_shape[1]))
                        else:
                            sub_chunks = sub_source.chunks
                    
                    print(f"  Creating subarray: {subname}, shape: {sub_shape}, chunks: {sub_chunks}")
                    sub_target_array = target_group.create_array(
                        subname, 
                        shape=sub_shape, 
                        dtype=sub_dtype, 
                        chunks=sub_chunks,
                        compressor=compressor,
                        fill_value=sub_source.fill_value if hasattr(sub_source, 'fill_value') else None
                    )
                    
                    # Copy attributes
                    copy_attributes(sub_source, sub_target_array)
                    
                    # Copy data
                    print(f"  Copying data for {subname}")
                    if len(sub_shape) == 0:  # scalar
                        sub_target_array[()] = sub_source[()]
                    else:
                        sub_target_array[:] = sub_source[:]
                elif hasattr(sub_source, 'create_group'):
                    # Handle nested subgroups recursively
                    print(f"  Processing nested subgroup: {subname}")
                    nested_group = target_group.create_group(subname)
                    copy_attributes(sub_source, nested_group)
                    # Future enhancement: Add deeper recursion if needed
    
    # Consolidate metadata for better performance with xarray
    print("Consolidating metadata...")
    zarr.consolidate_metadata(target_path)
    print(f"Rechunking complete. New zarr archive saved to: {target_path}")


def combine_and_rechunk_zarrs(source_paths, target_path, dim_chunks=None):
    """
    Combine multiple Zarr archives along the time dimension and rechunk them.
    
    Parameters:
    -----------
    source_paths : list
        List of paths to source Zarr archives
    target_path : str
        Path to create the new combined and rechunked Zarr archive
    dim_chunks : dict, optional
        Dictionary mapping dimension names to their chunk sizes.
        Example: {'time': 36, 'cell': 48}
    """
    print(f"Processing {len(source_paths)} Zarr archives...")
    print(f"Source paths: {[str(p) for p in source_paths]}")  # Add this line

    # Default dimension chunks if none provided
    if dim_chunks is None:
        dim_chunks = {'time': 36, 'cell': 48}
    
    print(f"Processing {len(source_paths)} Zarr archives...")
    
    # Open first source to get metadata and structure
    first_source = zarr.open(source_paths[0], mode='r')
    print(f"\nVariables in first source: {list(first_source.keys())}")

    # Create target directory and zarr group
    Path(target_path).mkdir(parents=True, exist_ok=True)
    target = zarr.open(target_path, mode='w', zarr_version=2)
    
    # Process each variable
    for name in first_source:
        print(f"\nProcessing variable: {name}")
        if not (hasattr(first_source[name], 'shape') and hasattr(first_source[name], 'dtype')):
            print(f"  Skipping {name} - not an array")
            continue
            
        source_array = first_source[name]
        print(f"  Shape: {source_array.shape}, Dtype: {source_array.dtype}, Chunks: {source_array.chunks}")
        dims = source_array.attrs.get('_ARRAY_DIMENSIONS', None)
        print(f"  Dimensions: {dims}")
        if dims is None:
            dims = infer_dimensions(name, source_array.shape)
            print(f"  No dimensions found, inferring: {dims}")
        print(f"  Dimensions: {dims}")
        # Determine chunks based on dimensions
        chunks = []
        for dim in dims:
            if dim in dim_chunks:
                chunks.append(dim_chunks[dim])
                print(f"  Using specified chunk size for {dim}: {dim_chunks[dim]}")
            else:
                # Use source chunking for dimensions not specified
                dim_index = dims.index(dim)
                chunks.append(source_array.chunks[dim_index])
                print(f"  Using source chunk size for {dim}: {source_array.chunks[dim_index]}")
        # Skip if not a time-dependent variable
        if dims is None or 'time' not in dims:
            print(f"Copying non-time variable: {name}")
            # Create array first, then write data
            target_array = target.create_array(
                name,
                shape=source_array.shape,
                chunks=source_array.chunks,
                dtype=source_array.dtype,
                compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
                fill_value=source_array.fill_value if hasattr(source_array, 'fill_value') else None
            )
            target_array[:] = source_array[:]  # Write data after creation
            copy_attributes(source_array, target_array)
            continue
        
        # Calculate combined shape
        time_axis = dims.index('time')
        combined_shape = list(source_array.shape)
        combined_shape[time_axis] = 0
        
        # Calculate total time steps
        for src_path in source_paths:
            src = zarr.open(str(src_path), mode='r')
            combined_shape[time_axis] += src[name].shape[time_axis]
        
        print(f"Creating combined array: {name}")
        print(f"  Shape: {combined_shape}")
        print(f"  Chunks: {chunks}")
        
        # Create target array
        target_array = target.create_array(
            name,
            shape=combined_shape,
            chunks=tuple(chunks),
            dtype=source_array.dtype,
            compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
            fill_value=source_array.fill_value if hasattr(source_array, 'fill_value') else None
        )
        
        # Copy attributes
        copy_attributes(source_array, target_array)
            
        # In the data copying section:
        time_offset = 0
        for src_path in source_paths:
            src = zarr.open(str(src_path), mode='r')  # Convert Path to string
            src_array = src[name]
            time_slice = slice(time_offset, time_offset + src_array.shape[time_axis])
            
            # Create index tuple for proper dimension handling
            idx = [slice(None)] * len(combined_shape)
            idx[time_axis] = time_slice
            
            print(f"  Copying data from {src_path}")
            print(f"    Current time_offset: {time_offset}")
            print(f"    Array shape: {src_array.shape}")
            print(f"    Writing to slice: {idx}")
            
            # Add try-except to catch any writing errors
            try:
                target_array[tuple(idx)] = src_array[:]
                print(f"    Successfully wrote data chunk")
            except Exception as e:
                print(f"    Error writing data: {e}")
                raise
                
            time_offset += src_array.shape[time_axis]
            print(f"    New time_offset: {time_offset}")

        print(f"Finished processing variable {name}")
    
    # Consolidate metadata
    print("Consolidating metadata...")
    zarr.consolidate_metadata(target_path)
    print(f"Combining and rechunking complete. New zarr archive saved to: {target_path}")






# # Example usage in __main__:
if __name__ == "__main__":
    src_loc = Path("/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/3hr")
    src_zarr = sorted(src_loc.glob("DYAMOND_diag_3h.3.75km.*_to_hp1.zarr"))
    
    print(f"Found {len(src_zarr)} source Zarr archives")
    for path in src_zarr:
        print(f"  {path}")
        # # Verify each source exists and is readable
        # try:
        #     z = zarr.open(str(path), mode='r')
        #     print(f"    Variables: {list(z.keys())}")
        # except Exception as e:
        #     print(f"    Error opening: {e}")
    
    tgt_zarr_loc = Path("/glade/campaign/cgd/cas/brianpm/hack25/rechunk")
    tgt_zarr_prefix = "DYAMOND_diag_3h_to_hp1_rechunk"
    new_zarr_path = tgt_zarr_loc / f"{tgt_zarr_prefix}.zarr"
    
    print(f"\nTarget path: {new_zarr_path}")
    
    dim_chunks = {
        'time': 36,
        'cell': 48
    }
    
    print(f"\nUsing dimension chunks: {dim_chunks}")
    
    combine_and_rechunk_zarrs(src_zarr, new_zarr_path, dim_chunks)
    # Verify the result
    print("\nVerifying target Zarr:")
    try:
        result = zarr.open(str(new_zarr_path), mode='r')
        print(f"Variables in result: {list(result.keys())}")
        for name in result:
            if hasattr(result[name], 'shape'):
                print(f"  {name}: shape={result[name].shape}, chunks={result[name].chunks}")
    except Exception as e:
        print(f"Error verifying result: {e}")

# if __name__ == "__main__":

#     source_zarr_path = '/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/v4/DYAMOND_diag_1h_to_hp10.zarr'
#     new_zarr_path = '/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/1hr/DYAMOND_diag_1h_to_hp10.zarr'
    
#     # Define custom chunking for specific variables if needed
#     # This is an example - adjust based on your data
#     chunk_sizes = {
#         # Format: 'variable_name': (chunks_dim1, chunks_dim2, ...)
#         # For 1D variables (usually time)
#         'time': (48,),
#         # For 2D variables (time, cell)
#         'PRECT': (48, 262144),
#         'PS': (48, 262144),
#         # Add more as needed
#     }
    
#     # Run the rechunking process
#     rechunk_zarr_archive(source_zarr_path, new_zarr_path)