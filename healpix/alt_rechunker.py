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

if __name__ == "__main__":
    source_zarr_path = '/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/v4/DYAMOND_diag_1h_to_hp10.zarr'
    new_zarr_path = '/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/1hr/DYAMOND_diag_1h_to_hp10.zarr'
    
    # Define custom chunking for specific variables if needed
    # This is an example - adjust based on your data
    chunk_sizes = {
        # Format: 'variable_name': (chunks_dim1, chunks_dim2, ...)
        # For 1D variables (usually time)
        'time': (48,),
        # For 2D variables (time, cell)
        'PRECT': (48, 262144),
        'PS': (48, 262144),
        # Add more as needed
    }
    
    # Run the rechunking process
    rechunk_zarr_archive(source_zarr_path, new_zarr_path)