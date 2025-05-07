import zarr
import numpy as np
import os
import shutil
from pathlib import Path
import json



print(zarr.__version__)
print(dir(zarr))
def fix_zarr_dimensions(source_path, target_path):
    """
    Fix zarr dimensions for xarray compatibility.
    
    Parameters:
    -----------
    source_path : str
        Path to the source zarr archive
    target_path : str
        Path to create the fixed zarr archive
    """
    print(f"Opening source zarr: {source_path}")
    source = zarr.open(source_path, mode='r')
    
    # Create target directory if it doesn't exist
    if os.path.exists(target_path):
        print(f"Removing existing target: {target_path}")
        shutil.rmtree(target_path)
    
    Path(target_path).mkdir(parents=True, exist_ok=True)
    
    # Create new zarr group
    target = zarr.group(store=target_path, overwrite=True)

    # Analyze the structure to understand dimensions
    print("Analyzing zarr structure...")
    array_info = {}
    for name in source:
        if hasattr(source[name], 'shape') and hasattr(source[name], 'dtype'):
            arr = source[name]
            shape = arr.shape
            dims = arr.attrs.get('_ARRAY_DIMENSIONS', None)
            print(f"Array: {name}, Shape: {shape}, Chunks: {arr.chunks}, Dims: {dims}")
            array_info[name] = {
                'shape': shape,
                'dims': dims,
                'chunks': arr.chunks
            }
    
    # Infer missing dimensions
    dimension_sizes = {}
    
    # First pass - collect known dimensions
    for name, info in array_info.items():
        if info['dims']:
            for i, dim_name in enumerate(info['dims']):
                if dim_name not in dimension_sizes:
                    dimension_sizes[dim_name] = info['shape'][i]
    
    print(f"Identified dimensions: {dimension_sizes}")
    
    # Second pass - fix dimension issues and copy arrays
    for name in source:
        if hasattr(source[name], 'shape') and hasattr(source[name], 'dtype'):
            source_array = source[name]
            shape = source_array.shape
            dtype = source_array.dtype
            
            # Check and fix dimensions
            dims = source_array.attrs.get('_ARRAY_DIMENSIONS', None)
            chunks = source_array.chunks
            
            # CRITICAL FIX: If dims is None or wrong length, create proper dimensions
            if dims is None or len(dims) != len(shape):
                print(f"  Fixing dimensions for {name}, shape: {shape}")
                
                # Try to infer dimensions based on size
                if len(shape) == 1:
                    # Check if this could be a known dimension
                    found_dim = False
                    for dim_name, dim_size in dimension_sizes.items():
                        if dim_size == shape[0]:
                            dims = [dim_name]
                            found_dim = True
                            print(f"    Assigned dimension: {dims}")
                            break
                    
                    if not found_dim:
                        # Default to using the variable name as dimension if it's a coordinate
                        if name in dimension_sizes or name == 'time' or name.endswith('_bnds'):
                            dims = [name]
                        else:
                            # Generic dimension name
                            dims = [f"dim_{shape[0]}"]
                        print(f"    Created new dimension: {dims}")
                
                elif len(shape) == 2:
                    # For 2D arrays, try using the name as prefix
                    # Special case for common coordinate bounds
                    if name.endswith('_bnds'):
                        base_name = name[:-5]  # Remove _bnds
                        if base_name in dimension_sizes:
                            dims = [base_name, 'bnds']
                    else:
                        # Try to match first dimension to known dims
                        first_dim = None
                        for dim_name, dim_size in dimension_sizes.items():
                            if dim_size == shape[0]:
                                first_dim = dim_name
                                break
                        
                        if first_dim:
                            dims = [first_dim, f"{name}_dim"]
                        else:
                            dims = [f"{name}_dim0", f"{name}_dim1"]
                    
                    print(f"    Created dimensions: {dims}")
                
                else:
                    # For higher dimensions, use generic names
                    dims = [f"{name}_dim{i}" for i in range(len(shape))]
                    print(f"    Created generic dimensions: {dims}")
            
            # Validate chunks
            if chunks and len(chunks) != len(shape):
                print(f"  Warning: Chunks length {len(chunks)} doesn't match shape length {len(shape)} for {name}")
                # Fix chunks to match shape
                chunks = tuple(min(c, s) for c, s in zip(chunks, shape)) if len(chunks) <= len(shape) else tuple(s for s in shape)
            create_array_kwargs = dict(
                shape=shape,
                dtype=dtype,
            )
            # Only set chunks if not scalar and chunks is not None/empty
            if len(shape) > 0 and chunks and len(chunks) > 0 and all(c > 0 for c in chunks):
                print("---- set chunks ----")
                create_array_kwargs['chunks'] = chunks
            if hasattr(source_array, 'fill_value'):
                create_array_kwargs['fill_value'] = source_array.fill_value
            
            # Create the array with fixed dimensions
            print(f"  Creating array {name} with dims {dims} and chunks {chunks} (shape={shape})")
            if name in target:
                print(f"  Array {name} already exists in target, skipping creation.")
                target_array = target[name]
            else:
                # Always use the root group for creation
                if name == 'crs':
                    print("---------- CHECK")
                    print(create_array_kwargs)
                    print(f"{type(target) = }, {hasattr(target, 'create_array')}")
                    print(f"{dir(target) = }")
                target_array = target.create_dataset(name, **create_array_kwargs)  # zarr v2 uses create_dataset
                print("-------------------- DONE!!!!!!")
            
            
            # Copy attributes and set fixed dimensions
            for key, value in source_array.attrs.items():
                if key != '_ARRAY_DIMENSIONS':  # Skip dimensions, we'll set our fixed version
                    target_array.attrs[key] = value
            
            # Set the fixed dimensions
            target_array.attrs['_ARRAY_DIMENSIONS'] = dims
            
            # Copy data
            print(f"  Copying data for {name}")
            if len(shape) == 0:  # Scalar
                target_array[()] = source_array[()]
            else:
                target_array[:] = source_array[:]

        elif hasattr(source[name], 'create_group'):  # It's a group
            print(f"Processing group: {name}")
            if name not in target:
                target_group = target.create_group(name)
            else:
                target_group = target[name]
            
            # Copy attributes
            for key, value in source[name].attrs.items():
                target_group.attrs[key] = value
            
            # TODO: Handle nested groups if needed
    
    # Add global attributes if any
    if hasattr(source, 'attrs'):
        for key, value in source.attrs.items():
            target.attrs[key] = value
    
    # Consolidate metadata
    print("Consolidating metadata...")
    zarr.consolidate_metadata(target_path)
    print(f"Fix complete. New zarr archive saved to: {target_path}")
    
    # Verify the dimensions match chunks for all arrays
    verify_zarr_dimensions(target_path)
    
    return target_path

def verify_zarr_dimensions(zarr_path):
    """Verify that dimensions and chunks match for all arrays"""
    print(f"\nVerifying zarr dimensions at: {zarr_path}")
    store = zarr.open(zarr_path, mode='r')
    
    all_valid = True
    
    for name in store:
        if hasattr(store[name], 'shape'):
            arr = store[name]
            dims = arr.attrs.get('_ARRAY_DIMENSIONS', [])
            shape = arr.shape
            chunks = arr.chunks
            
            # Check if dimensions match shape
            if len(dims) != len(shape):
                print(f"❌ {name}: Dimensions ({len(dims)}) don't match shape ({len(shape)})")
                all_valid = False
            else:
                print(f"✓ {name}: Dimensions match shape")
            
            # Check if chunks match shape
            if chunks and len(chunks) != len(shape):
                print(f"❌ {name}: Chunks ({len(chunks)}) don't match shape ({len(shape)})")
                all_valid = False
            else:
                print(f"✓ {name}: Chunks match shape")
    
    if all_valid:
        print("✅ All arrays have valid dimensions and chunks")
    else:
        print("❌ Some arrays have dimension or chunk issues")

if __name__ == "__main__":
    source_zarr_path = '/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/v4/DYAMOND_diag_1h_to_hp1.zarr'
    new_zarr_path = '/glade/campaign/cgd/cas/brianpm/hack25/DYAMOND_diag_1h_to_hp1_fixed.zarr'
    
    # Fix the zarr dimensions
    fixed_path = fix_zarr_dimensions(source_zarr_path, new_zarr_path)
    
    print("\nTry opening with xarray:")
    print("import xarray as xr")
    print(f"ds = xr.open_zarr('{fixed_path}')")
    print("print(ds)")