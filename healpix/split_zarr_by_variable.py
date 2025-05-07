import zarr
import os
from pathlib import Path


prefix = "DYAMOND1_history"
zoom = 1

# Define the path of the existing Zarr archive
source_zarr_path = Path(f'/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND1/{prefix}_to_hp{zoom}.zarr')

# Open the existing Zarr archive
source_zarr = zarr.open(source_zarr_path, mode='r')

# Create a directory to store the new Zarr archives, if it doesn't exist
output_directory = Path('/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND1/variables')
output_directory.mkdir(parents=True, exist_ok=True)

# Iterate through the variables in the existing Zarr archive
for variable in source_zarr.keys():
    # Define the new Zarr archive path for this variable
    variable_zarr_path = output_directory / f"{prefix}_{variable}_to_hp{zoom}.zarr"
    
    # Get the source variable data
    source_variable = source_zarr[variable]

    # Extract chunk sizes from the nested tuple
    if hasattr(source_variable, "chunks"):
        if isinstance(source_variable.chunks, tuple):
            chunks = tuple(c[0] for c in source_variable.chunks)
        elif isinstance(source_variable.chunks, int):
            chunks = (source_variable.chunks,)  # Convert to a tuple
        else:
            chunks = None
    else:
        chunks = None  # or some default chunking scheme

    # Create a new Zarr array for this variable with the correct shape and dtype
    variable_zarr = zarr.array(variable_zarr_path, mode='w', shape=source_variable.shape, 
                               dtype=source_variable.dtype, chunks=chunks)
    
    # No need to set node_type for arrays

    # Check if the variable is scalar
    if source_variable.ndim == 0:
        # If scalar, directly assign the value
        variable_zarr[...] = source_variable[()]
    else:
        # If not scalar, copy the entire array
        variable_zarr[:] = source_variable[:]
    
    print(f"Saved variable '{variable}' to {variable_zarr_path}")

    # Consolidate metadata - only needed if the root is a group
    # zarr.consolidate_metadata(variable_zarr_path)

print("Conversion complete.")