import zarr
import numpy as np

# Open the existing Zarr archive
source_zarr_path = '/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/v4/DYAMOND_diag_1h_to_hp1.zarr'
source_zarr = zarr.open(source_zarr_path, mode='r', zarr_format=2)

# Specify the new Zarr path
new_zarr_path = '/glade/campaign/cgd/cas/brianpm/hack25/DYAMOND_diag_1h_to_hp1_rechunk.zarr'
# Create a new Zarr archive
store = zarr.storage.LocalStore(new_zarr_path)

new_zarr = zarr.group(store, overwrite=True, zarr_format=2)

# Iterate through the variables in the existing Zarr archive
for var_name in source_zarr:
    variable = source_zarr[var_name]
    print(f"Rechunking variable: {var_name}, shape: {variable.shape}")  # Verify shape

    # Adjust new_chunk_size based on the variable's shape
    if len(variable.shape) == 1:
        new_chunk_size = (48,)  # Example for 1D variable
        dimensions = ("time",)  # Replace with actual dimension name if known
        new_variable = new_zarr.create_array(var_name, shape=variable.shape, dtype=variable.dtype, chunks=new_chunk_size)
        new_variable.attrs["_ARRAY_DIMENSIONS"] = dimensions
    elif len(variable.shape) == 2:
        new_chunk_size = (48, 262144)  # Example for 2D variable
        dimensions = ("time", "cell")  # Replace with actual dimension names if known
        new_variable = new_zarr.create_array(var_name, shape=variable.shape, dtype=variable.dtype, chunks=new_chunk_size)
        new_variable.attrs["_ARRAY_DIMENSIONS"] = dimensions
    elif len(variable.shape) == 0:
        print("Scalar variable, copying directly.")
        new_variable = new_zarr.create_array(var_name, shape=variable.shape, dtype=variable.dtype)
        new_variable[...] = variable[...]
        continue  # Skip the chunking loop
    else:
        raise ValueError(f"Unexpected number of dimensions: {len(variable.shape)}")

    # Transfer data to new variable in chunks
    if len(variable.shape) > 0:  # Only transfer data for non-scalar variables
        for i in range(0, variable.shape[0], new_chunk_size[0]):
            chunk_slice = slice(i, min(i + new_chunk_size[0], variable.shape[0]))
            new_variable[chunk_slice, ...] = variable[chunk_slice, ...]

# Consolidate metadata
zarr.consolidate_metadata(new_zarr_path)

print("Rechunking complete.")