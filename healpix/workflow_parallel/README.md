# netCDF to HEALPix+Zarr Processing for MPAS DYAMOND

This code provides a workflow used to process MPAS DYAMOND files from netCDF, remapping to HEALPix grids at zoom level from from 10 down to 1, and then storing as zarr archives.

## `create_file_manifest.py`
Scans specified location for netCDF files to build a file manifest.
This is necessary to:
- determine the number and order of files to be included
- (optionally) determine the number of time samples per file.

## `initialize_healpix_zarr.py`
Sets up an empty version of the Zarr archive(s) that will be filled with the remapped data.

Uses the manifest CSV file to figure out size of time dimension.

Uses first file from manifest to determine variables and dimension sizes.

The reason for this step is to set up Zarr parameters so that chunks can be written in parallel in next step.

## `process_single_file_region.py`

Takes a single netCDF as input to remap to HEALPix grid.

NOTE: Since MPAS files can have variables on cell centers or cell vertices, two separate weights files are currently hard-coded into this script. Each variable is scanned to determine which to use. All variables get remapped to the same HEALPix grids.

The `time_offset` argument is needed to specify where to write the file in the Zarr's time dimension. This is crucial for parallel processing, but note that this workflow expects to process files in sequence assuming 1-time-per-file; modification is necessary for other circumstances with care taken to avoid trying to write to the same chunk simultaneously.

## `run_parallel_regridding_pbs.sh`
Shell script that orchestrates the workflow.


## Additional utilities

### `fix_time_in_zarr.py`

An error in writing the time information was identified in an earlier version of the workflow. This script was developed to derive correct time values and write them into an existing zarr archive.

### `simpler_rechunker_v1.py`
Because the MPAS files are saved with multiple variables per file, and one time sample per file, this workflow developed to process file-by-file. To simplify the parallelization, the time dimension uses chunks of size 1. This leads to very large numbers of files written. This script rechunks the files (and corrects the time dimension if needed, as described above). This results in larger chunks and many fewer files inside the zarr.

### `rch_dy1_z1.sh`
Example shell script that can be used to run the rechunking script.