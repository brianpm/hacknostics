# healpix

Tools and workflows for converting MPAS/DYAMOND climate model output from netCDF to HEALPix grids stored in Zarr format.

## Active workflow

**`workflow/`** — Current production pipeline (`workflow_parallel` renamed). Contains the full MPAS → HEALPix Zarr conversion pipeline:
- `create_file_manifest.py` — Inventory input files
- `initialize_healpix_zarr.py` — Set up output Zarr structure
- `process_single_file_region.py` — Parallel remapping worker
- `fix_time_in_zarr.py` — Post-processing time coordinate corrections
- PBS job scripts for DYAMOND 1, 2, and 3 datasets at multiple time resolutions

## Archive

**`archive/`** — Older/experimental workflow versions kept for reference:
- `archive/ncar_mpas_workflow/` — Earlier version of the parallel pipeline
- `archive/workflow_parallel_v1/` — First parallel implementation
- `archive/ddaz/` — DDAZ-specific conversion scripts (includes Fortran weight adjustments)

## Supporting scripts

Loose scripts and notebooks at this level are utility/exploratory tools used during development (rechunking, Zarr repair, weight checking, etc.).
