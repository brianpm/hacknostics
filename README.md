# hacknostics

Some atmospheric diagnostics, focused on analysis of CESM.

The repo is organized into Notebooks, utilities (`util`), and notes (`docs`). 

## Examples
This is now the main feature of hacknostics. Included are both Notebooks and python scripts. These are recreating many of the NCL examples. Specifically using minimal packages; focused on numpy, xarray, matplotlib, and cartopy. Lots of notes and comments about what works well and what doesn't. These examples try to keep the basic style of the NCL examples to the extent possible.

## Notebooks

Example applications are shown in these Notebooks.

- `vertical_cross_section_example` shows how to make a vertical cross section on pressure levels.
- `dev_metadata_decorator` is in development, but shows how one might design a decorator to a function that will preserve metadata.
- `Regrid_with_xesmf` is an example that shows horizontal regridding using xesmf as well as simple interpolation to pressure levels that uses Numpy's `interp` along with just-in-time compilation using Numba.

## util

The `util` directory contains `.py` files that have useful functions or provide command line interfaces. See the README in that directory for detailed descriptions.


