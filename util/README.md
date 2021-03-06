# hacknostics util directory

This directory contains some example scripts to show a few useful applications.

## `simplest_comparison_map.py`

This script is a little more complex than `map_example.py`, but not by much. The main thing is that this one does use an input from the command line. The command line is parsed using the `argparse` package, which is part of the standard library in python. The argument should be in the form `-f filename.json`, providing a JSON file that will be used to drive the script. This is essentially like making a configuration file that will be fed into the script. The user provides the essential information in this JSON file, and the script does what it needs to. The JSON file looks basically like a python dictionary, and in this case it needs to have entries that give two netCDF file paths, two variables to use, and a place to put the output plot (see the `assert` statements). It can optionally include `slice` definitions to select temporal subsets of the datasets. 

## `ncrcat_files.py`  

This is a command line utility that I wrote and actually use. This is an example of useing python to replace shell scripts. It takes a list of files and a list of variables and uses those along with `ncrcat` to produces a concatenated time series. I wrote this because `xarray` can be very inefficient in dealing with large datasets that have both a large number of variables and a large number of files. In practice, I use this to make compressed netCDF time series files for one or a few variables that are then more portable and easy to deal with (say on my laptop).

## `wind_rose_plot_example.py`
This script is only semi-functional. It was developed as a prototype and proof-of-concept. Takes horizontal wind components (u, v) from a data set and uses `np.arctan2` to get wind direction. Then take histogram of the directions, and plot it on a polar plot to make a wind rose. Apply sensible labels. This should be further refined and turned into a callable function or a pair of functions.

## `cesm_restom.py`
This is a command line utility that calculates the top-of-model and top-of-atmosphere net radiative balance for a CESM data set. To use it, run it with one argument that provides the location of the data. If the argument is a file, it needs to contain `FLNT` and either `FSNT`, `FSNTOA`, or both. The argument can also be a glob pattern, which can be either expanded by the shell or (if in double quotes) by using the pathlib glob functionality. In those cases, the script uses Xarray's `open_mfdataset` to open the dataset, and proceeds the same as for a single file afterward.

The script calculates the global mean, time mean net radiative flux and just prints the result to standard output. If it finds both `FSNT` and `FSNTOA` it will provide both, otherwise whichever is missing will be indicated.