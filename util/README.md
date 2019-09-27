# hacknostics util directory

This directory contains some example scripts to show a few useful applications.

## `xy_1.py`
A very simple line plot. This example directly follows the NCL xy_1 example ([http://www.ncl.ucar.edu/Applications/xy.shtml]). It draws two plots, the first is totally bare-bones, so lacks some of the features of the default NCL plot. The second one adds features to try to mimic NCL (even when NCL makes bad choices). Requires python 3.6+, xarray, and matplotlib.

*to run:* `> python xy_1.py`

## `xy_2.py`
Follows the NCL examples for multiple line plot with varying colors, thicknesses, and markers [http://www.ncl.ucar.edu/Applications/xy.shtml]. In this one, I do not try to replicate all the styling of NCL. Instead, I tried to achieve some of the same customizations in the lines, and showed a couple of ways that you could do them. I think these examples start to show how python can achieve the same level of customization in a more compact form. 

*to run:* `> python xy_2.py`


## `bar_1.py`
Examples of bar charts. These are roughly based on `bar_1.ncl` [http://www.ncl.ucar.edu/Applications/bar.shtml]. 
- Example 1 shows a basic bar chart. 
- Example 2 shows a step-filled chart.
- Example 3 switches to horizontal bars, pretty much covering NCL's `bar_2.ncl`. 
- Example 4 shows one way to achieve the coloring effect shown in NCL's `bar_3.ncl`.
- Example 5 shows how to change the thickness of the bars (note, use the kwarg `height` for horizontal bars, but `width` for vertical bars.) 
- Compare Example 3 and 4 to see how to change the outline color, as in `bar_4.ncl`
- Example 6 shows how to change colors for bars, basically like `bar_6.ncl`.
- Example 7 shows bars by category, with different colors, following `bar_7.ncl`.
- Example 8 shows one way to do a stacked bar graph, following `bar_16.ncl`.
- _Example 9_ approximates `bar_22.ncl`. There are 4 panels, a gray background, white gridlines, colored bars, custom xtick labels, common/shared axes, common y-axis label, individual panel titles. I point out that the NCL example is _211 LINES_ (including comments and blanks) while this example is _24 LINES_ of code. 

*to run:* 
    - modify line `f = xr.open_dataset("/Users/brianpm/Downloads/SOI_Darwin.nc")` to point to your file [https://www.ncl.ucar.edu/Applications/Data/cdf/SOI_Darwin.nc]
    `> python bar_1.py`


## `map_example.py`
A fairly bare-bones script that makes a map that is not too ugly.

At the bottom of the script we find the `if __name__ == "__main__"` block that really allows this script to be run like a command line tool. This is where I usually would put in code that would read command line arguments as input, but in this example, the script is set up to just use the values that are set in this block. Once these parameters are set, the data in read using xarray, and the functions at the top are used to make the map.


## `simplest_comparison_map.py`

This script is a little more complex than `map_example.py`, but not by much. The main thing is that this one does use an input from the command line. The command line is parsed using the `argparse` package, which is part of the standard library in python. The argument should be in the form `-f filename.json`, providing a JSON file that will be used to drive the script. This is essentially like making a configuration file that will be fed into the script. The user provides the essential information in this JSON file, and the script does what it needs to. The JSON file looks basically like a python dictionary, and in this case it needs to have entries that give two netCDF file paths, two variables to use, and a place to put the output plot (see the `assert` statements). It can optionally include `slice` definitions to select temporal subsets of the datasets. 

## `ncrcat_files.py`  

This is a command line utility that I wrote and actually use. This is an example of useing python to replace shell scripts. It takes a list of files and a list of variables and uses those along with `ncrcat` to produces a concatenated time series. I wrote this because `xarray` can be very inefficient in dealing with large datasets that have both a large number of variables and a large number of files. In practice, I use this to make compressed netCDF time series files for one or a few variables that are then more portable and easy to deal with (say on my laptop).

## `wind_rose_plot_example.py`
This script is only semi-functional. It was developed as a prototype and proof-of-concept. Takes horizontal wind components (u, v) from a data set and uses `np.arctan2` to get wind direction. Then take histogram of the directions, and plot it on a polar plot to make a wind rose. Apply sensible labels. This should be further refined and turned into a callable function or a pair of functions.

## `cesm_restom.py`
This is a command line utility that calculates the top-of-model and top-of-atmosphere net radiative balance for a CESM data set. To use it, run it with one argument that provides the location of the data. If the argument is a file, it needs to contain `FLNT` and either `FSNT`, `FSNTOA`, or both. The argument can also be a glob pattern, which can be either expanded by the shell or (if in double quotes) by using the pathlib glob functionality. In those cases, the script uses Xarray's `open_mfdataset` to open the dataset, and proceeds the same as for a single file afterward.

The script calculates the global mean, time mean net radiative flux and just prints the result to standard output. If it finds both `FSNT` and `FSNTOA` it will provide both, otherwise whichever is missing will be indicated.