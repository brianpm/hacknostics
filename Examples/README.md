# hacknostics examples

This directory contains examples based on NCL's examples page. Most of the examples are just plain python scripts (`.py`), but there are some Jupyter notebooks as well. Find brief descriptions of the files in the sections below.


# Scripts Contents

## `xy_1.py`
A very simple line plot. This example directly follows the NCL xy_1 example ([http://www.ncl.ucar.edu/Applications/xy.shtml]). It draws two plots, the first is totally bare-bones, so lacks some of the features of the default NCL plot. The second one adds features to try to mimic NCL (even when NCL makes bad choices). Requires python 3.6+, xarray, and matplotlib.

**to run:**
    
    Modify the *variable* `NCARG_ROOT` that points to the data file. I found ths path on my computer with `> which ncl` and hunting from there. In the NCL examples, this is an environment variable.

    `> python xy_1.py`

## `xy_2.py`
Follows the NCL examples for multiple line plot with varying colors, thicknesses, and markers [http://www.ncl.ucar.edu/Applications/xy.shtml]. In this one, I do not try to replicate all the styling of NCL. Instead, I tried to achieve some of the same customizations in the lines, and showed a couple of ways that you could do them. I think these examples start to show how python can achieve the same level of customization in a more compact form. 

**to run:** `> python xy_2.py`


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
- __Example 9__ approximates `bar_22.ncl`. There are 4 panels, a gray background, white gridlines, colored bars, custom xtick labels, common/shared axes, common y-axis label, individual panel titles. I point out that the NCL example is __211 LINES__ (including comments and blanks) while this example is __24 LINES__ of code. 

**to run:** 

    first, modify line `f = xr.open_dataset("/Users/brianpm/Downloads/SOI_Darwin.nc")` to point to your file [https://www.ncl.ucar.edu/Applications/Data/cdf/SOI_Darwin.nc]

    `> python bar_1.py`

## `ave_1.py`
Simple example of doing global average and plotting time series.

## `climo_1.py`

Reads a dataset, takes a 3-month moving average, computes seasonal climatology, tests significance of difference between 1970s and 1990s climates, and plots the result. Plot shows drawing a map with filled continents, contour lines (with effects), and stippled significance contours. Demonstrates how to do t-test with scipy. Demonstrates adding cyclic point to data. Shows how to get index of a value, but then how you do not need to actually use it. Uses both mean and standard deviation from xarray. Defines two simple functions that can control contour line style. Shows how to use cartopy to get latitude and longitude tick labels. 

## `climo_4.py`
Reads a dataset, processes integer time values (YYMM) to proper datetime objects, calculates climatology and zonal mean. Defines a custom colormap. Specifies contour levels. Customizes tick marks and text labels.  

## `map_example.py`
A fairly bare-bones script that makes a map that is not too ugly.

At the bottom of the script we find the `if __name__ == "__main__"` block that really allows this script to be run like a command line tool. This is where I usually would put in code that would read command line arguments as input, but in this example, the script is set up to just use the values that are set in this block. Once these parameters are set, the data in read using xarray, and the functions at the top are used to make the map.


## `box_1.py`

This goes through all the boxplot examples from NCL's example page. The functionality is somewhat different, but graphically these show similar effects. The last one is not really a box plot, and I show how to draw a violin plot.

## `vector.py`
Covers basic vector plots. Compare with NCL examples at [http://www.ncl.ucar.edu/Applications/generic_vector.shtml].

## `vector_scalar.py`
Covers putting vectors and scalars on same plot. Compare with NCL examples at [http://www.ncl.ucar.edu/Applications/generic_vecscal.shtml].

## `streamlines.py`
Covers basic streamlines plots using matplotlib's streamplot function. Compare with examples at [http://www.ncl.ucar.edu/Applications/generic_stream.shtml].

## `gsn_xy.py`
Simple line graphs based on the gsn_xy examples: [http://www.ncl.ucar.edu/Applications/generic_xy.shtml]

## `latlon_subset.py`
This is the first set of geographic subsetting examples from [https://www.ncl.ucar.edu/Applications/latlon_subset.shtml]. Shows how to start from a global data set and pull out a lat-lon region and make a map. Shades land. Shows simple example of making a function to automatically add titles. Shows simple grid lines. Demonstrates adding some markers to the plot. Includes an example of how to implement `lonFlip` in python/xarray to convert longitude values on a global grid (0-360 <=> -180-180).

# Notebooks Content
## `generic_contours.ipynb`    
Based on `gsn_contour_*.ncl` from the NCL examples. These show basic contour plots. Most of the interesting things are actually to do with customizing the colorbar and fonts.

## `maps_and_shapefiles.ipynb` 

## `polarstereographic.ipynb`

## `Scatter_Plots.ipynb`
Goes through most of the NCL scatter plot examples. Shows how to customize scatter plots, use polar coordinates, markers on maps, rolling means, polynomial curve fitting, etc. Compare with [http://www.ncl.ucar.edu/Applications/scatter.shtml].

## `indices_nino_1.ipynb`
Calculate NINO3.4 index from SST data. Use specified base period for climatology; calculate climatology and anomalies. Average, standardize, and smooth the index. Plot anomalies and standardized anomalies as time series; shade above/below zero in different colors.

## `indices_oni_1.ipynb`
Calculate the Oceanic Nino Index (ONI) from the same SST data as in `indices_nino_1.ipynb`. Do similar manipulation and time series plot. Then make a table showing the monthly values for many years; color table cells based on value.
_I notice that my table and the NCL example seem to be off by 1-month. I have not been able to identify the cause of the mismatch._