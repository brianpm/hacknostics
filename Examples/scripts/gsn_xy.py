# gsn_xy.py
# based on:
# gsn_xy_1.ncl
# gsn_xy_2.ncl
# gsn_xy_3.ncl
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#=================================================#
# open file and read in data
#=================================================#
data_location = Path("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/asc/")
data_file = data_location / "data.asc"

# SO HOW DO YOU READ AN ASCII FILE?
# you really have to look at it to figure out the format.
# This one is a space-separate list of columns with a blank first line and no header.

#
# METHOD 1: Use pandas
#
import pandas as pd
data = pd.read_csv(data_file, sep='\s+', header=None)
# 500 rows, 6 columns
x = data[1]
y = data[4]
wks, ax = plt.subplots()
ax.plot(x, y)
wks.savefig("/Users/brianpm/Desktop/gsn_xy_1.png")

#
# METHOD 2: Use numpy
#

# Also show adding titles as in NCL example 2

data_np = np.loadtxt(data_file)
wks, ax = plt.subplots()
ax.plot(x, y, '-k', data_np[:,1], data_np[:,4],':r')
ax.set_title("An xy plot Example")
ax.set_ylabel("Dust (ppm)")
ax.set_xlabel("Time")
wks.savefig("/Users/brianpm/Desktop/gsn_xy_2.png")
# Shows that we got the same answer.

#
# NCL Example 3 does a smoothing and plots both
#
# NOTE: Xarray using the rolling method to make running averages really easy, Pandas has the same thing.
# Here we can just use numpy, but that requires doing the calculation:
def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')

ysmooth = moving_average(y.values, periods=25)

ysmooth_pd = y.rolling(25).mean()

print(ysmooth.shape)
wks, ax = plt.subplots()
ax.plot(x, y, '-k', x, ysmooth_pd,':r', x, ysmooth, "--b")
ax.set_title("The dangers of filtering")
ax.set_ylabel("Dust (ppm)")
ax.set_xlabel("Time")
wks.savefig("/Users/brianpm/Desktop/gsn_xy_3.png")

# NOTE: These are both correct, but they use a different form for the smoothing function.
# By eye with this example, it seems like the numpy convolve method is doing a btter job.
# Worth taking a closer look.