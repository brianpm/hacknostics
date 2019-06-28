#!/usr/bin/env python
# from the matplotlib docs
# https://matplotlib.org/1.2.1/examples/pylab_examples/polar_bar.html

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

print(f"CHECK: atan2(1,-1) should be 2.36: {np.arctan2(1,-1):.2f}")

# get the data
ds = xr.open_dataset("/Users/brianpm/Dropbox/DataTemporary/CAM6_Eval/c2b9_f2000_cosp.cam.h0.0001-01.nc")
U = ds["U"].isel(lev=-5).squeeze()
V = ds["V"].isel(lev=-5).squeeze()
# print(V.shape)

# TESTS: 
# Pure Eastward wind:
# V *= 0.0 
# Pure Northward:
# U *= 0.0

Dir = np.arctan2(-1*U,-1*V) 

# convert so that Northerly wind is +90degrees:
Dir = (Dir + (np.pi/2)) % (2*np.pi)

print(f"Min direction is {np.degrees(Dir.min().values)} and Max direction is {np.degrees(Dir.max().values)}")

# TODO: We could use color or text to denote wind speed in each bin.
# Spd = np.sqrt(U**2 + V**2)

# BINS: 
# I decided to use arange() and a width to make sure I could 
# keep track of the bin centers, and I wanted centers on 0, 90, 180, 270 for
# plotting purposes. 
# bins = np.linspace(-np.pi, np.pi, 181) # edges / <-- simple bin spec.
bin_width = (2.*np.pi)/72.
bins = np.arange(0-(bin_width/2), 2*np.pi + (bin_width/2), bin_width)
center = (bins[:-1] + bins[1:]) / 2
# CHECK: print(np.degrees(center))
# array gets flattened, don't worry about dims.
DirHist, _ = np.histogram(Dir, bins=bins, density=True)

print(f"Sanity Check -- SUM OF HIST: {DirHist.sum()}")
print(f"Sanity Check 2 -- SUM OF HIST x width: {(DirHist*bin_width).sum()}")
# Explanation: np.histogram uses 'density' not 'mass' so multiply by bin width to get frequency.


# MAKE PLOT
fig, ax = plt.subplots(subplot_kw={"polar":True})

N = len(bins)
theta = center
radii = DirHist * bin_width  # account for density
width = 2*np.pi/N
bars = ax.bar(theta, radii, width=width, bottom=0.0)

ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
ax.set_xticklabels(["E", "", "N", "", "W", "", "S", ""])

# instead of applying rotation to Dir:
# ax.set_theta_offset(np.pi/2)
fig.savefig("/Users/brianpm/Desktop/wind_rose_example.png")
