#***************************************************************
# climo_4.ncl
#
# Concepts illustrated:
#   - Drawing a latitude/time contour plot
#   - Calculating a zonally averaged annual cycle
#   - Setting contour colors using RGB triplets
#   - Explicitly setting tickmarks and labels on the bottom X axis
#   - Explicitly setting contour levels
#   - Transposing an array
#

import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
#****************************************************
# open file and read in monthly data
#****************************************************

# NOTE: 
# This netCDF file is not CF-compliant. The time dimension is just integers
# of form YYMM. Once we check the file to discover this, we can take appropriate
# corrective measures. Since we want to make climatologies, we'd like to be able
# to use groupby with the time acccessor, so we need to get time into regular
# datetime objects. Here is one way to do it.

ds = xr.open_dataset("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/cdf/xieArkin-T42.nc")

# correct time:
import pandas as pd
otime = ds['time'].astype(int)
times = []
for t in otime:
    str_time = str(t.item())
    yy = str_time[0:2]
    mm = str_time[2:]
    yyint = int(yy)
    if (1900 + yyint) <= 2000:
        yyyy = 1900 + yyint
    else:
        yyyy = 2000 + yyint
    date = f'{yyyy}-{mm}-15'
#     print(f"YEAR: {yy} MONTH: {mm} ==> DATE: {date}")
    times.append(date)
time = pd.to_datetime(times)
ds['time'] = time
P = ds['prc']
lat = ds['lat']


Pclm  = P.groupby('time.month').mean(dim='time', keep_attrs=True)     
# time need not be multiple of 12
Pzone = Pclm.mean(dim='lon', keep_attrs=True)  # lat x month

#
# generate colormap using rgb triplets
#
colors = np.array([[255,255,255],
                  [244,255,244],
                  [217,255,217], 
                  [163,255,163], 
                  [106,255,106],
                  [43,255,106], 
                  [0,224,0], 
                  [0,134,0],
                  [255,255,0],
                  [255,127,0]] ) / 255.
newcmp = ListedColormap(colors)


wks, ax = plt.subplots()
cnLevels = [0.5,1.0,2.0,3.0,4.0,5.0,6.5,7.5]
mlat, mmonth = np.meshgrid(lat, np.linspace(1,12,12))
ax.contourf(mmonth, mlat, Pzone, cmap=newcmp, levels=cnLevels)  
ax.set_xticks = np.linspace(1,12,12)
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])  
wks.suptitle("Zonally Averaged Annual Cycle")
wks.savefig("/Users/brianpm/Desktop/climo_4.png")

#
# Make a plot that more closely matches NCL style
#
plt.rcParams["font.weight"] = "bold"

wks2,ax2 = plt.subplots(constrained_layout=True)
ax2.tick_params(bottom=True, top=True, right=True, axis ='both', which ='major', 
                direction='out')
ax2.tick_params(which='major', length=10)
ax2.tick_params(which='minor', length=5, left=True, right=True, bottom=False)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

ax2.set_xticks(np.arange(1,13,1))
cnLevels = [0.5,1.0,2.0,3.0,4.0,5.0,6.5,7.5]
mlat, mmonth = np.meshgrid(lat, np.linspace(1,12,12))
# norm = mpl.colors.Normalize(vmin=0.5, vmax=7.5)
norm = mpl.colors.BoundaryNorm(cnLevels, newcmp.N)

img = ax2.contourf(mmonth, mlat, Pzone, extend='both', levels=cnLevels, norm=norm,cmap=newcmp)  

ax2.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])  
wks2.suptitle("Zonally Averaged Annual Cycle", fontweight='bold')
wks2.colorbar(img, orientation='horizontal', extend='both', cmap=newcmp, ticks=cnLevels, boundaries=cnLevels) 


if hasattr(Pzone, 'units'):
    ax2.set_title(Pzone.units, loc='right', y=1.05, fontweight='bold')
else:
    print("No units? Did you remember to use keep_attrs when averaging?")
    print(Pzone)
if hasattr(Pzone, 'long_name'):
    ax2.set_title(Pzone.long_name, loc='left', y=1.05, fontweight='bold')
else:
    print("No long_name attribute found.")    


# BONUS: plot a point at the maximum value
# mxloc = np.nonzero(Pzone.values == Pzone.max().values.item())
# mxpt = Pzone[xy[0][0], xy[1][0]]
# ax2.plot(mxpt.month, mxpt.lat, 'xk')
# ax2.text(mxpt.month, mxpt.lat, f"max: {Pzone.max().values.item():.1f}")

wks2.savefig("/Users/brianpm/Desktop/climo_4b.png")

# NOTE: Why is there no yellow contour, like in the NCL example?
# I think this has to do with mpl's algorithm for drawing contours,
# but I do not fully understand it.