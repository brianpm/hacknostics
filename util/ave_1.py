#**************************************
#  ave_1.py --- based on ave_1.ncl
#
# Concepts illustrated:
#   - Calculating a global weighted average
#   - Drawing a time series plot
#   - Copying attributes from one variable to another
#
#**************************************
#
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# NOTE: 'in' is a python keyword, so do not use it as a variable name.
#       A usual convention for xarray is to use 'ds' for 'DataSet'
# NOTE: I didn't find the exact dataset, but it is clearly 100 years of TS,
#       so I just use one locally.
ds = xr.open_dataset("/Users/brianpm/Dropbox/DataTemporary/B1850_c201_CTL.cam.h0.TS.ncrcat.nc")

ts = ds['TS']                            
if 'gw' in ds:
	gw = ds['gw'] # get gaussian weights for ave
else:
	gw = np.cos(np.radians(ds['lat']))  # otherwise use cos(lat)

#****************************************
# calculate averages
#****************************************

# normalized weights:
ywnorm = gw / gw.sum()
# do weighted average in latitude, unweighted in longitude:
globav = (ts * ywnorm).sum(dim='lat', keep_attrs=True).mean(dim='lon', keep_attrs=True)
globav.attrs = ts.attrs  # ts*ywnorm removes metadata, so copy it from orig.

#****************************************
# Create plot
#****************************************
fig, ax = plt.subplots()
tiYAxisString= f"{globav.long_name}  ({globav.units})"
tiXAxisString= "Time Steps"
tiMainString = "Global Weighted Average"
x = np.arange(0, len(globav), 1)
ax.plot(x, globav) 
ax.set_xlabel(tiXAxisString)
ax.set_ylabel(tiYAxisString)
ax.set_title(tiMainString)
fig.savefig("/Users/brianpm/Desktop/ave_1.png")


