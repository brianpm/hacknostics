#
# climo_1.py based on climo_1.ncl
#
# Concepts illustrated:
#   - Calculating seasonal means
#   - Creating a difference plot
#   - Selecting specific decades of data
#   - Calculating decadal means and standard deviations
#   - Calculating probabilities
#   - Calculating differences between decades
#   - Selectively shading between contour levels (using an old method)
#   - Overlaying a stipple pattern to show area of interest
#   - Changing the density of contour shaded patterns
#   - Drawing the zero contour line thicker
#   - Copying coordinate arrays from one variable to another
#
#********************************************************
# (1) Read NCEP/NCAR Reanalysis SLP 
# (2) Use "runave" to compute seasonal [3-month] means
# (3) Use the "ind" function to select the 70s and 90s decades
# (4) Use clmMonLLT and stdMonLLT to compute the decadal
#     means and standard deviations [contributed.ncl]
# (5) Use "ttest" to compute the probabilities
# (6) Use copy_VarCoords [contributed.ncl] to copy coordinate info
# (7) Calculate the differences between the decades
# (8) plot: ZeroNegDashLineContour and ShadeLtContour
#           are in shea_util.ncl. 
#***********************************************************
#
# These files are loaded by default in NCL V6.2.0 and newer
# load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
# load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
# load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
#
# This file still has to be loaded manually
# load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

import numpy as np
import xarray as xr
from scipy import stats  # for ttest
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# NOTE: NCL has some convenience functions to modify contour lines,
#       With these two functions, we can replicate some 
#       of that:
def set_contourthickness(contour, value, thickness=2):
    """Procedure to make contour line at value have given thickness."""
    contour.collections[np.argwhere(contour.levels == value).item()].set_linewidth(thickness) 


def set_NegativeContourColor(contour, color):
    # see: https://stackoverflow.com/questions/14264006/highlight-single-contour-line
    negcontours = np.where(contour.levels < 0)[0]
    [contour.collections[i].set_color(color) for i in negcontours]



# NOTE: the ncl examples use various conventions for naming
#       input files/data. Here they used 'fili' and 'f', but I will 
#       will try to stick with the xarray convention of calling
#       an input dataset 'ds'
ds = xr.open_dataset("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/cdf/slpNCEP.nc")
date = ds['date'].astype(int) # for some reason it will otherwise be float
slp  = ds['SLP']
# compute SEASONAL means
# slp_ts = slp(lat|:,lon|:,time|:)    # (lat,lon,time)
# slp_ts = runave (slp_ts, 3, 0)      # (lat,lon,time) 3-mo run ave
slp_ts = slp.rolling(time=3, center=True).mean()

#********************************
# indices for special dates
#*********************************
# ind7001 = ind(date.eq.197001)       # Jan 1970 
# ind7012 = ind(date.eq.197912)       # Dec 1979
# ind9001 = ind(date.eq.199001)       # Jan 1990
# ind9912 = ind(date.eq.199912)       # Dec 1999

# NOTES:
# to get indices, there are different approaches,
# this is an easy one using where. But note that
# where returns a tuple, so take first entry,
# then convert to scalar using item()
ind7001 = np.where(date == 197001)[0].item()
# another version is just slightly simpler looking:
# *note: this variable is incorrectly named in the ncl example
ind7912 = np.argwhere(date.values == 197912).item()
# another way is to use python's index method on a list;
# this is not recommended because converting between 
# numpy array and list is inefficient:
ind9001 = list(date.values).index(199001)
# could also use argmax to find the FIRST index where
# the array is equal to its max value
ind9912 = np.argmax(date.values == 199912)

# NOTE: we don't even need to use these, though.

#********************************
# compute [2-mon] climatologies
# over specified periods
#********************************
# slpAve7079 = clmMonLLT ( slp_ts(:,:,ind7001:ind7012) )
# slpStd7079 = stdMonLLT ( slp_ts(:,:,ind7001:ind7012) )

# slpAve9099 = clmMonLLT ( slp_ts(:,:,ind9001:ind9912) )
# slpStd9099 = stdMonLLT ( slp_ts(:,:,ind9001:ind9912) )

# note: not sure what "[2-mon]" means here. --> 12-month, I think
# These seem to be the average/standard-deviation
# of the 3-month rolling average of SLP for the 70s and 90s.
# We can directly access these by date instead of
# extracting integer indices:

slpAve7079 = slp_ts.sel(time=slice('1970-01-01', '1979-12-31')).groupby('time.month').mean(dim='time')
slpStd7079 = slp_ts.sel(time=slice('1970-01-01', '1979-12-31')).groupby('time.month').std(dim='time')
slpAve9099 = slp_ts.sel(time=slice('1990-01-01', '1999-12-31')).groupby('time.month').mean(dim='time')
slpStd9099 = slp_ts.sel(time=slice('1990-01-01', '1999-12-31')).groupby('time.month').std(dim='time')
# these will have a dimension 'month' instead of 'time'

#********************************
# compute probabilities for means difference
#******************************** 
# prob = ttest(slpAve9099, slpStd9099^2, 10 \
#             ,slpAve7079, slpStd7079^2, 10 ,True, False )

tstat, prob = stats.ttest_ind_from_stats(slpAve9099, slpStd9099, 10,
  slpAve7079, slpStd7079, 10, equal_var=False)

# copy_VarCoords (slpAve9099, prob)
prob = xr.DataArray(prob, dims=slpAve9099.dims, coords=slpAve9099.coords)
prob.attrs['long_name'] = "Probability: difference between means"

difAve = slpAve9099 - slpAve7079 
# copy_VarCoords (slpAve9099, difAve)
difAve.attrs['long_name'] = "9099-7079" 
difAve.attrs['units']     = "mb"
#********************************
# create plot
#********************************    
nmo = 0                                # for demo, only plot Dec-Feb

# add cyclic point
wrap_data, wrap_lon = add_cyclic_point(difAve, coord=difAve['lon'], axis=-1)
wrap_p = add_cyclic_point(prob)
lons, lats = np.meshgrid(wrap_lon, difAve['lat'])


wks, ax = plt.subplots(figsize=(12,6), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True)

# res = True                             # plot mods desired
# res@cnLevelSelectionMode = "ManualLevels"  # set manual contour levels
# res@cnMinLevelValF       = -6.             # set min contour level
# res@cnMaxLevelValF       =  6.             # set max contour level
# res@cnLevelSpacingF      =   2.            # set contour spacing
cnLevels = np.arange(-6, 6+2, 2)

# res@gsnDraw              = False           # Do not draw plot
# res@gsnFrame             = False           # Do not advance frome

# res@tiMainString         = "SLP Difference: 9099-7079"
# res@gsnCenterString      = "5% stippled"
# res@gsnLeftString        = "DJF"
tiMainString = "SLP Difference: 9099-7079"
CenterString = "5% stippled"
LeftString = "DJF"

CS = ax.contour(lons, lats, wrap_data[nmo,:,:], transform=ccrs.PlateCarree(),
                levels=cnLevels, colors='k')  
set_contourthickness(CS, 0, 3)
set_NegativeContourColor(CS, (.20, .20, .20))


wks.suptitle(tiMainString)
ax.set_title(CenterString)
ax.set_title(LeftString, loc='left')
ax.set_title(difAve.attrs['units'], loc='right')
ax.add_feature(cartopy.feature.LAND, color='lightgray')


# res2 = True                            # res2 probability plots
# res2@gsnDraw             = False       # Do not draw plot
# res2@gsnFrame            = False       # Do not advance frome
# res2@cnLevelSelectionMode = "ManualLevels" # set manual contour levels
# res2@cnMinLevelValF      = 0.00        # set min contour level
# res2@cnMaxLevelValF      = 1.05        # set max contour level
# res2@cnLevelSpacingF     = 0.05        # set contour spacing
# res2@cnInfoLabelOn       = False
# res2@cnLinesOn           = False       # do not draw contour lines
# res2@cnLineLabelsOn      = False       # do not draw contour labels
# res2@cnFillScaleF        = 0.6         # add extra density
# delete(prob@long_name)


# add cyclic point
# plot2   = gsn_csm_contour(wks,gsn_add_cyclic_point(prob(:,:,nmo)), res2) 
# opt = True                    # set up parameters for pattern fill
# opt@gsnShadeFillType = "pattern"  # specify pattern fill
# opt@gsnShadeLow      = 17         # stipple pattern
# plot2   = gsn_contour_shade(plot2, 0.071, 300, opt) # stipple all areas < 0.07 contour
# overlay (plot, plot2)

# cnLevels2 = np.arange(0.0, 1.1, 0.05)
ax.contourf(lons, lats, wrap_p[nmo, :, :], levels=[0.0, 0.07, prob.max().values.item()], 
  hatches=['....', ''], colors='none')

ax.set_xticks(np.arange(-180, 180+30, 30), crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(top=True, right=True, which='both', direction='out')
# Text to make a contour label:
ax.text( 20, -130, f"CONTOUR FROM {min(CS.levels)} to {max(CS.levels)} by {CS.levels[1]-CS.levels[0]}", bbox=dict(facecolor='none', edgecolor='black'))


wks.savefig("/Users/brianpm/Desktop/climo.png")
