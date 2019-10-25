import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

trythese = ['isobaricInhPa',
    'meanSea',
    'surface',
    'heightAboveGround',
    'isothermZero',
    'tropopause',
    'maxWind',
    'pressureFromGroundLayer',
    'heightAboveGroundLayer',
    'unknown',
    'entireAtmosphere',
    'depthBelowLand',
    'cloudBase',
    'cloudTop']

fil = "/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/grb/ruc.grb"
# for t in trythese:
#     ds = xr.open_dataset(fil, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': t}})
#     if "TMP_236_SPDY" in ds:
#        print(f"THIS ONE: {t}")
#     else:
#         # print(ds)
#         print(ds.dims)

# # Does not occur in file as TMP_236_SPDY
# I think this is because GRIB files are so simple,
# and rely on software and lookup tables to construct
# metadata (?). 
# ncl_filedump:
# float TMP_236_SPDY ( lv_SPDY3, gridx_236, gridy_236 )
#          center :	US National Weather Service - NCEP (WMC)
#          long_name :	Temperature
#          units :	K
#          _FillValue :	1e+20
#          coordinates :	gridlat_236 gridlon_236
#          level_indicator :	116
#          grid_number :	236
#          parameter_table_version :	2
#          parameter_number :	11
#          model :	Rapid Refresh (RAP)
#          forecast_time :	1
#          forecast_time_units :	hours
#          initial_time :	06/14/2005 (00:00)

# from above, we find that in isobaricInhPa there is t

# ds = xr.open_dataset(fil, engine='cfgrib', 
#         backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
# a = ds['t']

# print(a)
# It has 37 Levels, not 6, so doesn't match NCL example.

# Changing to look at dim sizes, looks like pressureFromGroundLayer is size 6

ds = xr.open_dataset(fil, engine='cfgrib', 
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'pressureFromGroundLayer'}})
print(ds.data_vars)
temp = ds['t']
print(f"Shape of temp is {temp.shape}")
# print(temp)
# This is definitely matching the NCL example.

lat2d = ds['latitude']
lon2d = ds['longitude']
# print(temp)

# So minimal amount of effort plot is this:
fig, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
img = ax.pcolormesh(lon2d, lat2d, temp[0,:,:])
fig.savefig("/Users/brianpm/Desktop/latlon_subset_A.png")
# NOTE: Matplotlib/Cartopy automatically set the extent, so it "zooms in" on the area of data.

# Next, add some decoration
fig, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
img = ax.pcolormesh(lon2d, lat2d, temp[0,:,:])
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.STATES)
ax.set_title(temp.GRIB_units, loc='right')
ax.set_title(temp.GRIB_name, loc='left')
ax.tick_params(bottom=True, left=True)
ax.set_yticks(np.arange(20, 60, 10))
# not sure why, but the longitude values seem to be 0:360, but the plotting is happening in -180:180 (?)
ax.set_xticks(np.arange(-135, -65, 15))
fig.suptitle("Plotting with 2D coordinate arrays")
cb = fig.colorbar(img, orientation='horizontal')
fig.savefig("/Users/brianpm/Desktop/latlon_subset_B.png")

# Zoom in on New Mexico and Colorado

# The NCL example shows how to get the indices of temp
# That doesn't seem necessary here.
# But if you did need to do that, you could use the alternative
# that is commented below.
lat_min =   31
lat_max =   42
lon_min = 360-110
lon_max =  360-102 

# basic method: mask values outside region with nan
temp_sub = np.where((lat2d > lat_min) & (lat2d < lat_max), temp[0,:,:], np.nan)
temp_sub = np.where((lon2d > lon_min) & (lon2d < lon_max), temp_sub, np.nan)

fig, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
img = ax.pcolormesh(lon2d, lat2d, temp_sub)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.STATES)
ax.set_title(temp.GRIB_units, loc='right')
ax.set_title(temp.GRIB_name, loc='left')
ax.tick_params(bottom=True, left=True)
ax.set_yticks(np.arange(20, 60, 10))
# not sure why, but the longitude values seem to be 0:360, but the plotting is happening in -180:180 (?)
ax.set_xticks(np.arange(-135, -65, 15))
fig.suptitle("Plotting with 2D coordinate arrays")
cb = fig.colorbar(img, orientation='horizontal')
fig.savefig("/Users/brianpm/Desktop/latlon_subset_C.png")


# ALTERNATIVE -- Getting indices
region_mask = np.nonzero((lat2d.values > lat_min) & (lat2d.values < lat_max) & (lon2d.values > lon_min) & (lon2d.values < lon_max))
# region_mask is a tuple of arrays for x, y
# print(region_mask)
print(f"Region mask dim0 shape: {region_mask[0].shape}, dim1 shape: {region_mask[1].shape}")
# Then just get the sub-arrays to use for the plot
temp_sub2 = temp[:,region_mask[0].min():region_mask[0].max(), region_mask[1].min():region_mask[1].max()]
lonsub = lon2d[region_mask[0].min():region_mask[0].max(), region_mask[1].min():region_mask[1].max()]
latsub = lat2d[region_mask[0].min():region_mask[0].max(), region_mask[1].min():region_mask[1].max()]
# print(temp_sub2)

# NOTE: Since the sub-arrays are now just covering a small area
# the plot automatically zooms in on the region.
fig, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
img = ax.pcolormesh(lonsub, latsub, temp_sub2[0,:,:], transform=ccrs.PlateCarree())
longrd = []
for i in range(lonsub.shape[0]):
    longrd.append(ax.plot(lonsub[i,:], latsub[i,:], linewidth=0.5, color='gray', transform=ccrs.PlateCarree()))
latgrd = []
for i in range(latsub.shape[1]):
    latgrd.append(ax.plot(lonsub[:,i], latsub[:,i], linewidth=0.5, color='gray', transform=ccrs.PlateCarree()))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.STATES)
ax.set_title(temp.GRIB_units, loc='right')
ax.set_title(temp.GRIB_name, loc='left')
ax.tick_params(bottom=True, left=True)
ax.set_yticks(np.arange(20, 60, 10))
# not sure why, but the longitude values seem to be 0:360, but the plotting is happening in -180:180 (?)
ax.set_xticks(np.arange(-135, -65, 15))
fig.suptitle("Plotting with 2D coordinate arrays")
cb = fig.colorbar(img, orientation='horizontal')
print(lonsub.min(), lonsub.max())
ax.set_extent([lonsub.min(), lonsub.max(), latsub.min(), latsub.max()])

fig.savefig("/Users/brianpm/Desktop/latlon_subset_D.png")

# Following the NCL examples, next we put the two methods of
# getting the region side by side
fig, ax = plt.subplots(figsize=(10,5), ncols=2, subplot_kw={"projection":ccrs.PlateCarree()})

# make sure we use the same colors for both:
cnorm = mpl.colors.Normalize(vmin=275, vmax=310)
img0 = ax[0].pcolormesh(lon2d, lat2d, temp_sub, norm=cnorm, transform=ccrs.PlateCarree())
img1 = ax[1].pcolormesh(lonsub, latsub, temp_sub2[0,:,:], norm=cnorm, transform=ccrs.PlateCarree())

ax[0].add_feature(cartopy.feature.STATES)
ax[1].add_feature(cartopy.feature.STATES)

# we didn't bother with setting extents, but now let's make sure they are the same
ax[0].set_extent([lonsub.min()-2, lonsub.max()+2, latsub.min()-2, latsub.max()+2])
ax[1].set_extent([lonsub.min()-2, lonsub.max()+2, latsub.min()-2, latsub.max()+2])



# put appropriate titles
ax[0].set_title("mask with np.where")
ax[1].set_title("subset array with indices")

fig.suptitle("The SupTitle Up Here")
cb = fig.colorbar(img1, ax=ax.ravel().tolist())
fig.savefig('/Users/brianpm/Desktop/latlon_subset_E.png')