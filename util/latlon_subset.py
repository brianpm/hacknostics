
# latlon_subset_1.py
# Based on latlon_subset_1.ncl
#----------------------------------------------------------------------
# Concepts illustrated:
#   - Using coordinate values to extract a lat/lon region
#   - Subsetting a rectilinear grid
#   - Drawing a lat/lon grid using cartopy
#   - Adding markers to a map
#   - Replicating NCL's lonFlip to convert 0 to 360 longitudes to -180 to 180
#   - Zooming in on a particular area on a map
#----------------------------------------------------------------------
# The data file for this example can be downloaded from
# http://www.ncl.ucar.edu/Applications/Data/#cdf
#
# For an example of subsetting data represented by 2D lat/lon arrays,
# see latlon_subset_2.ncl and the "getind_latlon2d" function.
#----------------------------------------------------------------------
#
# NCL -> Python Notes
# I try to reproduce the figures shown on the NCL site,
# but there are discrepancies between the NCL code and the figures.
# I also try to save the figures as png files with names like those
# on the website, but the examples do not show that step.


# Load external packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

# define the directory where output plots will go:
# (better to use a path object, but a string is ok, and NCL-like)
out_dir = "/Users/brianpm/Desktop/"

# Read in netCDF file.
a = xr.open_dataset("/Users/brianpm/Dropbox/DataTemporary/B1850_c201_CTL.cam.h0.TS.ncrcat.nc")
ts = a['TS'].isel(time=0)

# Print information about ts
print(ts.dims)       # 192 x 288

# a little preparation:
lat = a['lat']
lon = a['lon']
lons, lats = np.meshgrid(lon, lat)

# NCL: wks = gsn_open_wks("x11","latlon_subset")
# You can think of "wks" as the figure object that holds each plot in an axes object
# They can be constructed separately, or using a shortcut to make both at once

# Plot resources in matplotlib are set using a series of methods on the figure and axes
# gsn_csm_contour_map -> use cartopy to turn ax to GeoAxes with projection

wks, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

#  res@cnFillOn              = True     # turn on contour fill
CS = ax.contourf(lons, lats, ts, cmap='nipy_spectral', transform=ccrs.PlateCarree ())

#  res@cnLinesOn             = False    # turn off contour lines
# --> overlay contour
# CS2 = ax.contour(lons, lats, CS, levels=CS.levels[::2], colors='k', transform=ccrs.PlateCarree())

#  res@cnLineLabelsOn        = False    # turn off contour line labels


#  res@tiMainString          = "Plotting full range of data"
wks.suptitle("Plotting full range of data")
ax.set_title("Surface Temperature", loc='left')
ax.set_title("K", loc='right')
ax.coastlines()
cb = wks.colorbar(CS, ax=ax, orientation='horizontal', fraction=0.1)
#---Draw plot of full data
wks.savefig(out_dir+"latlon_subset_1_1.png")

# ----------------------------------------------------------------------
# Zoom in on area of interest using coordinate subscripting 
# via special syntax "{" and "}". You can only use this syntax 
# when your data is represented by one-dimensional coordinate arrays.
# (see "printVarSummary" output)
# ----------------------------------------------------------------------
lat_min = 20
lat_max = 50
lon_min = 60
lon_max = 120
ts_sub  = ts.sel(lat=slice(lat_min,lat_max), lon=slice(lon_min,lon_max))
print(ts_sub.dims)        # 32 x 49

wks2, ax2 = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})

tiMainString = "Plotting lat/lon subset of data"
#  res@pmTitleZone           = 4          # Moves title down
#  res@gsnAddCyclic          = False      # This is very important!

# --- Draw plot of subsetted data
#  plot = gsn_csm_contour_map(wks,ts_sub,res)
lons2, lats2 = np.meshgrid(ts_sub['lon'], ts_sub['lat'])
CS2 = ax2.contourf(lons2, lats2, ts_sub, cmap='nipy_spectral', transform=ccrs.PlateCarree())
ax2.coastlines()
# # --- Zoom in on map and add some lines and markers to see the grid and lat/lon points.
#   res@gsnDraw               = False 
#   res@gsnFrame              = False
#   res@mpMinLatF             = min(ts_sub&lat)-2
#   res@mpMaxLatF             = max(ts_sub&lat)+2
#   res@mpMinLonF             = min(ts_sub&lon)-2
#   res@mpMaxLonF             = max(ts_sub&lon)+2
ax2.set_extent((min(ts_sub['lon']) - 2, 
                max(ts_sub['lon']) + 2, 
                min(ts_sub['lat']) - 2, 
                max(ts_sub['lat']) + 2))

#   res@mpCenterLonF          = (res@mpMinLonF + res@mpMaxLonF) / 2.
#   res@pmTickMarkDisplayMode = "Always"  # nicer map tickmarks



cb2 = wks2.colorbar(CS2, ax=ax2, orientation='horizontal')
# plot = gsn_csm_contour_map(wks,ts_sub,res)

# #---Attach lat/lon grid lines of subsetted data.
#   gsres                  = True
#   gsres@gsnCoordsAsLines = True
#   gsres@gsnCoordsAttach  = True
#   gsn_coordinates(wks,plot,ts_sub,gsres)
# NOTE: it is not clear to me what feature these lines are supposed to demonstrate,
#       so I just include one way to add some grid lines:
ax2.gridlines()

# --- Attach two markers showing two lat,lon corners of interest
# NOTE: there are lots of ways you could add markers, and probably depends on how many you need to add.
# here I'll make a list [] of tuples () and add each using a loop.
points = [(lon_min, lat_min),
          (lon_max, lat_max)]
[ax2.plot(p[0], p[1], marker='o', color='black', markersize=15) for p in points]
#   mkres               = True
#   mkres@gsMarkerIndex = 16     # filled dot
#   mkres@gsMarkerColor = "black"
#   mkres@gsMarkerSizeF = 15
#   mkid1 = gsn_add_polymarker(wks,plot,(/lon_min,lon_max/),(/lat_min,lat_max/),mkres)

# #---Drawing the plot will draw the attached lines and markers.
#   draw(plot)
#   frame(wks)

# Decoration
wks2.suptitle(tiMainString, weight='bold')
# NOTE: NCL's 'gsn' interface does some work behind the scenes,
# we can easily replicate some of it. For example, we can always
# check if there's a long_name and units attribute associated with
# the data.
def add_gsn_titles(ax=ax, data=ts_sub):
    if hasattr(data, "long_name"):
        ax.set_title(data.long_name, loc='left')
    if hasattr(data, 'units'):
        ax.set_title(data.units, loc='right')

add_gsn_titles(ax=ax2, data=ts_sub)
# NOTE: the ncl example shows filled continents, but not the code to do it.
#       We can use cartopy to do this, but you might need internet access
#       to download the NaturalEarth data that it uses.
ax2.add_feature(cartopy.feature.LAND)

wks2.savefig(out_dir+"latlon_subset_1_3_lg.png")


# NOTE: Since the following example does not demonstrate
#       any new plotting features, I do not replicate the plot,
#       but I show how to define and apply lonFlip.


# lonFlip() is a specialty function in NCL. We can replicate
# its functionality pretty easily for xarray data structures.
# Not sure whether NCL's version has anything more sophisticated.

def lonFlip(data):
    # NOTE: this assumes global values
    tmplon = data['lon']
    tmpdata = data.roll(lon=len(lon) // 2, roll_coords=True)
    lonroll = tmpdata['lon'].values
    if tmplon.min() >= 0:
        # flip to -180:180
        tmpdata['lon'].values = np.where(lonroll >= 180, lonroll - 360, lonroll)
    else:
        # flip from -180:180 to 0:360
        tmpdata['lon'].values = ((lonroll + 360) % 360)
    return tmpdata


# To subscript with longitude values < 0, you must "flip" the longitudes
print(f"Original lon values: MIN: {ts['lon'].min().values.item()}, MAX: {ts['lon'].max().values.item()}, in {len(lon)} steps.")
ts = lonFlip(ts)     # convert from 0:360 to -180:180
print(f"Flipped lon values: MIN: {ts['lon'].min().values.item()}, MAX: {ts['lon'].max().values.item()}, in {len(lon)} steps.")

