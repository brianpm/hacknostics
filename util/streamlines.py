# =================================================#
# streamlines.py
# based on: gsn_stream_1.ncl
#           gsn_stream_2.ncl
#           gsn_stream_3.ncl
#           gsn_stream_4.ncl
# =================================================#
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#=================================================#
# open file and read in data
#=================================================#
data_location = Path("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/cdf/")
data_file = data_location / "uvt.nc"

f1 = xr.open_dataset(data_file)
u = f1['U'][0,0,:,:].values    # read in example data [2D only here]
v = f1['V'][0,0,:,:].values
speed = (u**2 + v**2)**0.5
# NOTE: I changed these to numpy arrays immediately, since we don't need metadata (doesn't really matter)
# =================================================#
# PLOT 1
# =================================================#

shp = u.shape
x = np.arange(shp[1])
y = np.arange(shp[0])
outfile_ext = "png"
outfilename = "gsn_stream"
wks, ax = plt.subplots()
plot = ax.streamplot(x, y, u, v, density=3, linewidth=1, arrowsize=1)  # increase density to look more like NCL example
ax.set_title("Example of a streamline plot")
wks.savefig("/Users/brianpm/Desktop/"+outfilename+"."+outfile_ext)

# =================================================#
# PLOT 2  - change color & tweak size
# =================================================#
outfilename = "gsn_stream2"
wks, ax = plt.subplots()
plot = ax.streamplot(x, y, u, v, density=3, linewidth=1.5, arrowsize=1, color='orange')  # increase density to look more like NCL example
ax.set_title("Example of a streamline plot")
wks.savefig("/Users/brianpm/Desktop/"+outfilename+"."+outfile_ext)


# =================================================#
# PLOT 3  - color by scalar field
# =================================================#
outfilename = "gsn_stream3"
wks, ax = plt.subplots()
# let's get fancy and make the colors go +/- 1 standard deviation:
speedmean = np.mean(speed)
speedsigma = np.std(speed)
print(speedsigma)
norm = colors.DivergingNorm(vmin=speedmean-speedsigma, vcenter=speedmean, vmax=speedmean+speedsigma)
plot = ax.streamplot(x, y, u, v, color=speed, density=3, linewidth=1, arrowsize=0.5, cmap='cividis', norm=norm)  # increase density to look more like NCL example
ax.set_title("Example of a streamline plot")
wks.savefig("/Users/brianpm/Desktop/"+outfilename+"."+outfile_ext)

#
# PLOT 4  -  Put it on a map.
#
outfilename = "gsn_stream4"
lon = f1['lon'].values
lat = f1['lat'].values
lons, lats = np.meshgrid(lon, lat)
wks, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
plot3 = ax.streamplot(lons, lats, u, v, color=speed, density=3, linewidth=1, arrowsize=0.5, cmap='cividis', norm=norm, transform=ccrs.PlateCarree())
ax.set_title("Basic Streamline/Scalar/Map Plot")
# ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
ax.coastlines()
wks.colorbar(plot3.lines, shrink=0.4)  # NOTE: have to give the lines attribute b/c plot3 isn't mappable.
wks.savefig("/Users/brianpm/Desktop/"+outfilename+"."+outfile_ext)

