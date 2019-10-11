#================================================#
#  vector_scalar.py
#  based on: gsn_vec_scal_1.ncl,
#            gsn_vec_scal_2.ncl,
#            gsn_vec_scal_3.ncl
#================================================#
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#=================================================#
# open file and read in data
#=================================================#
data_location = Path("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/cdf/")
data_file = data_location / "uvt.nc"

f1 = xr.open_dataset(data_file)
u = f1['U'][0,0,:,:]    # read in example data [2D only here]
v = f1['V'][0,0,:,:]
speed = (u**2 + v**2)**0.5
#=================================================#
# PLOT 1 - Vector field colored by a scalar.
#=================================================#
outfile_ext = "png"
outfilename = "gsn_vec_scal"
wks, ax = plt.subplots()
plot = ax.quiver(u,v,speed)

# you can change the relative size of the arrows
# with the scale kwarg, but it requires quite
# a bit of tuning.
# plot = ax.quiver(u,v,speed, scale=350)

# you can still concatenate strings with +:
wks.savefig("/Users/brianpm/Desktop/"+outfilename+"."+outfile_ext)

#=================================================#
# PLOT 2  -  Contour plot with vectors on top
#=================================================#
wks2, ax2 = plt.subplots()
plot2 = ax2.contourf(speed[10:30,20:40])   # contour the variable
plotV = ax2.quiver(u[10:30, 20:40], v[10:30, 20:40])
wks2.savefig("/Users/brianpm/Desktop/"+outfilename+"2."+outfile_ext)
#=================================================#
# Plot 3  -  Put it on a map
#=================================================#
wks3, ax3 = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
lon = f1['lon']
lat = f1['lat']
lons, lats = np.meshgrid(lon, lat)
plot3 = ax3.quiver(lons, lats, u, v, speed, transform=ccrs.PlateCarree())
ax3.set_title("Basic Vector/Scalar/Map Plot")
ax3.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
ax3.coastlines()
ax3.set_xticks(np.arange(-180, 180, 30))
ax3.set_yticks(np.arange(-90, 90, 30))
ax3.grid()
wks3.savefig("/Users/brianpm/Desktop/"+outfilename+"3."+outfile_ext)
