# =================================================#
# gsn_vector_1.ncl
# =================================================#
#
# This file is loaded by default in NCL V6.2.0 and newer
# load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
# =================================================#
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

data_location = Path("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/cdf/")
data_file = data_location / "uvt.nc"

f = xr.open_dataset(data_file)
u = f["U"][0,0,:,:]    # read in example data [2D only here]
v = f["V"][0,0,:,:]
# =================================================#
# create plots
# =================================================#
wks, ax = plt.subplots(figsize=(10,10), constrained_layout=True)
outfile_extension = "png"
outfile_name = "gsn_vector"
outfile_dir = Path("/Users/brianpm/Desktop/")
tiMainString     = "Basic Vector Plot"
vcRefMagnitudeF  = 5.0                       # add a reference vector
vcRefLengthF     = 0.045                     # what the ref length is
vcGlyphStyle     = "CurlyVector"             # turn on curly vectors

plot = ax.quiver(u[::2, ::2], v[::2, ::2])
# Getting a "reference vector" is not as nice as it should be.
# Positioning it outside the axes isn't working correctly for me.
# Getting a box around it is also not working.
# Both of these could be fixed with a custom function that would add a box and put the quiverkey inside it;
# this is exactly the kind of functionality that "we" could contribute to Matplotlib.
qk = ax.quiverkey(plot, 0.8, 0.03, 20, '20 m/s', coordinates='axes', labelpos='S', color='red')
wks.savefig(outfile_dir / ".".join([outfile_name , outfile_extension]) , bbox_inches='tight', bbox_extra_artists=(qk,))
print(f'DONE WITH FIRST EXAMPLE: {outfile_dir / ".".join([outfile_name , outfile_extension])}')

# NOTE: As far as I can tell, quiver can only use straight arrows, not "CurlyVector" style.
# But we can try to use streamplot:

# Need the x & y grid
# to make it look better, increase density, scale lines/arrows
shp = u.shape
yvec = np.arange(0, shp[0]) # lat
xvec = np.arange(0, shp[1]) # lon
wks, ax = plt.subplots()
outfile_extension = "png"
outfile_name = "gsn_vectorStreamPlot"
outfile_dir = Path("/Users/brianpm/Desktop/")
ax.streamplot(xvec, yvec, u, v, linewidth=0.5, arrowsize=0.5, density=4)
wks.savefig(outfile_dir / ".".join([outfile_name , outfile_extension]))


# That looks fairly similar to the NCL example
# compare with using a projection to geographic space
lat = f['lat']
lon = f['lon']
lons, lats = np.meshgrid(lon, lat)
wks, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
outfile_extension = "png"
outfile_name = "gsn_vectorStreamPlotCartopy"
outfile_dir = Path("/Users/brianpm/Desktop/")
ax.streamplot(lons, lats, u.values, v.values, density=3, linewidth=0.5, arrowsize=0.5)
ax.coastlines()
wks.savefig(outfile_dir / ".".join([outfile_name , outfile_extension]) )

# When we switch from quiver to streamplot, the idea of a "reference vector"
# does not hold, so we could just color by the magnitude.
wks, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
outfile_extension = "png"
outfile_name = "gsn_vectorStreamPlotCartopyMagnitude"
outfile_dir = Path("/Users/brianpm/Desktop/")
magnitude = (u ** 2 + v ** 2) ** 0.5
ax.streamplot(lons, lats, u.values, v.values, density=3, linewidth=0.5, arrowsize=0.5, color=magnitude.values)
ax.coastlines()
wks.savefig(outfile_dir / ".".join([outfile_name , outfile_extension]) )
