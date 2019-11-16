import xarray as xr

from windspharm.xarray import VectorWind

f = xr.open_dataset("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/cdf/uv300.nc")

u = f["U"]
v = f["V"]
w = VectorWind(u, v)
## VERY IMPORTANT: VectorWind apparently reverses latitude to be decreasing (90 to -90)
vort, div = w.vrtdiv()  # Relative vorticity and horizontal divergence.
sf, vp = w.sfvp()  # The streamfunction and velocity potential respectively.
uchi, vchi, upsi, vpsi = w.helmholtz()


# plot the results
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
fig, ax = plt.subplots(figsize=(12,12), nrows=2, subplot_kw={"projection":ccrs.PlateCarree()}, constrained_layout=True)
N = mpl.colors.Normalize(vmin=-8e6, vmax=8e6)
x,y = np.meshgrid(f['lon'], f['lat'])
im0 = ax[0].contourf(x, y, vp[0,::-1,:], norm=N, transform=ccrs.PlateCarree(), cmap='Spectral_r')
ax[0].quiver(x, y, uchi[0,::-1,:].values, vchi[0,::-1,:].values, transform=ccrs.PlateCarree(), regrid_shape=25)
ax[0].coastlines()
ax[0].set_title("Velocity Potential, Divergent Wind", loc='left')
ax[0].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[0].set_xticks(np.arange(-180, 180+30, 30))
fig.colorbar(im0, ax=ax[0])

im1 = ax[1].contourf(x, y, sf[0,::-1,:], 25, transform=ccrs.PlateCarree(), cmap='Spectral_r')
ax[1].quiver(x, y, upsi[0,::-1,:].values, vpsi[0,::-1,:].values, transform=ccrs.PlateCarree(), regrid_shape=25)
ax[1].coastlines()
ax[1].set_title("Streamfunction, Rotational Wind")
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_xticks(np.arange(-180, 180+30, 30))
fig.colorbar(im1, ax=ax[1])
fig.savefig("/Users/brianpm/Desktop/wind_3.png")
