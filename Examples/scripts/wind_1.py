import xarray as xr

f = xr.open_dataset("/Users/brianpm/Documents/www.ncl.ucar.edu/Applications/Data/cdf/uv300.nc")

u = f["U"]
v = f["V"]
print(u.shape)
# div = uv2dvG_Wrap(u,v)                ; u,v ==> divergence
# Computes the divergence using spherical harmonics given the
# u and v wind components on a gaussian grid (retains metadata).

# NOTE: We could try to start at a lower level, and use
#       scipy's spherical harmonics functions and derive
#       the winds.
#       Instead, let's take the easy way and use Andrew Dawson's
#       windspharm package:
#       https://ajdawson.github.io/windspharm/latest/
#       Dawson, A., 2016.
#       Windspharm: A High-Level Library for Global Wind Field Computations
#       Using Spherical Harmonics.
#       Journal of Open Research Software, 4(1), p.e31.
#       DOI: http://doi.org/10.5334/jors.129
#
#       install: conda install -c conda-forge windspharm

from windspharm.xarray import VectorWind

# Create a VectorWind instance to handle the computation of streamfunction and
# velocity potential.
w = VectorWind(u, v)  # there's a keyword argument gridtyp=['regular','gaussian'] but only for 'standard' not 'xarray'
## VERY IMPORTANT: VectorWind apparently reverses latitude to be decreasing (90 to -90)

# calculate divergence
# div = w.divergence()  # optional truncation can be supplied

# divergent wind components
# uchi, vchi = w.irrotationalcomponent()

# both divergent and rotational:
uchi, vchi, upsi, vpsi = w.helmholtz()

print(uchi[0,:,:])

# plot the results
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
fig, ax = plt.subplots(nrows=2, subplot_kw={"projection":ccrs.PlateCarree()}, constrained_layout=True)
x,y = np.meshgrid(f['lon'], f['lat'])  # if using orig, reverse latitude below / otherwise use coords from w.
ax[0].quiver(x, y, uchi[0,::-1,:].values, vchi[0,:,:].values, transform=ccrs.PlateCarree(), regrid_shape=25)
ax[0].coastlines()
ax[0].set_title("Divergent Wind", loc='left')
ax[0].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[0].set_xticks(np.arange(-180, 180+30, 30))

ax[1].quiver(x, y, upsi[0,::-1,:].values, vpsi[0,:,:].values, transform=ccrs.PlateCarree(), regrid_shape=25)
ax[1].coastlines()
ax[1].set_title("Rotational Wind")
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_xticks(np.arange(-180, 180+30, 30))
fig.savefig("/Users/brianpm/Desktop/wind_1.png")