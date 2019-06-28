import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!  NEEDED FOR SAVING ON LINUX / COMMENT FOR NOTEBOOK
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point  # Not used here, but this is how to get rid of white lines at 0E
import numpy as np
import xarray as xr

from os.path import expanduser  # Not necessary once you put in correct paths.
home = expanduser("~")


def basic_map(data):
    fig, ax = plt.subplots(subplot_kw={'projection':ccrs.Geostationary()})
    lons, lats = np.meshgrid(data['lon'], data['lat'])
    im1 = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap='inferno')
    ax.coastlines()
    ax.set_title("Pincus Map")
    fig.colorbar(im1)
    fig.savefig(home+"/Desktop/my_fancy_map.png")


def mean_w_diff(lons, lats, data1, data2, case1, case2, var, plotname):
    difference = data2 - data1
    zmax = np.max([data1.max(), data2.max()])
    zmin = np.min([data1.min(), data2.min()])
    gs = gridspec.GridSpec(2, 4)  # defines the grid for subplots
    fig1 = plt.figure(figsize=(12, 6))  # sets the figure with size

    ax1 = plt.subplot(gs[0, :2], projection=ccrs.Robinson())
    ax2 = plt.subplot(gs[0, 2:], projection=ccrs.Robinson())
    ax3 = plt.subplot(gs[1, 1:3], projection=ccrs.Robinson())

    im1 = ax1.pcolormesh(lons, lats, data1, transform=ccrs.PlateCarree(), cmap='inferno', vmin=zmin, vmax=zmax)
    ax1.coastlines()
    ax1.set_title(case1)
    im2 = ax2.pcolormesh(lons, lats, data2, transform=ccrs.PlateCarree(), cmap='inferno', vmin=zmin, vmax=zmax)
    ax2.coastlines()
    ax2.set_title(case2)
    dmax = np.max([np.abs(difference.max()), np.abs(difference.min())])
    dmin = -dmax
    im3 = ax3.pcolormesh(lons, lats, difference, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=dmin, vmax=dmax)
    ax3.coastlines()
    fig1.subplots_adjust(right=0.8)
    cbar_ax = fig1.add_axes([0.85, 0.6, 0.025, 0.3])
    cbar = fig1.colorbar(im2, cax=cbar_ax)
    cbar.ax.set_ylabel(var)
    dbar_ax = fig1.add_axes([0.85, 0.1, 0.025, 0.3])
    dbar = fig1.colorbar(im3, cax=dbar_ax)
    dbar.ax.set_ylabel('difference')
    fig1.savefig(plotname)
    plt.close(fig1)


if __name__ == "__main__":
    # FILL THIS IN:
    path_to_nc = home+"/Desktop/test_nc_file.nc"
    ds = xr.open_dataset(path_to_nc)
    variable_name = "cltmmodis"
    X = ds[variable_name]
    if "time" in X.dims:
        X = X.mean(dim="time")
    basic_map(X)
    print(f"If you make it this far, check for {home+'/Desktop/my_fancy_map.png'}")
    print("Advanced homework: try to use mean_w_diff()")
