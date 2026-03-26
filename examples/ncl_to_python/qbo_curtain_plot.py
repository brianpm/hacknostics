from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
import esmlab


# interpolate to pressure levels:
def pres_from_hybrid(psfc, hya, hyb, p0=100000.):
    """Return pressure field on hybrid-sigma coordinates, 
       assuming formula is 
       p = a(k)*p0 + b(k)*ps.
    """
    return hya*p0 + hyb*psfc


def lev_to_plev(data, ps, hyam, hybm, P0=100000., new_levels=None):
    """Interpolate data from hybrid-sigma levels to isobaric levels."""
    pressure = pres_from_hybrid(ps, hyam, hybm, P0)
    if new_levels is None:
        pnew = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]  # mandatory levels
    else:
        pnew = new_levels
    data_interp = data.interp(lev=pnew)
    return data_interp


def qbo_curtain_plot(udata):
    fig, ax = plt.subplots()
    cnorm = mpl.colors.TwoSlopeNorm(vmin=-50, vcenter=0.0, vmax=50)
    cnStep = 10
    levels = np.arange(-50, 50+cnStep, cnStep)
    times, levs = np.meshgrid(np.arange(len(udata['time'])), udata['lev'])
    img = ax.contourf(times, levs, udata.transpose("lev","time"), norm=cnorm, cmap=cc.cm.coolwarm, levels=levels)
    ax.invert_yaxis()
    ax.set_yscale("log") 
    ax.set_title("QBO Curtain Plot")
    fig.colorbar(img, ax=ax, shrink=0.3)
    return fig, ax

ds = xr.open_dataset("/Volumes/Glyph6TB/vert_res/vres_L128/vres_L128.cam.h0.2000-01.ncrcat.U.nc")
u = ds['U']
ps = ds["PS"]
hyam = ds["hyam"]
hybm = ds["hybm"]
wgt = np.cos(np.radians(ds['lat']))

uplev = lev_to_plev(u, ps, hyam, hybm)
smth2 = uplev.rolling(time=5,center=True).mean()
u_eq_smth2 = esmlab.weighted_mean(smth2.sel(lat=slice(-5,5)), dim=["lat"], weights=wgt.sel(lat=slice(-5,5))).mean(dim='lon')
fig04, ax04 = qbo_curtain_plot(u_eq_smth2)