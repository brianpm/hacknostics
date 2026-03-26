#!/usr/bin/env python
# coding: utf-8


# intraseasonal_varaince.py
# =========================

# # Fraction of variance in 20-100 day band
# 
# Plots maps of percent variance accounted for by intraseasonal time scale for CESM2 Large Ensemble.
# 
# info: https://www.cesm.ucar.edu/projects/community-projects/LENS2/
# 
# This makes maps that are consistent with the description in the MJO diagnostics paper: [doi: 10.1175/2008JCLI2731.1](https://doi.org/10.1175/2008JCLI2731.1)
# 
# ## implementation notes
# 
# - Uses time series files of daily means. 
# - Constructs a dictionary to figure out which macro/micro combinations are available.
# - Currently is only working on the historical runs, and uses 1850-1899
# - Currently makes plots for DJF
# - Uses GeoCAT's bandpass filter that is based on FFT. 

import multiprocessing as mp
import numpy as np
import xarray as xr
from pathlib import Path
import geocat.comp as gc
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
import cartopy.crs as ccrs


# In[18]:


def get_coord_bnds(crd):
    dcrd = crd[1]-crd[0]
    result = np.zeros(len(crd)+1)
    result[0:-1] = crd - 0.5*dcrd
    result[-1] = crd[-1]+0.5*dcrd
    return result


# In[3]:


def season_selector(dset, season):
    in_season = dset.time.dt.season == season
    return dset.sel(time=in_season)


def calc_variance_filtered(fld, season=None):
    '''Calculate total and bandpass filtered variance from daily field, optionally for a given season.'''
    x_sampling_freq = 1 # 1 sample per day
    cflow = (1/100)
    cfhigh = (1/20)
    x_bandpass = gc.fourier_filters.fourier_band_pass(fld.sel(lat=slice(-30,30)), 1, cflow, cfhigh, time_axis=0)
    if season is not None:
        x_seas = season_selector(fld, season)
        x_bandpass_seas = season_selector(x_bandpass, season)
    else:
        x_seas = fld
        x_bandpass_seas = x_bandpass
    x_totalvariance = x_seas.sel(lat=slice(-30,30)).var(dim='time').compute()    
    x_bandpass_variance = x_bandpass_seas.sel(lat=slice(-30,30)).var(dim='time').compute()
    return x_totalvariance, x_bandpass_variance
    


# In[41]:


def make_the_plot(vt, vf, titlestr=None, oname=None):

    lon_bnds = get_coord_bnds(vt.lon)
    lat_bnds = get_coord_bnds(vt.lat)
    fig, ax = plt.subplots(figsize=(12,4), subplot_kw={"projection":ccrs.PlateCarree()})

    color_norm = mpl.colors.Normalize(vmin=0.1, vmax=0.5)
    color_map = cc.cm.CET_L19
    img = ax.pcolormesh(lon_bnds, lat_bnds, vf/vt, rasterized=True, transform=ccrs.PlateCarree(), cmap=color_map, norm=color_norm)
    ax.contour(vt.lon, vt.lat, vt, colors='gray')
    ax.coastlines()
    ax.set_title(titlestr, loc='left')
    ax.text(0., -.2, "Lines: Total variance", rotation=0, transform=ax.transAxes)
    fig.colorbar(img, shrink=0.4, label="Fraction of variance\n in 20-100 day band")
    if oname is None:
        plt.show()
    else:
        fig.savefig(oname, bbox_inches='tight', dpi=200)
    return fig, ax


def construct_catalog(data_loc):
    """Constructs a dictionary of dictionaries to point to each member of the CESM2 Large Ensemble.
    
    Simpler than intake-esm, but requires knowing where files are and how things are named and organized.
    
    """
    # get all the MACRO / MICRO names:
    all_files = list( (data_loc / "TS").glob("*.nc"))
    print(f"For TS there are a total of {len(all_files)} files.")

    all_files_names = pd.Series([f.name for f in all_files])
    all_files_names_split = all_files_names.str.split(".", expand=True)
    all_files_names_split.columns = ["coupling", "version", "compset", "resolution", "macro", "micro", "ostream", "hstream", "variable", "time_range", "suffix"]
    all_compset_names = list(set(all_files_names_split['compset']))
    print(f"All compset names: {all_compset_names}")
    all_macro = list(set(all_files_names_split["macro"]))
    all_micro = list(set(all_files_names_split["micro"]))
    print(f"All the macro options: {all_macro}")
    print(f"All the micro options: {all_micro}")
    # dictionary that says which macro are included for each case name:
    cat = {}
    for i in all_compset_names:
        cat[i] = set( all_files_names_split[all_files_names_split['compset'] == i]['macro'] )  # all the macro that are available for this compset name
        cattmp = {}
        for j in cat[i]:
            cattmp[j] = list(set(all_files_names_split[(all_files_names_split['compset'] == 'BSSP370cmip6') & (all_files_names_split['macro'] == j)]['micro']))
        cat[i] = cattmp
    # and that is the catalog. We know the location of the files, and now we can use macro and micro to build the full paths
    return cat


def forworker(files, VAR, fns, odir):
    ds = xr.open_mfdataset(files).sel(time=slice("1849-12-31", "1899-12-31"))
    data = ds[VAR]
    vartotal, varfilter =  calc_variance_filtered(data, season="DJF")
    #print("variance done")
    # make the plot
    plot_file_name = odir / f"{fns}.variancefilter.png"
    # title for plot:
    plot_title = f"{fns} : {VAR}"
    # render and save the plot
    fig, ax = make_the_plot(vartotal, varfilter, titlestr=plot_title, oname=plot_file_name)
    #print("\t ... plot done")

    
def main(cat, data_loc, compset, odir):
    """Uses input information to make the plots of fraction of intraseasonal variance.
    
    cat      : "catalog" of cases, used to search for time series files
    
    data_loc : root directory of the data, start of the search for files
    
    compset  : the compset that is going to be selected for processing (this could be generalized to do all compsets)
    
    odir     : ouput directory, where plots will be written
    
    Searches for files for the list of variables ["FLUT", "PRECT", "U200", "U850"], 
    gathers information into a list of arguments needed to run `forworker`,
    and sets up a multiprocessing pool to run all the jobs in parallel. 
    
    """

    # COMPSET = compset # "BHISTcmip6"
    number_of_cpu = mp.cpu_count()
    print(f"`main` thinks we have {number_of_cpu} to use for the pool.")
    # constructs list with arguments to pass to `forworker`
    list_of_arguments = []
    for VAR in ["FLUT", "PRECT", "U200", "U850"]:
        for macro in cat[compset]:
            for micro in cat[compset][macro]:
                file_namer_str = f"{compset}.{macro}.{micro}.{VAR}.1850-1999"
                find_files_str = f"b.e21.{compset}.f09_g17.{macro}.{micro}.cam.h1.{VAR}.*.nc"
                files = sorted(list( (data_loc / VAR).glob(find_files_str)))
                list_of_arguments.append( (files, VAR, file_namer_str, odir) )

    with mp.Pool(processes=number_of_cpu) as p:
        result = p.starmap(forworker, list_of_arguments)


if __name__ == "__main__":
    data_loc = Path("/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1")
    
    cat = construct_catalog(data_loc)
    main(cat, data_loc, "BHISTcmip6", Path("/glade/scratch/brianpm/"))
