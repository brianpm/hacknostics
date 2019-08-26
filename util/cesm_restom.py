#!/usr/bin/env python

from pathlib import Path
import numpy as np
import xarray as xr
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

# local functions that do the calculations.
# GOTO bottom for the CLI program.

def _fil_parse(p_in):
    '''Parse a string into a file or glob.'''
    p = Path(p_in)
    isfil = p.is_file()
    logging.debug(f"Checked if input is a file, result: {isfil}")
    if not isfil:
        directory = p.parent
        g = p.name
        logging.debug(f"We now assume the director is: {directory} \n and glob is {g}")
        fils = sorted(list(directory.glob(g)))
        logging.debug(f"We will use fils list of length: {len(fils)}")
    else:
        fils = p
    return fils


def _load_data(inputfils):
    f = sorted(list(inputfils))
    logging.debug(f"[_load_data] expecting a list. Has length {len(f)} ")
    if len(f) == 0:
        raise IOError("No files are found.")
    elif len(f) == 1:
        ds = xr.open_dataset(f[0])
    else:
        ds = xr.open_mfdataset(f, combine='by_coords')
    return ds

def global_average(fld, gw):
    """
    Calculate the global average.
    :param fld: Array-like with named dimensions and a "mean" method (i.e. xarray).
    :param gw: Weights.
    :return: Global average; performs time-average if there is a time dimension.
    """
    if "time" in fld.dims:
        return wgt_areaave_xr(fld, gw).mean(dim="time")
    else:
        return wgt_areaave_xr(fld, gw)


def wgt_areaave_xr(data, yweight, xweight=1.0):
    """
    A xarray implementation of NCL's wgt_areaave. Preserves metadata.
    Assumes data has dims (..., ydim, xdim).
    :param data: And array-like object, assumes axis = 1 is longitude, axis = 2 is latitude (to be fixed)
    :param yweight: latitude weights, array like, same size as data's latitude axis
    :param xweight: optional longitude weight, defaults to 1.
    :return: array of weighted average over latitude and longitude.
    """
    if type(xweight) == float or type(xweight) == int:
        xlength = data.shape[-1]
        # logging.debug('xlength'+str(xlength))
        # numpy: xw = np.ones(xlength) * xweight
        xw = xr.DataArray(np.ones(xlength))
    else:
        xw = xweight
    # logging.debug(xw)
    latdim = list(data.coords).index("lat")
    londim = list(data.coords).index("lon")

    # normalized weights:
    ywnorm = yweight / yweight.sum()
    xwnorm = (
        xw / xw.sum()
    )  ##**** CHECK WHETHER mfdataset makes gw(time,lat) break this method.

    xdimname = data.dims[-1]
    ydimname = data.dims[-2]

    # apply average in each direction:
    if not (xw == 1).all().values.all():
        a = (data * xwnorm).sum(dim=xdimname)
    else:
        # just apply a regular mean
        a = data.mean(dim=xdimname)

    a = (a * ywnorm).sum(dim=ydimname)
    return a



def calc_lat_weight(lat):
    return np.cos(np.radians(lat))


def get_cesm_weight(ds):
    """
    Get latitude weights for CESM data set.
    :param ds: and xr dataset (or dataarray)
    :return: weights(lat). Will try to use gw variable and correct for multifile datasets. If not there, then cos(lat) as np array.
    """
    # latitude weighting:
    if "gw" in ds:
        gw = ds["gw"]
        # if it comes out 2d things get broken.
        if any([x == "time" for x in gw.dims]):
            print("We have a time,lat version of gw. Fixing.")
            gw = gw.isel(time=0, drop=True)
    else:
        gw = calc_lat_weight(ds["lat"])  # metadata will be preserved, so slice is okay
    return gw


def _calculate(ds):
    # check for all available data:
    have_fsnt = "FSNT" in ds
    have_flnt = "FLNT" in ds
    have_fsntoa = "FSNTOA" in ds

    wgt = get_cesm_weight(ds)

    if have_flnt:
        flnt_timeseries = wgt_areaave_xr(ds['FLNT'], wgt)
    if have_fsnt:
        fsnt_timeseries = wgt_areaave_xr(ds['FSNT'], wgt)
    if have_fsntoa:
        fsntoa_timeseries = wgt_areaave_xr(ds['FSNTOA'], wgt)
    if have_fsnt and have_flnt:
        restom = (fsnt_timeseries - flnt_timeseries).mean()
    else:
        restom = "missing"
    if have_fsntoa and have_flnt:
        restoa = (fsntoa_timeseries - flnt_timeseries).mean()
    else:
        restoa = "missing"
    return restom, restoa



if __name__ == "__main__":
    # A CLI that just calculates the TOA imbalance from an input dataset.
    # Input defined by a glob pattern or a filename.
    # Loaded with xarray.
    # Look for FSNT, FLNT, FSNTOA (Add up/down components as backup?)
    # For now, just print the answer to output.

    parser = argparse.ArgumentParser(description="Calculate TOA imblance of supplied CESM dataset.")
    parser.add_argument('Path', metavar='path', nargs='*', help='The path to the data.')
    # the nargs='*' allows for shell globbing, but means we alwasy get a list;
    # --> If list, then if the length is 1, just parse that string,
    #     otherwise assume it is the full list of files.
    args = parser.parse_args()
    input_path = args.Path
    # now let's find out if it is a file (if not, assumes a glob.)
    logging.debug(f"type of input_path is {type(input_path)}")
    if isinstance(input_path, list):
        # assume this is a shell glob
        logging.debug(f"Length of input list is {len(input_path)}")
        fils = input_path
        if len(fils) == 1:
            fils = _fil_parse(fils[0])
    else:
        fils = _fil_parse(input_path)

    data = _load_data(fils)
    calctop, calctoa = _calculate(data)
    print(f"Identified {len(fils)} files to use.")
    print(f"Top-of-model residual: {calctop.values.item()} \n Top-of-atmosphere residual: {calctoa.values.item()}")
