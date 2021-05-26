#! /usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats  # for easy linear regression and testing

# GOAL: replace the "Tables" set in AMWG
#	Set Description
#   1 Tables of ANN, DJF, JJA, global and regional means and RMSE.
#
# STRATEGY:
# I think the right solution is to generate one CSV (or other?) file that
# contains all of the data.
# So we need:
# - a function that would produces the data, and 
# - then call a function that adds the data to a file
# - another function(module?) that uses the file to produce a "web page"

# IMPLEMENTATION:
# - assume that we will have time series of global averages already ... that should be done ahead of time
# - given a variable or file for a variable (equivalent), we will calculate the all-time, DJF, JJA, MAM, SON
#   + mean
#   + standard error of the mean
#     -- 95% confidence interval for the mean, estimated by:
#     ---- CI95 = mean + (SE * 1.96)
#     ---- CI05 = mean - (SE * 1.96)
#   + standard deviation
# AMWG also includes the RMSE b/c it is comparing two things, but I will put that off for now.

# DETAIL: we use python's type hinting as much as possible

# in future, provide option to do multiple domains
# They use 4 pre-defined domains:
domains = {"global": (0, 360, -90, 90),
           "tropics": (0, 360, -20, 20),
           "southern": (0, 360, -90, -20),
           "northern": (0, 360, 20, 90)}

# and then in time it is DJF JJA ANN

# within each domain and season
# the result is just a table of
# VARIABLE-NAME, RUN VALUE, OBS VALUE, RUN-OBS, RMSE

def main(inargs : dict):
    """Main function goes through series of steps:
    - uses inargs['File'] and inargs['Variable'] load the data
    - Determine whether there are spatial dims; if yes, do global average (TODO: regional option)
    - Apply annual average (TODO: add seasonal here)
    - calculates the statistics
      + mean
      + sample size
      + standard deviation
      + standard error of the mean
      + 5/95% confidence interval of the mean
      + linear trend
      + p-value of linear trend
    - puts statistics into a CSV file
    - generates simple HTML that can display the data
    """
    data = _load_data(inargs.File, inargs.Variable)
    if hasattr(data, 'units'):
        unit_str = data.units
    else:
        unit_str = '--'
    # we should check if we need to do area averaging:
    if len(data.dims) > 1:
        # flags that we have spatial dimensions
        # Note: that could be 'lev' which should trigger different behavior
        # Note: we should be able to handle (lat, lon) or (ncol,) cases, at least
        data = _spatial_average(data)  # changes data "in place"
    # In order to get correct statistics, average to annual or seasonal
    data = data.groupby('time.year').mean(dim='time') # this should be fast b/c time series should be in memory
                                                      # NOTE: data will now have a 'year' dimension instead of 'time'
    # Now that data is (time,), we can do our simple stats:
    data_mean = data.mean()
    data_sample = len(data)
    data_std = data.std()
    data_sem = data_std / data_sample
    data_ci = data_sem * 1.96  # https://en.wikipedia.org/wiki/Standard_error
    data_trend = stats.linregress(data.year, data.values)
    # These get written to our output file:
    # create a dataframe:
    cols = ['variable', 'unit', 'mean', 'sample size', 'standard dev.', 'standard error', '95% CI', 'trend', 'trend p-value']
    row_values = [inargs.Variable, unit_str, data_mean.data.item(), data_sample, data_std.data.item(), data_sem.data.item(), data_ci.data.item(), 
    f'{data_trend.intercept : 0.3f} + {data_trend.slope : 0.3f} t', data_trend.pvalue]
    dfentries = {c:[row_values[i]] for i,c in enumerate(cols)}
    df = pd.DataFrame(dfentries)
    print(df)
    # check if the output file exists:
    ofil = Path(inargs.Output)
    if ofil.is_file():
        df.to_csv(ofil, mode='a', header=False, index=False)
    else:
        df.to_csv(ofil, header=cols, index=False)
    # last step is to write the html file; overwrites previous version since we're supposed to be adding to it
    _write_html(ofil, Path(inargs.TableFile))


def _load_data(dataloc : str, varname : str) -> xr.DataArray:
    ds = xr.open_dataset(dataloc)
    return ds[varname]


def _spatial_average(indata : xr.DataArray) -> xr.DataArray:
    assert 'lev' not in indata.coords
    assert 'ilev' not in indata.coords
    if 'lat' in indata.coords:
        weights = np.cos(np.deg2rad(indata.lat))
        weights.name = "weights"
    elif 'ncol' in indata.coords:
        print("WARNING: We need a way to get area variable. Using equal weights.")
        weights = xr.DataArray(1.)
        weights.name = "weights"
    else:
        weights = xr.DataArray(1.)
        weights.name = "weights"
    weighted = indata.weighted(weights)
    # we want to average over all non-time dimensions
    avgdims = [dim for dim in indata.dims if dim != 'time']
    return weighted.mean(dim=avgdims)
    

def _write_html(f : Path, out : Path):
    print(out)
    print(type(out))
    df = pd.read_csv(f)
    print(df)
    html = df.to_html(index=False, border=1, justify='center', float_format='{:,.3f}'.format)  # should return string
    preamble = f"""<html><head></head><body><h1>{f.stem}<h1>"""
    ending = """</body></html>"""
    with open(out, 'w') as hfil:
        hfil.write(preamble)
        hfil.write(html)
        hfil.write(ending)
    

#
# --- okay, now just have the CLI stuff:
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate global mean diagnostics table entry.')
    parser.add_argument('File', type=Path, help='path of file')
    parser.add_argument('Variable', type=str, help='Variable name within File')
    parser.add_argument('Output', type=Path, help='write output to this file')
    parser.add_argument('TableFile', type=Path, help='the HTML file')
    args = parser.parse_args()
    main(args)

