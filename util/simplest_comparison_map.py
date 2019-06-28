from pathlib import Path
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/brianpm/Code/blottr')
import blottr
import time
import argparse
import cartopy.crs as ccrs



def _load(f, v):
    dataset = xr.open_dataset(f, decode_cf=False)
    if "time_bnds" in dataset:
        tb = dataset['time_bnds'].mean(axis=1)
        dataset['time'].values = tb
        dataset = xr.decode_cf(dataset)
    else:
        dataset = xr.decode_cf(dataset)
    return dataset[v]


def plot(data, dataname, outname):
    lons, lats = np.meshgrid(data['lon'], data['lat'])
    vmax = np.max([data.max(), -data.min()])
    vmin = -vmax
    fig, ax = blottr.scalar_map(lons, lats, data, dataname, None, fig=None, ax=None,
                                **{'cmap':'coolwarm', 'vmax':vmax, 'vmin':vmin, 'subplot_kw':{'projection':ccrs.Robinson()}})

    fig.savefig(outname)


def _main(inputs):
    f1 = config['ncfile1']
    f2 = config['ncfile2']
    v1 = config['variable1']
    v2 = config['variable2']
    ploc = config['out_plot_dir']
    x = _load(f1, v1)
    y = _load(f2, v2)
    if 'slice1' in inputs:
        x = x.isel(time=slice(inputs['slice1'][0], inputs['slice1'][1]))
    if 'slice2' in inputs:
        y = y.isel(time=slice(inputs['slice1'][0], inputs['slice1'][1]))
    if 'dateslice1' in inputs:
        x = x.sel(time=slice(inputs['dateslice1'][0], inputs['dateslice1'][1]))
    if 'dateslice2' in inputs:
        y = y.sel(time=slice(inputs['dateslice1'][0], inputs['dateslice1'][1]))
    if 'time' in x.dims:
        x = x.mean(dim='time')
    if 'time' in y.dims:
        y = y.mean(dim='time')
    print(x)
    print(y)
    difference = y - x
    outputname = Path(ploc) / f"{v2}_minus_{v1}.pdf"
    plot(difference, f"{v2} - {v1}", outputname)



if __name__ == "__main__":
    """Provides CLI for running this simple map application."""
    start_time = time.time()  # Time before the operations start
    # Get the inputs from json file:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', help="json file with inputs.")
    args = parser.parse_args()  # (['-i', test])
    with open(args.filepath, 'r') as f:
        config = json.load(f)
    assert "ncfile1" in config
    assert "ncfile2" in config
    assert "variable1" in config
    assert "variable2" in config
    assert "out_plot_dir" in config
    _main(config)
    print(f"Total time: {time.time() - start_time} seconds.")
