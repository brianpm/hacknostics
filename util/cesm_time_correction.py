import xarray as xr

print(f"xarray version {xr.__version__}")

def cesm_correct_time(ds):
    """Given a Dataset, check for time_bnds,
       and use avg(time_bnds) to replace the time coordinate.

       Purpose is to center the timestamp on the averaging inverval.   

       NOTE: ds should have been loaded using `decode_times=False`
    """
    assert 'time_bnds' in ds
    assert 'time' in ds
    correct_time_values = ds['time_bnds'].mean(dim='nbnd')
    # copy any metadata:
    correct_time_values.attrs = ds['time'].attrs
    ds = ds.assign_coords({"time": correct_time_values})
    ds = xr.decode_cf(ds)  # decode to datetime objects
    return ds


if __name__ == '__main__':
    # TEST
    ids = xr.open_dataset("/Users/brianpm/Dropbox/Data/b.e21.BCO2x4cmip6.f09_g17.CMIP6-abrupt4xCO2.001.cam.h0.TS.ncwa_global.000101-015012.nc", decode_times=False)    
    print(ids)
    ids = cesm_correct_time(ids)
    print(ids)
