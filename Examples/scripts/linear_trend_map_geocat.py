import numpy as np
import xarray as xr
import geocat.comp as gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# CAUTION -- This script is incomplete and could contain errors. (17 Sept 2021)

# See also the notebook called linear_trend_map.ipynb

def main(ifil, v, method='xarray', ofil=None):
    if ofil is None:
        ofil = "./test.png"
    ds = xr.open_dataset(ifil, decode_times=False)
    if 'time_bnds' in ds:
        correct_time = ds['time_bnds'].mean(dim='nbnd')
        correct_time.attrs = ds['time'].attrs
        ds = ds.assign_coords({"time":correct_time})
    ds = xr.decode_cf(ds)
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude':'lon'})
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude':'lat'})
    X = ds[v]
    X = X.groupby('time.year').mean(dim='time')
    X = X.rename({'year':'time'})
    print(f"X coords: {X.coords}")
    print(f"X shape: {X.shape}")
    print(f"X min: {X.min().item()}, max: {X.max().item()}")
    # yearfrac = X.time.dt.year + X.time.dt.month/12
    yearfrac = X.time
    print(f"Time shape: {yearfrac.shape}")
    if method == 'geocat':
        trend = gc.polynomial.ndpolyfit(yearfrac, X.values, deg=1, axis=0, meta=False, full=False)
        y_fitted = gc.polynomial.ndpolyval(trend, yearfrac)
        print(y_fitted)
        yhat = y_fitted
    elif method == 'xarray':
        trend = X.polyfit(dim='time', deg=1, skipna=True)  # dataset with polyfit_coefficients[degree, lat, lon]
        slope = trend['polyfit_coefficients'][0,:,:]  # SLOPE aka TREND... units of X per units of the time dimension, [lat, lon]
        y_fitted = xr.polyval(X.time, trend) # fits the linear model to the data
        yhat = y_fitted['polyfit_coefficients'] # the estimated data
    else:
        print(f"ERROR method = {method}")

    # Goodness of fit / Correlation of determination
    #     # # r**2
    ybar = X.mean(dim='time')
    ssreg = ((yhat - ybar)**2).sum(dim='time')
    sstot = ((X - ybar)**2).sum(dim='time')
    rsquared = 1 - (ssreg / sstot)

    # Student's t-test (two-sided since we could have negative or positive trends)


    # Decide what to plot
    # map_rsquared(rsquared, ofil)
    # print(rsquared)
    print(slope)
    map_trend(slope, ofil)


def map_trend(data, ofil):
    print(f"Trend min: {data.min().item()}, max: {data.max().item()}")
    fig, ax = plt.subplots(subplot_kw={"projection":ccrs.Mollweide()})
    lons, lats = np.meshgrid(data.lon, data.lat)
    ax.coastlines()
    ax.set_global()
    # img = ax.pcolormesh(lons, lats, data.values, cmap='plasma_r', rasterized=False)
    img = ax.contourf(lons, lats, data.values, cmap='plasma_r')
    fig.colorbar(img)
    plt.show()
    fig.savefig(ofil, bbox_inches='tight', dpi=200)


def map_rsquared(data, ofil):
    print(f"shape of data is {data.shape}")
    fig, ax = plt.subplots(subplot_kw={"projection":ccrs.Mollweide()})
    lons, lats = np.meshgrid(data.lon, data.lat)
    img = ax.pcolormesh(lons, lats, data, cmap='plasma_r', norm=mpl.colors.Normalize(vmin=0,vmax=1), rasterized=True, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    fig.colorbar(img)
    fig.savefig(ofil, bbox_inches='tight', dpi=200)
    



if __name__ == "__main__":
    input_file = "/Users/brianpm/Dropbox/Data/HadCRUT/HadCRUT.4.6.0.0.median.nc"
    main(input_file, "temperature_anomaly", method='xarray', ofil="/Users/brianpm/Desktop/example_trend_test.png")

