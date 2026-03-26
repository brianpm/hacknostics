import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import healpix as hp

import easygems
import easygems.remap as egr
import easygems.healpix as egh

import dask
import dask.array as da

import zarr

import shutil

# notes
# -----
# - online sources suggest using healpy intead of healpix? 
#   + import healpy 
#   + Possibly healpix was added later and built on healpy.
#   + HEALPix documentation: https://healpix.sourceforge.io/documentation.php

print(f"Python Version: {sys.version}")
print(f"Numpy {np.__version__}")
print(f"Xarray {xr.__version__}")
print(f"Dask {dask.__version__}")

print(f"Healpix {hp.__version__}")
print(f"Zarr {zarr.__version__}")

print(f"EasyGems doesn't provide a version attribute.")


def main():

    # SET NECESSARY INPUT AND OUTPUT PATHS

    # constant files
    dataloc = Path("/glade/campaign/univ/ucsu0085/coupled_l58_15km/hist_postproc_qu")
    datafils = sorted(dataloc.glob("EW_B2000_CAM7_15km_58L_2D_constant_fields.nc")) 
    oloc = Path("/glade/campaign/univ/ucsu0085/coupled_l58_15km/hist_postproc_healpix/2D_constant")
    out_prefix = "2D_constant"

    print(f"Identified {len(datafils)} files to remap to healpix and save to zarr.")

    # mesh description (maybe)
    meshloc = Path("/glade/campaign/univ/ucsu0085/coupled_l58_15km/hist_postproc_qu")
    meshfil = meshloc / "EW_B2000_CAM7_15km_58L_2D_constant.nc"

    # Set parameters needed for generation weights:
    zoom = order = 9

    weights_file = oloc / f"mpas_to_healpix_weights_order{zoom}.nc"

    vert_weights_file = oloc / f"mpas_to_healpix_vertex_weights_order{zoom}.nc"


    overwrite_weights = False

    overwrite_zarr = False

    # Add tracking file path
    tracking_file = oloc / f"{out_prefix}_processed_files.txt"
    processed_files = get_processed_files(tracking_file)

    # resubmit file (for pbs)
    resubmit_file = oloc / f"resubmit.txt"
    if not resubmit_file.is_file():
        # If resubmit doesn't exist, create it with initial value
        with open(resubmit_file, "w") as f:
            f.write("TRUE")
    
    # mpas_to_hp_zarr(ds_mpas_clean, ds_static, zoom, weights_file, vert_weights_file, oloc, out_prefix, clobber_wgts=False, clobber_zarr=True)

    # cell-center
    ds_static = xr.open_dataset(meshfil)
    lon, lat = get_mpas_lonlat(ds_static, 'lonCell', 'latCell', degrees=False, negative=True, verbose=True)
    vlon, vlat = get_mpas_lonlat(ds_static, 'lonVertex', 'latVertex', degrees=False, negative=True, verbose=True)

    # generate or load weights
    eweights = get_weights_to_healpix(lon, lat, zoom, weights_file, overwrite=overwrite_weights)
    evweights = get_weights_to_healpix(vlon, vlat, zoom, vert_weights_file, overwrite=overwrite_weights) # not needed for DYAMOND_diag_3h files

    for i, fil in enumerate(datafils):
        if str(fil) in processed_files:
            print(f"Skipping already processed file: {fil.name}")
            continue
        
        print(f"Processing file {i+1}/{len(datafils)}: {fil.name}")
        data = pre_proc_mpas_file(fil, ds_static)
        dsout = remap_mpas_to_hp(data, eweights, evweights, order)

        # save highest resolution output
        # fn = oloc / f"{out_prefix}_to_hp{order}.zarr"
        # WRITE INDIVIDUAL ZARR FOR EACH FILE:
        fn = oloc / f"{fil.stem}_to_hp{order}.zarr"
        save_to_zarr(dsout, fn, clobber=overwrite_zarr)

        # now coarsen and save zarr
        # mpas_hp_to_zarr(dsout, order, oloc, out_prefix, clobber=overwrite_zarr)

        # INDIVIDUAL FILES:
        mpas_hp_to_zarr(dsout, order, oloc, fil.stem, clobber=overwrite_zarr)

        # Mark as processed only if everything succeeded
        mark_file_as_processed(fil, tracking_file)

        # Mark that we have processed this file
        with open(resubmit_file, "w") as f:
            # Check if this is the last unprocessed file or not
            if i + 1 < len(datafils):
                f.write("TRUE")  # More files to process
            else:
                f.write("FALSE")  # No more files to process

        # having processed a file, end here:
        break

def pre_proc_mpas_file(datafil, meshfil):
    #DD - added mask_and_scale, unsure if necessary
    #ds_mpas = xr.open_dataset(datafil, engine='netcdf4', mask_and_scale=True, chunks={'Time': 'auto'})
    ds_mpas = xr.open_dataset(datafil, engine='netcdf4', mask_and_scale=True, chunks='auto')
    if isinstance(meshfil, xr.Dataset):
        ds_static = meshfil
    elif isinstance(meshfil, Path):
        ds_static = xr.open_dataset(meshfil)
    else:
        raise ValueError("meshfil needs to be a dataset or a path")

    ## Clean and convert xtime strings
    #time_str = ds_mpas.xtime.astype(str).values.astype('U').ravel()
    ## Remove extra whitespace and handle empty strings
    #time_str = [x.strip() for x in time_str]
    #time_str = [x.replace("_", " ") for x in time_str]

    ## Convert to datetime
    ## change coordinate (and index) from "Time" to "time"
    #time_coord = pd.to_datetime(time_str)

    #ds_mpas_new = ds_mpas.assign_coords(time=('Time', time_coord))

    #ds_mpas_new = ds_mpas_new.swap_dims({"Time":"time"})

    ds_mpas_new = ds_mpas

    # Find variables with dtype 'S64'
    s64_vars = [var for var in ds_mpas_new.variables if ds_mpas_new[var].dtype == 'S64']
    print(f"Variables with S64 dtype: {s64_vars}")

    # Drop these variables from the dataset
    ds_mpas_clean = ds_mpas_new.drop_vars(s64_vars)

    return ds_mpas_clean

# All the local functions that we need.
def get_processed_files(tracking_file):
    """Read the list of already processed files."""
    if Path(tracking_file).exists():
        with open(tracking_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_file_as_processed(filepath, tracking_file):
    """Append a processed file to the tracking list."""
    with open(tracking_file, 'a') as f:
        f.write(f"{filepath}\n")

def remove_directory(inpath):
    """
    Removes a directory and its contents recursively.

    Args:
        path_str: Path to the directory as a string.
    """
    if isinstance(inpath, str):
        path = Path(inpath)
    else:
        path = inpath
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed directory: {path}")
    else:
        print(f"Directory not found: {path}")


def get_mpas_lonlat(ds, lonname, latname, degrees=True, negative=True, verbose=False):
    '''Get latitude and longitude from MPAS "static" file,
       convert to degrees (default),
       convert to [-180, 180] convention (default)

    ds : xr.Dataset
        data set that needs to have lat and lon values
    latname : str
        name of the latitude variable
    lonname : str
        name of the longitude variable
    degrees : bool
        if true, convert to degrees (ASSUMES RADIANS)
    negative : bool
        if true, convert to -180 format if needed
        if false, convert to 360 format if needed
        Assumes unit is degrees, and the conversion is based on minimum longitude value being < 0 or maximum > 180
        Does not "roll" the coordinate (i.e. change the order of the longitudes)
    verbose : bool
        if true print stuff
    '''
    lonrad = ds[lonname]
    latrad = ds[latname]
    if verbose:
        print(f"Sizes: {lonrad.shape = }, {latrad.shape = } -- Compare with {ds['nCells'].shape}")
        print(f"[initial] Lat min/max: {latrad.min().item()}, {latrad.max().item()}, Lon min/max: {lonrad.min().item()},{lonrad.max().item()}")
    
    if degrees:
        # lon and lat are in radians
        lon = np.rad2deg(lonrad) 
        lat = np.rad2deg(latrad)
    else:
        lon = lonrad
        lat = latrad

    if verbose:
        print(f"[degrees] Lat min/max: {lat.min().item()}, {lat.max().item()}, Lon min/max: {lon.min().item()},{lon.max().item()}")

    if negative:
        if lon.max().item() >= 180:
            lon=(lon + 180) % 360 - 180  # [-180, 180)
    else:
        if lon.min().item() < 0:
            lon += 180
    if verbose:
        print(f"[final] Lat min/max: {lat.min().item()}, {lat.max().item()}, Lon min/max: {lon.min().item()},{lon.max().item()}")
    return lon, lat


def get_weights_to_healpix(lon, lat, order, weights_file, overwrite=None):
    # nside determines the resolution of the map, generally a power of 2
    # zoom & order are just the exponent:
    # nside = 2**(zoom)

    # npix is just the number of "pixels" (grid points on HEALPix grid)
    zoom = order
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)

    write = False
    if weights_file.is_file():
        if overwrite:
            write = True
            weights_file.unlink()
            print("Overwrite existing file.")
    else:
        write = True

    latlon = True

    print(f"The number of pixels is {npix}, based on {nside} = 2**{zoom}. WRITE: {write}. LATLON: {latlon}")

    if write:
        # gets the longitude and latitude of each
        # latlon: If True, input angles are assumed to be longitude and latitude in degree, otherwise, they are co-latitude and longitude in radians.
        hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=latlon, nest=True)

        # WE NEED TO SHIFT LONGITUDE TO [-180,180] CONVENTION
        # Probably only if source does??
        if latlon and np.any(hp_lon > 180):
            hp_lon = (hp_lon + 180) % 360 - 180  # [-180, 180)
            hp_lon += 360 / (4 * nside) / 4  # shift quarter-width  ##???????##
            # source lon shift already applied using get_mpas_lonlat
        else:
            print(f"Will not modify hp_lon. Min/Max: {hp_lon.min().item()}, {hp_lon.max().item()} Size: {hp_lon.shape}")

        # easygems weight generation
        # If latlon=True above, then we probably want source in degrees
        eweights = egr.compute_weights_delaunay((lon, lat),(hp_lon, hp_lat))

        # save the calculated weights for future use    
        eweights.to_netcdf(weights_file)
        print(f"Weights file written: {weights_file.name}")
        return eweights
    else: 
        return xr.open_dataset(weights_file) 

    # NOTE: write=True takes a while: ~9min



def apply_weights_hp(ds, weights, order, mpas_v_c=None):
    """remap to healpix using easygems generated weights
    
    ds and weights should be consistent
    mpas_v_c determines if using "nCell" or "nVertices" variables
    """
    assert (mpas_v_c in ["center", "vertex"]), f"mpas_v_c must be center or vertex, got {mpas_v_c}"
    # repeat:
    zoom = order
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)

    vertices_vars = []
    center_vars = []
    vars_to_drop = None
    for v in ds:
        if 'nVertices' in ds[v].dims:
            vertices_vars.append(v)
        elif 'nCells' in ds[v].dims:
            center_vars.append(v)
    if mpas_v_c == "center":
        vars_to_drop = vertices_vars
        core_dims_list = ["nCells"]
    elif mpas_v_c == "vertex":
        vars_to_drop = center_vars
        core_dims_list = ["nVertices"]
    if vars_to_drop:
        ds_filter = ds.drop_vars(vars_to_drop)
    else:
        ds_filter = ds
    
    #DD replaced egr.apply_weights with my_apply_weights
    #DD added allow_rechunk to deal with 3D var memory issues
    mpas_remap = xr.apply_ufunc(
        my_apply_weights,
        ds_filter,
        kwargs=weights,
        keep_attrs=True,
        input_core_dims=[core_dims_list],
        output_core_dims=[["cell"]],
        on_missing_core_dim='copy',
        output_dtypes=["f4"],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"cell": npix}, "allow_rechunk":True
        },
    )

    return mpas_remap


def remap_mpas_to_hp(ds, cell_weights, vertex_weights, zoom):
    c_vars = apply_weights_hp(ds, cell_weights, zoom, mpas_v_c="center")

    v_vars = apply_weights_hp(ds, vertex_weights, zoom, mpas_v_c="vertex")
    mrg = xr.merge([c_vars, v_vars])

    # Add the CRS
    mrg["crs"] = xr.DataArray(
        name="crs",
        data=0,
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": 2**zoom,
            "healpix_order": "nest",
        },
    )
    return mrg

# Write to ZARR

def get_dtype(da):
    if np.issubdtype(da.dtype, np.floating):
        return "float32"
    else:
        return da.dtype

def get_encoding(dataset):
    return {
        var: {
            # "compressor": get_compressor(),
            "dtype": get_dtype(dataset[var]),
            # "chunks": get_chunks(dataset[var].dims),
        }
        for var in dataset.variables
        if var not in dataset.dims
    }


def save_to_zarr(ds, fn, clobber=None):
    if fn.exists():
        if clobber:
            print(f"{fn} exists... remove")
            # do_save = True
            remove_directory(fn)
        else:
            print(f"{fn} exists... coarsen and append along time")
            # do_save = False
    # else:
    #     do_save = True
    # if do_save:
    store = zarr.storage.LocalStore(fn)
    if fn.exists():
        print('b4 dstozarr fn exists')
        # If the store exists, append to it
        #ds.chunk({"time": -1, "cell": -1}).to_zarr(store, append_dim='time', consolidated=False) # skip encoding once it is set in zarr
        ds.chunk({"cell": -1}).to_zarr(store, consolidated=False, zarr_format=2) # skip encoding once it is set in zarr
        print('after dstozarr fn exists')
    else:
        print('b4 dstozarr fn nonexists')
        # For the first write, don't use append_dim
        #ds.chunk({"time": -1, "cell": -1}).to_zarr(store, encoding=get_encoding(ds), consolidated=False)
        ds.chunk({"cell": -1}).to_zarr(store, encoding=get_encoding(ds), consolidated=False, zarr_format=2)
        print('after dstozarr fn nonexists')
    print(f'Saved: {str(fn)}')
    # else:
    #     print('Determined not to save to zarr.')


def mpas_hp_to_zarr(ds, zoom, outloc, zarr_name_prefix, clobber=None):
    """Save to zarr at zoom and lower resolutions
    
    notes
    -----
    the iteration is from zoom-1 down to zero. Have to use
    zoom-1 because the computation is to coarsen from the "current"
    healpix level down to the next one.
    """
    dn=ds.copy()
    for x in range(zoom-1,0,-1):
        fn = outloc / f"{zarr_name_prefix}_to_hp{x}.zarr"

        # coarsen by one level
        dx = dn.coarsen(cell=4).mean()
        save_to_zarr(dx, fn, clobber=clobber)
        # iterate
        dn = dx.copy()
    print("[mpas_hp_to_zarr] complete.")


def mpas_to_hp_zarr(data, grid_data, order, c_weights, v_weights, out_dir, zarr_prefix, clobber_wgts=None, clobber_zarr=None):
    # cell-center
    lon, lat = get_mpas_lonlat(grid_data, 'lonCell', 'latCell', degrees=True, negative=True, verbose=True)

    # generate or load weights
    eweights = get_weights_to_healpix(lon, lat, order, c_weights, overwrite=clobber_wgts)

    # MPAS files have variables at cell centers and vertices,
    # to remap them weights for each are needed:

    vlon, vlat = get_mpas_lonlat(grid_data, 'lonVertex', 'latVertex', degrees=True, negative=True, verbose=True)

    evweights = get_weights_to_healpix(vlon, vlat, order, v_weights, overwrite=clobber_wgts)

    dsout = remap_mpas_to_hp(data, eweights, evweights, order)

    # save highest resolution output
    fn = out_dir / f"{zarr_prefix}_to_hp{order}.zarr"
    save_to_zarr(dsout, fn, clobber=clobber_zarr)

    # now coarsen and save zarr
    mpas_hp_to_zarr(dsout, order, out_dir, zarr_prefix, clobber=clobber_zarr)


#DD a version of egr.apply_weights that proper includes nans instead of inserting
#DD    zeros
def my_apply_weights(var, src_idx, weights, valid):
    """Apply given remapping weights.

    Args:
        var (ndarray): Array to remap.
        kwargs: Remapping weights as returned by `compute_weights`.

    Returns:
        ndarray: Remapped values

    See also:
        `compute_weights`
    """

    #DD - make sure nan gets put into those points where there is no valid interpolation (was defaulting to zero earlier)
    temweights = np.where(np.isnan(var[src_idx]), 0, weights)
    temvar = np.where(np.isnan(var[src_idx]),0,var[src_idx]*weights)
    numer = np.sum(temvar, axis=-1)
    denom = np.sum(temweights, axis=-1)

    tem =  np.where(denom>0,  numer/denom, np.nan)
    return np.where(valid, tem, np.nan)


if __name__ == "__main__":
    main()

