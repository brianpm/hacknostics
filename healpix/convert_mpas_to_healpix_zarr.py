import sys
import os
import shutil
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

    # output location
    # oloc = Path("/glade/derecho/scratch/brianpm/healpix")
    oloc = Path("/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND3/v2")


    # resubmit file (for pbs)(expecting an environment variable)
    resubmit_file = Path(os.getenv('RESUBMIT_FILE', oloc / "non_resubmit_file.txt"))
    if resubmit_file is None:
        raise ValueError("RESUBMIT_FILE environment variable not set.")
    # resubmit file (for pbs)
    if not resubmit_file.is_file():
        # If resubmit doesn't exist, create it with initial value
        with open(resubmit_file, "w") as f:
            f.write("TRUE")
            print(f"Resubmit file created: {str(resubmit_file)}")

    # SET NECESSARY INPUT AND OUTPUT PATHS

    # dataloc = Path("/glade/campaign/mmm/wmr/skamaroc/NSC_2023/3.75km_simulation_output_save")
    # dataloc = Path("/glade/derecho/scratch/brianpm/healpix")
    # datafil = dataloc / "diag.3.75km.2020-10-21_07.30.00.nc"

    dataloc = Path("/glade/campaign/mmm/wmr/skamaroc/NSC_2023/3.75km_simulation_output_old_transport")

    # 3hr files:
    # datafils = sorted(dataloc.glob("DYAMOND_diag_3h.3.75km.*.nc")) # note: 70GB per file

    # 1hr files:
    datafils = sorted(dataloc.glob("DYAMOND_diag_1h.3.75km.*.nc")) # note: 4.4GB per file

    print(f"Identified {len(datafils)} files to remap to healpix and save to zarr.")

    print("TEST TEST TEST -- SHORTEN datafils to 1")
    datafils = [datafils[0],]

    # mesh description (maybe)
    meshloc = Path("/glade/campaign/mmm/wmr/skamaroc/NSC_2023")
    meshfil = meshloc / "x1.41943042.static.nc"

    # Set parameters needed for generation weights:
    zoom = order = 10

    weights_file = oloc / f"mpas_to_healpix_weights_order{zoom}_wrap4.nc"

    vert_weights_file = oloc / f"mpas_to_healpix_vertex_weights_order{zoom}_wrap4.nc"

    out_prefix = "DYAMOND_diag_1h"

    overwrite_weights = False

    overwrite_zarr = False

    # Add tracking file path
    tracking_file = oloc / f"{out_prefix}_processed_files.txt"
    processed_files = get_processed_files(tracking_file)

    # mpas_to_hp_zarr(ds_mpas_clean, ds_static, zoom, weights_file, vert_weights_file, oloc, out_prefix, clobber_wgts=False, clobber_zarr=True)

    # cell-center
    ds_static = xr.open_dataset(meshfil)
    lon, lat = get_mpas_lonlat(ds_static, 'lonCell', 'latCell', degrees=True, negative=False, verbose=True, wraparound=False)
    vlon, vlat = get_mpas_lonlat(ds_static, 'lonVertex', 'latVertex', degrees=True, negative=False, verbose=True, wraparound=False)

    # generate or load weights
    wfunc = get_weights_to_healpix_optimized # otherwise get_weights_to_healpix
    eweights = wfunc(lon, lat, zoom, weights_file, overwrite=overwrite_weights)
    evweights = wfunc(vlon, vlat, zoom, vert_weights_file, overwrite=overwrite_weights) # not needed for DYAMOND_diag_3h files

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

    # Do another check to see if we should resubmit
    # Resubmit FALSE if all datafils have been processed
    with open(tracking_file, 'r') as f:
        tracked_files = set(line.strip() for line in f)
    if len(tracked_files) == len(datafils):
        with open(resubmit_file, 'w') as f:
            f.write("FALSE")
        print("All files processed. Resubmit set to FALSE.")
    else:
        print("Not all files processed. Resubmit set to TRUE.")

def pre_proc_mpas_file(datafil, meshfil):
    ds_mpas = xr.open_dataset(datafil, engine='netcdf4',  chunks={'Time': 'auto'})
    if isinstance(meshfil, xr.Dataset):
        ds_static = meshfil
    elif isinstance(meshfil, Path):
        ds_static = xr.open_dataset(meshfil)
    else:
        raise ValueError("meshfil needs to be a dataset or a path")

    # Clean and convert xtime strings
    time_str = ds_mpas.xtime.astype(str).values.astype('U').ravel()
    # Remove extra whitespace and handle empty strings
    time_str = [x.strip() for x in time_str]
    time_str = [x.replace("_", " ") for x in time_str]

    # Convert to datetime
    # change coordinate (and index) from "Time" to "time"
    time_coord = pd.to_datetime(time_str)

    ds_mpas_new = ds_mpas.assign_coords(time=('Time', time_coord))
    
    ds_mpas_new = ds_mpas_new.swap_dims({"Time":"time"})

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


def get_mpas_lonlat(ds, lonname, latname, degrees=True, negative=True, verbose=False, wraparound=True):
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
    result = (lon, lat)

    # wraparound if we need to add periodic points
    # note: this shouldn't be necessary when using
    # the updated `get_weights_to_healpix` with hstack.
    if wraparound:
        orig_size = len(lon)
        lon_wrap = xr.concat([lon, lon], dim=lon.dims[0])
        lon_wrap[orig_size:] = lon+360
        orig_size = len(lat)
        lat_wrap = xr.concat([lat, lat], dim=lat.dims[0])
        if negative:
            if lon_wrap.max().item() >= 180:
                lon_wrap=(lon_wrap + 180) % 720 - 180
        else:
            if lon.min().item() < 0:
                lon += 180
        result = (lon, lat, lon_wrap, lat_wrap)            

    if verbose:
        print(f"[final] Lat min/max: {lat.min().item()}, {lat.max().item()}, Lon min/max: {lon.min().item()},{lon.max().item()}")
    return result


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

        # # WE NEED TO SHIFT LONGITUDE TO [-180,180] CONVENTION
        # # Probably only if source does??
        # if latlon and np.any(hp_lon > 180):
        #     hp_lon = (hp_lon + 180) % 360 - 180  # [-180, 180)
        #     hp_lon += 360 / (4 * nside) / 4  # shift quarter-width  ##???????##
        #     # source lon shift already applied using get_mpas_lonlat
        # else:
        #     print(f"Will not modify hp_lon. Min/Max: {hp_lon.min().item()}, {hp_lon.max().item()} Size: {hp_lon.shape}")

        # Ensure lon is periodic by stacking 3 copies:
        lon_periodic = np.hstack((lon - 360, lon, lon + 360))
        lat_periodic = np.hstack((lat, lat, lat))

        # easygems weight generation
        # If latlon=True above, then we probably want source in degrees
        eweights = egr.compute_weights_delaunay((lon_periodic, lat_periodic),(hp_lon, hp_lat))
        # Remap the source indices back to their valid range
        eweights = eweights.assign(src_idx=eweights.src_idx % lat.size)

        # save the calculated weights for future use    
        eweights.to_netcdf(weights_file)
        print(f"Weights file written: {weights_file.name}")
        return eweights
    else:
        return xr.open_dataset(weights_file)

    # NOTE: write=True takes a while: ~9min


def get_weights_to_healpix_optimized(lon, lat, order, weights_file, overwrite=None):
    # Convert input arrays to float32 if they aren't already
    lon = lon.astype(np.float32)
    lat = lat.astype(np.float32)
    
    # Basic setup
    zoom = order
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)
    
    # Check if we need to write a new file
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
    
    if not write:
        return xr.open_dataset(weights_file)
    
    # Get HEALPix grid points
    hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=latlon, nest=True)
    
    # Process in chunks based on latitude bands
    chunk_size = 10  # degrees of latitude per chunk
    lat_chunks = np.arange(-90, 91, chunk_size)
    
    # Initialize lists to store combined results
    all_src_idx = []
    all_weights = []
    all_valid = []
    all_tgt_idx = []
    
    for i in range(len(lat_chunks) - 1):
        lat_min, lat_max = lat_chunks[i], lat_chunks[i+1]
        
        # Get HEALPix points in this latitude band
        mask_hp = (hp_lat >= lat_min) & (hp_lat < lat_max)
        if not np.any(mask_hp):
            continue
            
        hp_lon_chunk = hp_lon[mask_hp]
        hp_lat_chunk = hp_lat[mask_hp]
        hp_idx_chunk = np.arange(npix)[mask_hp]
        
        # Get source points that could influence this latitude band
        # Add a buffer zone
        buffer = 2.0  # degrees
        mask_src = (lat >= lat_min - buffer) & (lat <= lat_max + buffer)
        
        if not np.any(mask_src):
            continue
            
        src_lon_chunk = lon[mask_src]
        src_lat_chunk = lat[mask_src]
        src_idx_chunk = np.arange(len(lat))[mask_src]
        
        # Handle periodicity only for points near the boundaries
        lon_buffer = 5.0  # degrees
        near_0 = src_lon_chunk <= lon_buffer
        near_360 = src_lon_chunk >= (360 - lon_buffer)
        
        # Only duplicate the boundary points
        lon_periodic = np.hstack((
            src_lon_chunk[near_360] - 360,
            src_lon_chunk,
            src_lon_chunk[near_0] + 360
        ))
        
        lat_periodic = np.hstack((
            src_lat_chunk[near_360],
            src_lat_chunk,
            src_lat_chunk[near_0]
        ))
        
        # Create mapping array from periodic indices to original indices
        periodic_to_original = np.hstack((
            src_idx_chunk[near_360],
            src_idx_chunk,
            src_idx_chunk[near_0]
        ))
        
        print(f"Processing latitude band {lat_min} to {lat_max}")
        print(f"Target points: {len(hp_lon_chunk)}")
        print(f"Source points in band: {len(src_lon_chunk)}")
        print(f"Total periodic points: {len(lon_periodic)}")
        
        try:
            # Compute weights for this chunk
            chunk_result = egr.compute_weights_delaunay(
                (lon_periodic, lat_periodic),
                (hp_lon_chunk, hp_lat_chunk)
            )
            
            # Remap the source indices to the original dataset
            # Get the src_idx array with shape (tgt_idx, tri)
            src_indices = chunk_result.src_idx.values
            
            # Map each source index through the periodic_to_original array
            # This converts from indices in the periodic array to original indices
            remapped_src_indices = periodic_to_original[src_indices]
            
            # Store the results with the global target indices
            all_src_idx.append(remapped_src_indices)
            all_weights.append(chunk_result.weights.values)
            all_valid.append(chunk_result.valid.values)
            all_tgt_idx.append(hp_idx_chunk)
            
            print(f"Successfully processed latitude band {lat_min} to {lat_max}")
            
        except Exception as e:
            print(f"Error processing latitude band {lat_min} to {lat_max}: {e}")
            # Continue with next chunk instead of failing completely
    
    if not all_src_idx:
        raise ValueError("Failed to compute weights for any chunk")
    
    # Combine all chunk results
    # We need to create a new xarray Dataset with the combined results
    
    # Calculate the total number of target points
    total_targets = sum(len(idx) for idx in all_tgt_idx)
    
    # Create arrays to hold the combined data
    combined_src_idx = np.zeros((total_targets, 3), dtype=np.int64)
    combined_weights = np.zeros((total_targets, 3), dtype=np.float32)
    combined_valid = np.zeros(total_targets, dtype=bool)
    
    # Fill the arrays with data from each chunk
    pos = 0
    for tgt_idx, src_idx, weights, valid in zip(all_tgt_idx, all_src_idx, all_weights, all_valid):
        n_points = len(tgt_idx)
        combined_src_idx[pos:pos+n_points] = src_idx
        combined_weights[pos:pos+n_points] = weights
        combined_valid[pos:pos+n_points] = valid
        pos += n_points
    
    # Create the final Dataset
    combined_result = xr.Dataset(
        data_vars={
            "src_idx": (("tgt_idx", "tri"), combined_src_idx),
            "weights": (("tgt_idx", "tri"), combined_weights),
            "valid": (("tgt_idx",), combined_valid),
            "tgt_idx": (("tgt_idx",), np.concatenate(all_tgt_idx))
        }
    )
    
    # Save the calculated weights
    combined_result.to_netcdf(weights_file)
    print(f"Weights file written: {weights_file.name}")
    
    return combined_result


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
    
    mpas_remap = xr.apply_ufunc(
        egr.apply_weights,
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
            "output_sizes": {"cell": npix},
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
        # If the store exists, append to it
        ds.chunk({"time": -1, "cell": -1}).to_zarr(store, append_dim='time', consolidated=False, zarr_format=2) # skip encoding once it is set in zarr
    else:
        # For the first write, don't use append_dim
        ds.chunk({"time": -1, "cell": -1}).to_zarr(store, encoding=get_encoding(ds), consolidated=False, zarr_format=2)
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


if __name__ == "__main__":
    main()
