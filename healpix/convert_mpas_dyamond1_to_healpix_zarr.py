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

import functools

import time

# SET UP LOGGING
import logging

# Configure logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# notes
# -----
# - online sources suggest using healpy intead of healpix? 
#   + import healpy 
#   + Possibly healpix was added later and built on healpy.
#   + HEALPix documentation: https://healpix.sourceforge.io/documentation.php

version_checks = f"Python Version: {sys.version}\n Numpy {np.__version__}\n Xarray {xr.__version__}\n Dask {dask.__version__}\n Healpix {hp.__version__}\n Zarr {zarr.__version__}"
logger.info(version_checks)

def timer(func):
    """Decorator to logger.info the runtime of a function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        msg = f"[TIMER] {func.__name__!r} executed in {end - start:.4f} seconds"
        logger.info(msg)
        return result
    return wrapper_timer


def main():

    # output location
    # oloc = Path("/glade/derecho/scratch/brianpm/healpix")
    oloc = Path("/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND1")
    # oloc = Path("/glade/campaign/cgd/cas/brianpm/hack25/mpas_DYAMOND1")


    # resubmit file (for pbs)(expecting an environment variable)
    resubmit_file = Path(os.getenv('RESUBMIT_FILE', oloc / "non_resubmit_file.txt"))
    if resubmit_file is None:
        raise ValueError("RESUBMIT_FILE environment variable not set.")
    # resubmit file (for pbs)
    if not resubmit_file.is_file():
        # If resubmit doesn't exist, create it with initial value
        with open(resubmit_file, "w") as f:
            f.write("TRUE")
            msg = f"Resubmit file created: {str(resubmit_file)}"
            logger.info(msg)

    # SET NECESSARY INPUT AND OUTPUT PATHS
    dataloc = Path("/glade/campaign/mmm/wmr/fjudt/projects/dyamond_1/3.75km")
    datafils = sorted(dataloc.glob("history.2016*.nc")) # note: 62GB per file
    msg = f"Identified {len(datafils)} files to remap to healpix and save to zarr."
    logger.info(msg)

    # mesh description (maybe)
    meshloc = Path("/glade/campaign/mmm/wmr/skamaroc/NSC_2023")
    meshfil = meshloc / "x1.41943042.static.nc"

    # Set parameters needed for generation weights:
    zoom = order = 10

    weights_file = oloc / f"mpas_to_healpix_weights_order{zoom}_wrap4ez4.nc"

    vert_weights_file = oloc / f"mpas_to_healpix_vertex_weights_order{zoom}_wrap4ez4.nc"

    out_prefix = "DYAMOND1_history"

    overwrite_weights = False

    overwrite_zarr = False

    # Add tracking file path
    tracking_file = oloc / f"{out_prefix}_processed_files.txt"
    processed_files = get_processed_files(tracking_file)

    # cell-center
    ds_static = xr.open_dataset(meshfil)
    lon, lat = get_mpas_lonlat(ds_static, 'lonCell', 'latCell', degrees=True, negative=False, verbose=True, wraparound=False)
    vlon, vlat = get_mpas_lonlat(ds_static, 'lonVertex', 'latVertex', degrees=True, negative=False, verbose=True, wraparound=False)

    # generate or load weights
    # wfunc: [get_weights_to_healpix, get_weights_to_healpix_optimized]
    wfunc = get_weights_to_healpix_optimized
    wopts = {"overwrite":overwrite_weights} # "optimized"
    # wopts = {"overwrite":overwrite_weights, "wraparound":True} # "ez"
    eweights = wfunc(lon, lat, zoom, weights_file, **wopts)
    evweights = wfunc(vlon, vlat, zoom, vert_weights_file, **wopts) # not needed for DYAMOND_diag_3h files

    for i, fil in enumerate(datafils):
        if str(fil) in processed_files:
            logger.info(f"Skipping already processed file: {fil.name}")
            continue
        
        logger.info(f"Processing file {i+1}/{len(datafils)}: {fil.name}")
        data = pre_proc_mpas_file(fil, ds_static)
        if hasattr(data, "compute"):
            load_start_time = time.perf_counter()
            data = data.compute()
            load_end_time = time.perf_counter()
            logger.info(f"[TIMER (1)] DATA LOADING TIME: {load_end_time - load_start_time} seconds.")
        dsout = remap_mpas_to_hp(data, eweights, evweights, order)
        load_start_time = time.perf_counter()
        if hasattr(dsout, "compute"):
            dsout = dsout.compute()
        load_end_time = time.perf_counter()
        logger.info(f"[TIMER] DATA LOADING TIME: {load_end_time - load_start_time} seconds.")

        # ..............................
        # save highest resolution output 
        # APPEND INTO ONE ZARR
        fn = oloc / f"{out_prefix}_to_hp{order}.zarr"

        # WRITE INDIVIDUAL ZARR FOR EACH FILE:
        # fn = oloc / f"{fil.stem}_to_hp{order}.zarr"
        # ..............................

        save_to_zarr(dsout, fn, clobber=overwrite_zarr)

        # now coarsen and save zarr
        # INDIVIDUAL FILES:
        # APPEND
        mpas_hp_to_zarr(dsout, order, oloc, out_prefix, clobber=overwrite_zarr)
        # INDIVIDUAL
        # mpas_hp_to_zarr(dsout, order, oloc, fil.stem, clobber=overwrite_zarr)

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
        logger.info("All files processed. Resubmit set to FALSE.")
    else:
        logger.info("Not all files processed. Resubmit set to TRUE.")

@timer
def pre_proc_mpas_file(datafil, meshfil):
    ds_mpas = xr.open_dataset(datafil, engine='netcdf4', mask_and_scale=True, chunks={'Time': 'auto'})

    if len(ds_mpas['Time']) > 1:
        logger.info("Multiple times detected... will probably break preprocessor")

    # if isinstance(meshfil, xr.Dataset):
    #     ds_static = meshfil
    # elif isinstance(meshfil, Path):
    #     ds_static = xr.open_dataset(meshfil)
    # else:
    #     raise ValueError("meshfil needs to be a dataset or a path")

    # Use a fixed reference date for all files
    ref_date = '2000-01-01 00:00:00'  # Or any other suitable fixed date

    # Clean and convert xtime strings
    time_str = ds_mpas.xtime.astype(str).values.astype('U').ravel()
    # Remove extra whitespace and handle empty strings
    time_str = [x.strip() for x in time_str]
    time_str = [x.replace("_", " ") for x in time_str]
    if isinstance(time_str, np.ndarray) or isinstance(time_str, list):
        time_str = "".join(time_str)

    # Convert to datetime
    # change coordinate (and index) from "Time" to "time"
    time_coord = pd.to_datetime(time_str)

    # Calculate hours since reference date for the coordinate values
    hours_since = (time_coord - pd.Timestamp(ref_date)) / pd.Timedelta('1h')
    if isinstance(hours_since, xr.DataArray):
        hours_since = hours_since.values
    elif isinstance(hours_since, float):
        hours_since = np.array([hours_since,])

    # Create time coordinate with specific encoding
    time_var = xr.DataArray(
        hours_since,
        dims='Time',
        name='time',
        attrs={'long_name': 'time', 
               'axis': 'T',
               'reference_date': ref_date},
               )
    time_var.encoding = {
        'dtype': 'float64',
        'units': f'hours since {ref_date}',
        'calendar': 'standard',
        '_FillValue': None
    }
    
    ds_mpas_new = ds_mpas.assign_coords(time=('Time', hours_since))
    # ds_mpas_new = ds_mpas.assign_coords(time=('Time', time_var.data))
    
    ds_mpas_new = ds_mpas_new.swap_dims({"Time":"time"})

    # Find variables with dtype 'S64'
    s64_vars = [var for var in ds_mpas_new.variables if ds_mpas_new[var].dtype == 'S64']
    logger.info(f"Variables with S64 dtype: {s64_vars}")


    # Drop these variables from the dataset
    ds_mpas_clean = ds_mpas_new.drop_vars(s64_vars)

    # Explicitly drop xtime and xtime_old if they are present:
    ds_mpas_clean = ds_mpas_clean.drop_vars(['xtime', 'xtime_old'], errors='ignore')

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
        logger.info(f"Removed directory: {path}")
    else:
        logger.info(f"Directory not found: {path}")


@timer
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
        if true logger.info stuff
    '''
    lonrad = ds[lonname]
    latrad = ds[latname]
    if verbose:
        logger.info(f"Sizes: {lonrad.shape = }, {latrad.shape = } -- Compare with {ds['nCells'].shape}")
        logger.info(f"[initial] Lat min/max: {latrad.min().item()}, {latrad.max().item()}, Lon min/max: {lonrad.min().item()},{lonrad.max().item()}")
    
    if degrees:
        # lon and lat are in radians
        lon = np.rad2deg(lonrad)
        lat = np.rad2deg(latrad)
    else:
        lon = lonrad
        lat = latrad

    if verbose:
        logger.info(f"[degrees] Lat min/max: {lat.min().item()}, {lat.max().item()}, Lon min/max: {lon.min().item()},{lon.max().item()}")

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
        logger.info(f"[final] Lat min/max: {lat.min().item()}, {lat.max().item()}, Lon min/max: {lon.min().item()},{lon.max().item()}")
    return result

@timer
def get_weights_to_healpix(lon, lat, order, weights_file, overwrite=None, wraparound=False):
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
            logger.info("Overwrite existing file.")
    else:
        write = True

    latlon = True

    logger.info(f"The number of pixels is {npix}, based on {nside} = 2**{zoom}. WRITE: {write}. LATLON: {latlon}")

    if write:
        # gets the longitude and latitude of each
        # latlon: If True, input angles are assumed to be longitude and latitude in degree, otherwise, they are co-latitude and longitude in radians.
        hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=latlon, nest=True)

        logger.info(f"[get_weights_to_healpix] hp_lon min/max: {hp_lon.min()}, {hp_lon.max()}")

        # # WE NEED TO SHIFT LONGITUDE TO [-180,180] CONVENTION
        # # Probably only if source does??
        # if latlon and np.any(hp_lon > 180):
        #     hp_lon = (hp_lon + 180) % 360 - 180  # [-180, 180)
        #     hp_lon += 360 / (4 * nside) / 4  # shift quarter-width  ##???????##
        #     # source lon shift already applied using get_mpas_lonlat
        # else:
        #     logger.info(f"Will not modify hp_lon. Min/Max: {hp_lon.min().item()}, {hp_lon.max().item()} Size: {hp_lon.shape}")

        if wraparound:
            # Ensure lon is periodic by stacking 3 copies:
            lon_periodic = np.hstack((lon-360, lon))
            lat_periodic = np.hstack((lat, lat))

            # easygems weight generation
            # If latlon=True above, then we probably want source in degrees
            eweights = egr.compute_weights_delaunay((lon_periodic, lat_periodic),(hp_lon, hp_lat))
            # Remap the source indices back to their valid range
            eweights = eweights.assign(src_idx=eweights.src_idx % lat.size)
        else:
            eweights = egr.compute_weights_delaunay((lon, lat),(hp_lon, hp_lat))

        # save the calculated weights for future use    
        eweights.to_netcdf(weights_file)
        logger.info(f"Weights file written: {weights_file.name}")
        return eweights
    else:
        return xr.open_dataset(weights_file)

    # NOTE: write=True takes a while: ~9min

@timer
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
            logger.info("Overwrite existing file.")
    else:
        write = True
    
    latlon = True
    logger.info(f"The number of pixels is {npix}, based on {nside} = 2**{zoom}. WRITE: {write}. LATLON: {latlon}")
    
    if not write:
        return xr.open_dataset(weights_file)
    
    # Get HEALPix grid points
    hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=latlon, nest=True)

    # Verify HEALPix coordinate ordering
    logger.info("Diagnosing HEALPix coordinates:")
    logger.info(f"HEALPix longitude range: {hp_lon.min():.2f} to {hp_lon.max():.2f}")
    logger.info(f"HEALPix latitude range: {hp_lat.min():.2f} to {hp_lat.max():.2f}")
    # Check if coordinates follow expected pattern
    pixel_coords = np.column_stack([hp_lon, hp_lat])
    logger.info(f"First few pixel coordinates:\n{pixel_coords[:5]}")
    logger.info(f"Last few pixel coordinates:\n{pixel_coords[-5:]}")

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

        # Add extra points near prime meridian
        # healpix grids think longitude goes to -45, so go that far
        prime_meridian_buffer = 45.0  # degrees
        near_prime = (lon <= prime_meridian_buffer) | (lon >= 360 - prime_meridian_buffer)
        mask_src = mask_src | (near_prime & (lat >= lat_min - buffer) & (lat <= lat_max + buffer))

        if not np.any(mask_src):
            continue
            
        src_lon_chunk = lon[mask_src]
        src_lat_chunk = lat[mask_src]
        src_idx_chunk = np.arange(len(lat))[mask_src]
        
        # Handle periodicity only for points near the boundaries
        # (I think to get this to work for HEALPix, longitude needs to go to -45)
        lon_buffer = 45.0  # degrees
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
        
        logger.info(f"Processing latitude band {lat_min} to {lat_max}")
        logger.info(f"Target points: {len(hp_lon_chunk)}")
        logger.info(f"Source points in band: {len(src_lon_chunk)}")
        logger.info(f"Total periodic points: {len(lon_periodic)}")
        
        try:
            # Compute weights for this chunk
            chunk_result = egr.compute_weights_delaunay(
                (lon_periodic, lat_periodic),
                (hp_lon_chunk, hp_lat_chunk)
            )
            # Check for gaps in coverage
            if np.any(~chunk_result.valid.values):
                invalid_points = np.where(~chunk_result.valid.values)[0]
                logger.info(f"Warning: {len(invalid_points)} invalid points in latitude band {lat_min} to {lat_max}")
                logger.info(f"Invalid point coordinates:")
                logger.info(f"Lon: {hp_lon_chunk[invalid_points]}")
                logger.info(f"Lat: {hp_lat_chunk[invalid_points]}")
                
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
            
            logger.info(f"Successfully processed latitude band {lat_min} to {lat_max}")
            
        except Exception as e:
            logger.info(f"Error processing latitude band {lat_min} to {lat_max}: {e}")
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

    # After combining chunks, sort by target index to restore HEALPix ordering
    sort_idx = np.argsort(np.concatenate(all_tgt_idx))

    # Create the final Dataset with sorted indices
    combined_result = xr.Dataset(
        data_vars={
            "src_idx": (("tgt_idx", "tri"), combined_src_idx[sort_idx]),
            "weights": (("tgt_idx", "tri"), combined_weights[sort_idx]),
            "valid": (("tgt_idx",), combined_valid[sort_idx]),
            "tgt_idx": (("tgt_idx",), np.arange(npix))  # Use sequential indices
        }
    )

    # Add HEALPix metadata
    combined_result.attrs.update({
        "healpix_ordering": "NEST",
        "nside": nside,
        "order": order,
        "npix": npix
    })

    logger.info("Diagnosing target index ordering:")
    logger.info(f"Target indices min/max: {combined_result.tgt_idx.min().item()}, {combined_result.tgt_idx.max().item()}")
    logger.info(f"Number of unique target indices: {len(np.unique(combined_result.tgt_idx))}")
    logger.info(f"Expected number of pixels: {npix}")
    # Check if indices are monotonic
    is_monotonic = np.all(np.diff(combined_result.tgt_idx) >= 0)
    logger.info(f"Target indices are monotonic: {is_monotonic}")

    # Save the calculated weights
    combined_result.to_netcdf(weights_file)
    logger.info(f"Weights file written: {weights_file.name}")
    
    return combined_result

@timer
def apply_weights_hp(ds, weights, order, mpas_v_c=None):
    """remap to healpix using easygems generated weights
    
    ds and weights should be consistent
    mpas_v_c determines if using "nCell" or "nVertices" variables
    """

    logger.info("Diagnosing weight application:")
    logger.info(f"Weight target indices shape: {weights.tgt_idx.shape}")
    logger.info(f"Weight source indices shape: {weights.src_idx.shape}")
    logger.info(f"Number of unique target pixels: {len(np.unique(weights.tgt_idx))}")

    assert (mpas_v_c in ["center", "vertex"]), f"mpas_v_c must be center or vertex, got {mpas_v_c}"
    # repeat:
    zoom = order
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)
    if not np.array_equal(weights.tgt_idx, np.arange(npix)):
        raise ValueError("Weight target indices are not in proper HEALPix order")
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
             "allow_rechunk":True
        },
    )
    return mpas_remap

@timer
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
    """Determine appropriate dtype for zarr encoding"""
    if da.dtype == 'float64':
        return 'float32'
    elif da.dtype == 'int64':
        return 'int32'
    else:
        return da.dtype

def get_encoding(dataset):
    encodings = {
        var: {
            "dtype": get_dtype(dataset[var]),
        }
        for var in dataset.variables
        if var not in dataset.dims
    }    
    # Add specific encoding for time coordinate
    if 'time' in dataset.coords:
        encodings['time'] = {
            'dtype': 'float64',
            '_FillValue': None,
            # Store units and calendar as attributes instead of encoding
            'compressor': None  # Disable compression for coordinate
        }
        # Set the units and calendar as attributes
        dataset['time'].attrs.update({
            'units': 'hours since 2000-01-01 00:00:00',
            'calendar': 'standard'
        })
    return encodings

@timer
def save_to_zarr(ds, fn, clobber=None):
    chunks = {dim: -1 for dim in ds.dims}
    chunks['time'] = 8
    chunks['cell'] = 262144
    ds = ds.chunk(chunks)
    if fn.exists():
        if clobber:
            logger.info(f"{fn} exists... remove")
            remove_directory(fn)
            # Create new store with encoding
            store = zarr.storage.LocalStore(fn)
            encoding = get_encoding(ds)
            ds.to_zarr(
                store, 
                encoding=encoding, 
                consolidated=True, 
                zarr_format=2
            )
        else:
            logger.info(f"{fn} exists... check time and append.")
            zds = xr.open_dataset(fn)
            ztime = zds['time']
            encoding = get_encoding(ds)
            de_t = xr.decode_cf(ds)['time']
            if len(de_t) == 1:
                already_done = any(de_t == zds['time'])
            else:
                logger.info("Length of dataset `time` is >1.")
                already_done = False
            ##########################
            if not already_done:
                # When appending, don't provide encoding
                store = zarr.storage.LocalStore(fn)
                ds.to_zarr(
                    store,
                    append_dim='time',
                    consolidated=True,
                    zarr_format=2
                )
            else:
                logger.warning("DETERMINED THIS TIME IS ALREADY IN THE ZARR. SKIP WRITING.")
    else:
        # For the first write, include encoding
        store = zarr.storage.LocalStore(fn)
        encoding = get_encoding(ds)
        ds.to_zarr(
            store,
            encoding=encoding,
            consolidated=True,
            zarr_format=2
        )    

    logger.info(f'Saved: {str(fn)}')
    # Consolidate metadata after writing
    zarr.consolidate_metadata(str(fn))

@timer
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

        # update the CRS info
        dx["crs"] = xr.DataArray(name="crs",data=0,
          attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": 2**x,
            "healpix_order": "nest",
          },
        )

        save_to_zarr(dx, fn, clobber=clobber)
        # iterate
        dn = dx.copy()
    logger.info("[mpas_hp_to_zarr] complete.")

@timer
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
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")