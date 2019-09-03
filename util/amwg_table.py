#

from pathlib import Path
import xarray as xr

# goal: replace the "Tables" set in AMWG
#	Set Description
#   1 Tables of ANN, DJF, JJA, global and regional means and RMSE.

# They use 4 pre-defined domains:
domains = {"global": (0, 360, -90, 90),
           "tropics": (0, 360, -20, 20),
           "southern": (0, 360, -90, -20),
           "northern": (0, 360, 20, 90)}

# and then in time it is DJF JJA ANN


# within each domain and season
# the result is just a table of
# VARIABLE-NAME, RUN VALUE, OBS VALUE, RUN-OBS, RMSE

def get_domain_subset(data, domain):
	# domain: Tuple: (west, east, south, north)
	# if domain is global, don't worry about it
	if domain[0] == 0 and domain[1] == 360:
		lon_global = True
	else:
		lon_global = False
	if domain[2] == -90 and domain[3] == 90:
	    lat_global = True
	else:
	    lat_global = False
	if lon_global and lat_global:
		return data  # don't do anything
	if lon_global:
		return data.sel(lat=slice(domain[2],domain[3]))
	if lat_global:
		return data.sel(lon=slice(domain[0],domain[1]))
	# if we get to here, then it must have lat and lon specified 
	return data.sel(lat=slice(domain[2],domain[3]), lon=slice(domain[0],domain[1]))


def get_season_average(data):
	# this is super easy as long as we've got xarray and correct time coordinate.
	# NOTE 1: that this doesn't properly weight by different lengths of months.
    # NOTE 2: this returns "DJF", "JJA", "MAM", "SON" for dim 'season'
	return data.groupby("time.season").mean(dim='time')


def weighted_mean(x):
	# here we assume that the input has ("lat","lon") coordinates attached
	w = np.cos(np.radians(x['lat']))
	# normalized weights:
    wnorm = w / w.sum()
    a0 = x.mean(dim='lon')
	return (x * wnorm).sum(dim='lat')


def get_rmse(data1, data2):
	# Assume that data1 and data2 are same size.
	# Do it correctly by calculating the average correctly.
	# We ignore time here; this is a measure of pattern difference.
	return np.sqrt((weighted_mean(data2 - data1))**2)


def construct_row(a, b):
	value_a = weighted_mean(a)
	value_b = weighted_mean(b)
	bias = value_b - value_a
	rmse = get_rmse(a, b)
	return variable, value_a, value_b, bias, rmse


def _work(a, b):
	# this assumes nice interval (jan-dec)
	if 'time' in a.dims:
		a_avg = a.mean(dim='time')
	else:
		a_avg = a
	if 'time' in b.dims:
		b_avg = b.mean(dim='time')
	else:
		b_avg = b
	a_seasons = get_season_average(a)
	b_seasons = get_season_average(b)
	# this needs changing, but just to mock out:
	table = {s: construct_row(a_seasons.sel(season=s), b_season.sel(season=s)) for s in ["DJF", "JJA", "MAM", "SON"]}
	table["all"] = construct_row(a_avg, b_avg)
	return table


def _loop(a, b):
	for d in domains:
		aa = get_domain_subset(a, d)
		bb = get_domain_subset(b, d)
		return _work(aa, bb)



if __name__ == "__main__":
	data_directory = Path("/Users/brianpm/Dropbox/Data/CERES")
	obs_ds = xr.open_dataset(data_directory/"CERES_EBAF-TOA_Edition4.0_200003-201810.nc")

	model_directory = Path("")
	model_fils = model_directory.glob("b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.*.100001-109912.ncra.nc")
	model_ds = xr.open_mfdataset(model_fils, combine='by_coords')
	#
	# Once we have data, we need to know what to do with it.
	#
	variables_to_process = [] # this should be passed in
	for variable in variables_to_process:
		v1, v2 = get_variable_data(ds1, ds2, variable)
		result = _loop(v1, v2)
		# what do we do with the result? 
		# ...
	#     get the variable out of each data set
	#         ===>>> some variables are derived -> need a way to specify the calculation
	#         ===>>> optionally check on units and grid
	#     run the loop => table with answer for each season


