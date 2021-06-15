
#
# --- below are some functions that are not used right now ---
#
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
