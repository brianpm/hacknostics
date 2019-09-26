# ***********************************************
# xy_1.py
#
# Concepts illustrated:
#   - Drawing a XY plot
#   - Directly analogous to xy_1.ncl (http://www.ncl.ucar.edu/Applications/xy.shtml)
#
#************************************************
# These packages are used
import xarray as xr
import matplotlib.pyplot as plt
#************************************************

#************************************************
# read in data
#************************************************

# on my system, this was the same as "NCARG_ROOT"
NCARG_ROOT = "/Users/brianpm/miniconda3/pkgs/ncl-6.6.2-h2f2bd2c_1"
f     = xr.open_dataset(f"{NCARG_ROOT}/lib/ncarg/data/cdf/uv300.nc")
u     = f["U"]                                    # get u data

#print(u) # <xarray.DataArray 'U' (time: 2, lat: 64, lon: 128)>

#************************************************
# plotting parameters
#************************************************

# Replicate the NCL example code as best I can
fig, ax = plt.subplots()
plot = ax.plot(u["lat"], u.sel(lon=82, method='nearest').isel(time=0)) # create plot
fig.suptitle("Basic XY plot")  # add title
fig.savefig("xy.png")  # send graphics to PNG file

# BUT... NCL example is using gsn_csm_xy
# Important: gsn_csm_xy knows about latitude and sets the variable name as label
# style choices to be more "NCLish"

fig2, ax2 = plt.subplots()
plot2 = ax2.plot(u["lat"], u.sel(lon=82, method='nearest').isel(time=0)) # create plot
# note equivalent to using u[0,:,lon82], where lon82 = np.argmin(np.abs(ghgfile['lon'] - 82))

# NOTE ON TICKS: there are several ways to control ticks, this seems to be the highest-level
ax2.minorticks_on()
ax2.tick_params(top="on", right='on', which='both')
ax2.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # set xtick lockations (data coords)
ax2.set_xticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"])  # set xtick labels

ax2.set_xlim([-90, 90])  # Limit latitidues

ax2.set_aspect(1 / ax2.get_data_ratio())  # Makes the size of the axes equal (so apparently square)

ax2.set_ylabel(u.attrs["long_name"])  # set ylabel (breaks if long_name not present)

fig2.suptitle("Basic XY plot")  # add title

fig2.savefig("xy_GSNified.png")  # send graphics to PNG file