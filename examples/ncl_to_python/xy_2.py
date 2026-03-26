# ***********************************************
# xy_2.py
#
# Concepts illustrated:
#   - Drawing an XY plot with multiple curves
#   - Changing the line color for multiple curves in an XY plot
#   - Changing the line thickness for multiple curves in an XY plot
#   - Drawing XY plot curves with both lines and markers
#   - Changing the default markers in an XY plot
#   - Making all curves in an XY plot solid
#
# ***********************************************

import xarray as xr
import matplotlib.pyplot as plt

#---Read in data
# find out NCARG_ROOT for your system.
NCARG_ROOT = "/Users/brianpm/miniconda3/pkgs/ncl-6.6.2-h2f2bd2c_1"
f = xr.open_dataset(f"{NCARG_ROOT}/lib/ncarg/data/cdf/uv300.nc")
u = f["U"]                                    # get u data

# NCL: "To plot multiple lines, you must put them into a mulidimensional array."
# Python: much more flexibility. Here we'll just define two DataArrays
data0 = u.isel(time=0).sel(lon=82, method='nearest')
data1 = u.isel(time=0).sel(lon=69, method='nearest')

# 1 matplotlib will automatically change the color of the second line
fig, ax = plt.subplots()
fig.suptitle("Two curve XY plot")       # add title
ax.plot(u["lat"], data0) # create 1st line 
ax.plot(u["lat"], data1) # create 2nd line
fig.savefig("xy_2.png")


# 2 an example for using more control
xyLineThicknesses = [1.0,   2.0]        # make second line thicker
xyLineColors      = ["blue","red"]      # change line color

fig2, ax2 = plt.subplots()
for i, d in enumerate([data0, data1]):
    ax2.plot(d['lat'], d, color=xyLineColors[i], linewidth=xyLineThicknesses[i])
fig2.suptitle("Two curve XY plot")
fig2.savefig("xy_2a.png")


# 3 abstract a little more to do 3 curves.
choose_longitudes = [82, 0, -69]
xyMarkers      = ["o", "v", "s"]  # https://matplotlib.org/3.1.1/api/markers_api.html
xyMarkerColors = ["blue", "red", "green"]
xyLineStyles   = ["solid", "dotted", "dashed"] # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
xyLineColors   = ["skyblue", "pink", "palegreen"]  # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
fig3, ax3 = plt.subplots()
for i, ll in enumerate(choose_longitudes):
    ax3.plot(u['lat'], u.isel(time=0).sel(lon=ll, method='nearest'),
             color=xyLineColors[i], markerfacecolor=xyMarkerColors[i],
             marker=xyMarkers[i], linestyle=xyLineStyles[i])
fig3.suptitle("So Pretty")
fig3.savefig("xy_2b.png")
