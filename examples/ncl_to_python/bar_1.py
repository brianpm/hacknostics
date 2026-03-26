#*************************************************
# bar_1.py
#
# Concepts illustrated:
#   - Drawing bars instead of curves in an XY plot
#   - Changing the aspect ratio of a bar plot
#   - Drawing bars up or down based on a Y reference value
#
#************************************************
#
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
#************************************************

def yyyymm_to_yyyyfrac(d, offset=0.5):
    # todo: check the math on this.
    out = []
    for i in d.values:
        tmp = str(int(i))
        y = int(tmp[0:4])
        m = int(tmp[-2:])
        f = ((m+offset)-1)/12
        out.append(y+f)
    return np.array(out)

f = xr.open_dataset("/Users/brianpm/Downloads/SOI_Darwin.nc")
# note: I could not locate soi.nc as in NCL examples
date  = f['date']                  # YYYYMM
dsoid = f['DSOI']

dateF = yyyymm_to_yyyyfrac(date)  # <- this is an NCL specialty; replicated above

dimDate = date.shape         # number of dates
print(f"The shape of date is {date.shape}")

# the original was decadal, average
# usually you can use xarray to do this using time coords, but
# this dataset has just ints for time, so we can just reshape it adn average with np
yearly = np.mean(dsoid.values.reshape(dimDate[0]//12, 12), axis=1)

# convert integer YYYYMM to float
tmp = dateF.reshape(dimDate[0]//12, 12)
dateyearly = np.mean(tmp, axis=1)
print(dateyearly)
#
# create plot
#
# 1. try to do it like NCL,
#    EXCEPT, have the bars originate from zero because that makes sense.
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(dateyearly, yearly, color=None, edgecolor='black', fill=False)
fig.savefig("bar_1.png")


# 2. outline only
fig, ax = plt.subplots(figsize=(10,5))
ax.step(dateyearly, yearly, color='black')
fig.savefig("bar_2.png")


# Note: I don't know how to make the bars originate from some other value,
#       like they do by default in NCL. Nor do I know of why you would ever want 
#       such behavior. You can set the bottom keyword argument, but that appears
#       to adjust the whole data set.
#       Instead, I show a very simple change to the orientation of the bars.

# 3. Change orientation
fig, ax = plt.subplots(figsize=(10,5))
ax.barh(dateyearly, yearly, color='red', edgecolor='black')
fig.savefig("bar_3.png")


# 4. Color based on value
fig, ax = plt.subplots(figsize=(10,5))
ax.barh(dateyearly[yearly < 0], yearly[yearly < 0], color='lightblue', edgecolor='')
ax.barh(dateyearly[yearly > 0], yearly[yearly > 0], color='pink', edgecolor='')
fig.savefig("bar_4.png")

# 5. change width of bars
fig, ax = plt.subplots(figsize=(10,5))
barwidth = 0.33
ax.barh(dateyearly[yearly < 0], yearly[yearly < 0], height=barwidth, color='lightblue', edgecolor='')
ax.barh(dateyearly[yearly > 0], yearly[yearly > 0], height=barwidth, color='pink', edgecolor='')
fig.savefig("bar_5.png")

# 6. Cycle through colors
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(dateyearly, yearly, color=["yellow", "orange", "red"], edgecolor='black')
fig.savefig("bar_6.png")

# 7. Categorical data
x = [1,2,3,4,5,6,7,8]
y = [154900,56600,40000,30200,29700,24400,21700,13900]
labels = ["Lung","Colon/rectum","Breast","Prostate","Pancreas",
	"Non-Hodgkin's\n Lymphoma","Leukemias","Ovary"]
barcolors = ["firebrick","red","orange","green",
			"navy","blue","SkyBlue","SlateBlue"]
fig, ax = plt.subplots(constrained_layout=True)
ax.bar(x, y, color=barcolors)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45.)
ax.tick_params(axis='x', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.suptitle("Estimated Cancer Deaths for 2002")
fig.savefig("bar_7.png")

# 8. Stacked bar chart
cities = ["a","b","c"]
ncities = len(cities)
d1 = np.array([1.0, 2.0, 3.0])
d2 = np.array([0.3, 0.5, 0.7])
d3 = np.array([0.5, 0.5, 0.5])
d4 = np.array([1.0, 1.0, 1.0])
fig, ax = plt.subplots(figsize=(3,9), constrained_layout=True)
b1 = ax.bar(np.arange(ncities), d1, label="Q1")
b2 = ax.bar(np.arange(ncities), d2, bottom=d1, label="Q2")
b3 = ax.bar(np.arange(ncities), d3, bottom=d1+d2, label="Q3")
b3 = ax.bar(np.arange(ncities), d4, bottom=d1+d2+d3, label="Q4")

ax.set_xticks(np.arange(ncities))
ax.set_xticklabels(cities)
fig.legend(loc='upper center')
fig.savefig("bar_8.png")

# Example 9
# Compare with http://www.ncl.ucar.edu/Applications/Images/bar_22_lg.png
# note: We do not try to set the bottom of bars to be below zero,
#       as it makes no sense to do so.
times = np.array([3, 4, 5, 6])    # hours
time_strings = [f"{t:02d}:00" for t in times]
sflow = np.array([[0.0, 0.16, 0.20, 0.19],
                  [0.0, 0.15, 0.71, 0.61],
                  [0.0, 0.0,  0.25, 0.14], 
                  [0.0, 0.0,  0.14, 0.19]])
ntime = len(times)
titles = [f"Station {i}" for i in range(1,ntime+1)]
clrs = ['navy', 'firebrick', 'goldenrod', 'green']
ntime = len(times)
time_strings = [f"{t:02d}:00" for t in times]
titles = [f"Station {i}" for i in range(1,ntime+1)]
fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True, sharex=True, sharey=True)
aa = ax.ravel()
[a.set_facecolor('lightgray') for a in aa]
[a.grid(b=True, which='both', axis='both', color='white', zorder=0) for a in aa]
for i in range(sflow.shape[0]):
    aa[i].bar(times, sflow[i,:], edgecolor='black', color=clrs, zorder=50)
aa[-1].set_ylim([-.1, 1])
[aa[i].set_title(t) for i, t in enumerate(titles)]
aa[-1].set_xticks(times)
aa[-1].set_xticklabels(time_strings)
fig.text(-0.05, 0.5, 'STREAMFLOW', va='center', rotation='vertical')
fig.savefig("bar_9.png", bbox_inches='tight')  # bbox_inches option to get ytitle in file