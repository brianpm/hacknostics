#*********************************************
# box_1.ncl
#
# Concepts illustrated:
#   - Drawing box plots
#   - Explicitly setting the X tickmark labels in a box plot
#
#*********************************************
#
import numpy as np
import matplotlib.pyplot as plt

# Create some fake data

# NOTE: 
# NCL's boxplot function takes horizontal (x) position,
# and then a vector of the values used for the whiskers and box.
# We could go through the motions of recreating that functionality,
# but I don't see how that would be useful. Instead, let's make some
# random data arrays that have similar properties as the ones from the
# box_1.ncl example.
N = 1000
x = np.array([-3., -1., 1.])	

yval = np.zeros((len(x), N))
yval[0, :] = np.random.normal(loc=1.5, scale=2.0, size=N)
yval[1, :] = np.random.normal(loc=1., scale=1.25, size=N)
yval[2, :] = np.random.weibull(1.11, size=N)
# yval = np.array([3, 5], dtype="float")
# yval[0, 0] = -3.
# yval[0, 1] = -1.
# yval[0, 2] = 1.5
# yval[0, 3] = 4.2
# yval[0, 4] = 6.

# yval[1, 0] = -1.
# yval[1, 1] = 0.
# yval[1, 2] = 1.
# yval[1, 3] = 2.5
# yval[1, 4] = 4.

# yval[2, 0] = -1.5
# yval[2, 1] = 0.
# yval[2, 2] = .75
# yval[2, 3] = 2.
# yval[2, 4] = 6.5

	
#**********************************************
# create plot
#**********************************************
fig, ax = plt.subplots()
tmXBLabels = ["Control", "-2Xna", "2Xna"]  # labels for each box
tiMainString = "Default Box Plot"
plot = ax.boxplot(yval.transpose(), vert=True, labels=tmXBLabels)
fig.savefig("/Users/brianpm/Desktop/box_1.png")

# "box2"
# This changes the style of the plot,
# each box/whisker gets a color change and is narrower than in box_1
# Easiest way to do this is probably to plot each one separately.
# We set properties of each part of the box.
lc = ['green', 'blue', 'red']
fig, ax = plt.subplots()
tmXBLabels = ["Control", "-2Xna", "2Xna"]  # labels for each box
tiMainString = "Tailored Box Plot"
for i in range(3):
  print(f"i = {i}; Label = {tmXBLabels[i]}; shape of data: {yval[i,:].shape}")
  ax.boxplot(yval[i,:], positions=[i+1], vert=True, widths=0.05,
             boxprops=dict(color=lc[i]),
             whiskerprops=dict(color=lc[i], linestyle='dashed'),
             medianprops=dict(color=lc[i]),
             flierprops=dict(markeredgecolor=lc[i]),
             capprops=dict(color=lc[i]))
ax.set_xlim([0,5])
fig.savefig("/Users/brianpm/Desktop/box_2.png")

# "box3" just adds some markers
# This is very trivial, so I don't worry about the details here,
# but I include some styling on the markers, and a couple ways to specify style.
fig, ax = plt.subplots()
tmXBLabels = ["Control", "-2Xna", "2Xna"]  # labels for each box
tiMainString = "Tailored Box Plot"
for i in range(3):
  print(f"i = {i}; Label = {tmXBLabels[i]}; shape of data: {yval[i,:].shape}")
  ax.boxplot(yval[i,:], positions=[i+1], vert=True,
             boxprops=dict(color=lc[i]),
             whiskerprops=dict(color=lc[i], linestyle='dashed'),
             medianprops=dict(color=lc[i]),
             flierprops=dict(markeredgecolor=lc[i]),
             capprops=dict(color=lc[i]))
ax.set_xlim([0,5])

ax.plot(1, -3., 'xr') 
ax.plot(2, -1., 'xr') 
ax.plot(2, 1., 'xr') 

gsMarkerColor = "navy"              
gsMarkerIndex = "o"                        
gsMarkerSizeF = 15.                      

ax.plot(1, 1.5, marker=gsMarkerIndex, markerfacecolor=gsMarkerColor, markeredgecolor='none', markersize=gsMarkerSizeF) 
ax.plot(2, 2, marker=gsMarkerIndex, markerfacecolor=gsMarkerColor, markeredgecolor='none',markersize=gsMarkerSizeF) 
ax.plot(3, 5, marker=gsMarkerIndex, markerfacecolor=gsMarkerColor, markeredgecolor='none',markersize=gsMarkerSizeF) 

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(tmXBLabels)
fig.savefig("/Users/brianpm/Desktop/box_3.png")


# box_4
# This doesn't do anything with boxes either,
# just adds some text.
fig, ax = plt.subplots()
tmXBLabels = ["Control", "-2Xna", "2Xna"]  # labels for each box
tiMainString = "Tailored Box Plot"
for i in range(3):
  print(f"i = {i}; Label = {tmXBLabels[i]}; shape of data: {yval[i,:].shape}")
  ax.boxplot(yval[i,:], positions=[i+1], vert=True, widths=0.05,
             boxprops=dict(color=lc[i]),
             whiskerprops=dict(color=lc[i], linestyle='dashed'),
             medianprops=dict(color=lc[i]),
             flierprops=dict(markeredgecolor=lc[i]),
             capprops=dict(color=lc[i]))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.text(0.67,0.75, "May", fontsize=18, ha='left')
fig.text(0.67, 0.71, "CTL", fontsize=12)
fig.text(0.75, 0.71, "-2Xna", fontsize=12)
fig.text(0.85, 0.71, "2Xna", fontsize=12)
fig.text(0.67,0.67,"Random Numbers", fontsize=15, ha='left')
fig.text(0.67, 0.63, "3.21", fontsize=12)
fig.text(0.75, 0.63, "3.45", fontsize=12)
fig.text(0.85, 0.63, "3.63", fontsize=12)
ax.set_xlim([0,5])
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(tmXBLabels)
fig.savefig("/Users/brianpm/Desktop/box_4.png")

# box_5.ncl: Shows how to calculate and plot the median, minimum value, 
# maximum value, and the 25th and 75th percentiles of two timeseries.
# NCL: nor  = random_normal(292,15,(/125/))
nor1 = np.random.normal(loc=292, scale=15.0, size=125)
# NCL: dimt = dimsizes(nor)
dimt = len(nor1)

x25  = round(.25*dimt)-1   
x75  = round(.75*dimt)-1
nor2 = np.random.normal(295, 14, 125)

nor1.sort()
nor2.sort()
iarr=np.empty((2, 5))
# NOTE:
# This method mirrors NCL's example very closely,
# but requires us to do sort step, and to approximate
# the location of the 25th and 75th percentiles.
# We could also use numpy to directly do that,
# which might be faster if they have optimized code underneath.
iarr[0, :] = [nor1.min(), nor1[x25], np.median(nor1), nor1[x75], np.max(nor1)]  
iarr[1, :] = [nor2.min(), nor2[x25], np.median(nor2), nor2[x75], np.max(nor2)]

compare_with = np.percentile(nor1, [25, 50, 75])
print(f"25th percentile: index method: {nor1[x25]}  np.percentile: {compare_with[0]}")
print(f"MEDIAN: np.median: {np.median(nor1)}  np.percentile: {compare_with[1]}")
print(f"75th percentile: index method: {nor1[x75]}  numpy func: {compare_with[2]}")
# NOTE: these do give nearly the same answers, and depends on whether the 
# numpy function finds the percentile between points and does interpolation
# versus the indexing approach that just takes one of the array's values.

# the NCL example then plots these, but as we have seen
# the function signature is different, so it doesn't really make 
# sense to just repeat the plot here.

# box_6.ncl 
# Mostly this example shows how to read ascii files,
# and uses stat_dispersion to get the box plot values.
# I could not locate the file "pr.djf.cx2.txt" so I skip this example.

# box_7.ncl
# Uses the same data as box_6.ncl and the colors in box_2.ncl
# Shows some text formatting, but that is MUCH easier in matplotlib.

# box_8.ncl
# An elaborate example that requires downloading a supplementary ncl file. 
# Below I replicate the data and show how to make a similar plot called a violin plot.
# I didn't add all the customization of the NCL example, but this demonstrates
# how far you can get with python/numpy/matplotlib in a very small number of lines.
data = np.empty((9, 28))
data[0,:]  = [-999.0, -999.0, -999.0,    0.9, -999.0,    1.4,    1.6,    1.1,    0.9, -999.0, -999.0, -999.0,    0.9,    0.8, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0,    1.1, -999.0, -999.0, -999.0, -999.0, -999.0]
data[1,:]  = [   1.6,    0.9,    1.5,    1.2,    1.3,    1.4,    1.6,    1.0, -999.0, -999.0,    1.3,    0.7, -999.0, -999.0,    0.7,    1.1,    2.0,    0.7,    1.2,    1.2,    1.4,    0.9,    1.3,    1.7,    1.4,    1.0,    1.3, -999.0]
data[2,:]  = [   1.6,    0.9,    1.8,    1.5,    1.3,    1.6,    1.9,    0.9, -999.0, -999.0,    1.1,    0.8, -999.0, -999.0,    0.8,    1.3,    2.2,    0.8,    1.4,    1.4,    1.4,    1.0,    1.6,    1.9,    1.4,    1.0,    1.0,    1.8]
# data[3,:]  = [-999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0]
data[3,:]  = [-999.0, -999.0, -999.0,    1.2, -999.0,    1.6,    1.5,    1.1,    1.1, -999.0, -999.0, -999.0,    1.0,    1.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0,    1.3, -999.0, -999.0, -999.0, -999.0, -999.0]
data[4,:]  = [   2.0,    1.4,    2.4,    1.7,    1.8,    1.6,    2.3,    1.1, -999.0, -999.0,    1.7,    0.9, -999.0, -999.0,    1.3,    1.7,    2.9,    0.9,    1.8,    1.6,    1.3,    1.4,    1.9,    2.3,    1.8,    1.2,    1.7, -999.0]
data[5,:]  = [   2.6,    1.8,    2.9,    2.4,    2.6,    2.4,    2.8,    1.8, -999.0, -999.0,    2.1,    1.7, -999.0, -999.0,    1.6,    2.1,    3.2,    1.6,    2.2,    2.0,    1.9,    1.9,    2.5,    2.8,    2.7,    2.0,    2.0,    2.8]
# data[7,:]  = [-999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0]
data[6,:]  = [-999.0, -999.0, -999.0,    1.1, -999.0,    1.2,    1.6,    1.0,    1.0, -999.0, -999.0, -999.0,    1.0,    0.9, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0, -999.0,    1.2, -999.0, -999.0, -999.0, -999.0, -999.0]
data[7,:]  = [   2.5,    1.8,    2.3,    2.1,    2.4,    2.0,    2.8,    1.7, -999.0, -999.0,    2.2,    1.5, -999.0, -999.0,    1.8,    2.0,    3.2,    1.4,    2.3,    1.8,    2.1,    1.9,    2.3,    2.8,    2.5,    1.8,    2.1, -999.0]
data[8,:] = [   4.7,    3.3,    4.7,    3.8,    4.2,    4.3,    4.7,    3.4, -999.0, -999.0,    3.6,    2.9, -999.0, -999.0,    3.1,    3.6,    5.4,    3.0,    4.0,    3.4,    3.7,    3.5,    4.1,    4.8,    4.2,    3.5,    3.5,    4.6]
data = np.where(data==-999.0, np.nan, data)
# Filter data using np.isnan BECAUSE the statistics that are calculated don't like nans.
mask = ~np.isnan(data)
filtered_data = [d[m] for d, m in zip(data, mask)]  # makes a list of arrays ... mimics a "ragged" array for our purposes
# print(filtered_data)
fig, ax = plt.subplots(constrained_layout=True)
VP = ax.violinplot(filtered_data, showmedians=True)
clrs = ['blue', 'blue', 'blue', 'gray', 'gray', 'gray', 'red', 'red', 'red']
for i, b in enumerate(VP['bodies']):
  b.set_facecolor(clrs[i])

ax.set_xticks(np.arange(1,9,1))
ax.set_xticklabels(["RCP2.6", "RCP4.5", "RCP8.5", "RCP2.6", "RCP4.5", "RCP8.5", "RCP2.6", "RCP4.5", "RCP8.5"], rotation=90, ha='right')

fig.savefig("/Users/brianpm/Desktop/box_8.png")