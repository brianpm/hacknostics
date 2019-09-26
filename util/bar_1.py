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
# print(f"reshape: {tmp.shape}")
# print(tmp[0,:])
# print("**")
# print(tmp[-1,:])
dateyearly = np.mean(tmp, axis=1)
print(dateyearly)
#
# create plot
#
# 1. try to do it like NCL ... not we actually need to do it wrong to do this.
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(dateyearly, yearly, color=None, edgecolor='black', fill=False)
fig.savefig("bar_1.png")

# #---------- First plot --------------------

#  res@tiMainString  = "Bar plot"
#  res@gsnXYBarChart = True            # Create bar plot

#  plot = gsn_csm_xy (wks,dateF(::8),dsoik(::8),res)

# #---------- Second plot --------------------

# # This is like drawing a regular curve, except with
# # flat bars for each point.

#  res@tiMainString             = "Bar plot with outlines"
#  res@gsnXYBarChartOutlineOnly = True

#  plot = gsn_csm_xy (wks,dateF(::8),dsoik(::8),res)

# #---------- Third plot --------------------

#  delete(res@gsnXYBarChartOutlineOnly)

# # When you include a reference line, then the bars
# # will be drawn pointing up or down, depending on
# # if they are above or below the ref line.

#  res@tiMainString = "Bar plot with a reference line"
#  res@gsnYRefLine  = 0.              # reference line   

#  plot = gsn_csm_xy (wks,dateF(::8),dsoik(::8),res)

# end