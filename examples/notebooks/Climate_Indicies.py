# %%
from pathlib import Path
import xarray as xr

# %%
diri   = Path("/glade/campaign/cgd/ccr/E3SM-LE")

case = "20200303.en7.LE_ens.ne30_oECv3_ICG.cori-knl" 

# print(list((diri / case).glob('*')))

to_data = "/atm/proc/tseries/month_1"

p = diri / case 
f = f"{to_data}" # /{case}.cam.h0.TS.*.nc"
print(f)
# fili   = sorted(list((diri / case ).glob(f))
# print(fili)
# ds     = xr.open_mfdataset(fili)


# %%

#*********************************
# The index code below assumes that each year has 12 months 
#*********************************
X      = mon_fullyear( in->SST(:,{latS:latN},{lonL:lonR}), 0)   ; all times on file

  YYYYMM = cd_calendar(X&time, -1)      ; ALL dates assciated with X
  tStrt  = ind(YYYYMM.eq.ymStrt)        ; indices of selected times
  tLast  = ind(YYYYMM.eq.ymLast)
  delete(YYYYMM)

  x      = X(tStrt:tLast,:,:)           ; subset to desired time interval
  yyyymm = cd_calendar(x&time, -1) 
  dimx   = dimsizes(x)
  ntim   = dimx(0)

  delete(X)                             ; no longer needed

#*********************************
# time indices for base climatology 
#*********************************

  iClmStrt = ind(yyyymm.eq.clStrt)     
  iClmLast = ind(yyyymm.eq.clLast)    
 #print(yyyymm(iClmStrt:iClmLast))

#*********************************
# Climatology and anomalies from base climatology   
#*********************************

  xClm     = clmMonTLL(x(iClmStrt:iClmLast,:,:))
  printVarSummary(xClm)

  xAnom    = calcMonAnomTLL (x,  xClm ) 
  xAnom@long_name = "SST Anomalies"
  printVarSummary(xAnom)

#*********************************
# Unweighted areal averages & anomalies (time series)
# Small latitudinal extent so no need to weight    
#*********************************

  x_avg     = wgt_areaave_Wrap(x    , 1.0, 1.0, 1)
  x_avg@long_name = "areal avg"

  xAnom_avg = wgt_areaave_Wrap(xAnom, 1.0, 1.0, 1)
  xAnom_avg@long_name = "areal avg anomalies"

  printVarSummary(xAnom_avg)

#*********************************
# Compute standardized anomalies; use clm period    
#*********************************

  xAnom_std = xAnom_avg
  xAnom_std = xAnom_avg/stddev(xAnom_avg(iClmStrt:iClmLast))
  xAnom_avg@long_name = "areal avg standardized  anomalies"
  printVarSummary(xAnom_std)

#*********************************
# Perform an unweighted nrun-month running average on the index
# 2 months lost at start & end if endopt=0  ... reflective if endopt=1
#*********************************

  endopt    = 1
  ii = ind(.not.ismissing(xAnom_std))
  xAnom_std(ii) = runave_n_Wrap (xAnom_std(ii), nrun, endopt, 0)

  print(yyyymm+"   "+x_avg+"   "+xAnom_avg+"   "+xAnom_std)
