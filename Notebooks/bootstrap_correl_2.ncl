;************************************************
; bootstrap_correl_2.ncl
;
; Concepts illustrated:
;   - Read SOI (Southern Oscilliation Index) Signal   
;   - Read gridded surface temperature         
;   - Calculate the monthly climatology and then anomalies
;   - Calculate conventional cross correlation (SOI vs Anomalies)
;   -   Calculate 2.5% and 97.5% bounds via conventional statistical methods
;   - Calculate bootstrapped cross correlations
;   -   Calculate 2.5% and 97.5% bounds via bootstrap methods 
;   - Create panel plot
;*************************************************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"   
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
;*************************************************
;--- Read SOI data file and extract LSAT
;*************************************************
  dSOI   = "./" 
  fSOI   = "SOI.nc"
  pSOI   = dSOI+fSOI
  f      = addfile(pSOI, "r")
  soi    = f->SOI_SIGNAL(984:1751)        ; 194801 - 201112
  nsoi   = dimsizes(soi)
  printVarSummary(soi)

  yyyymm = cd_calendar(soi&time, -1)
  ymStrt = yyyymm(0)
  ymLast = yyyymm(nsoi-1)
  print("ymStrt="+ymStrt+"  ymLast="+ymLast)

;***********************************************************
;--- Calculate some basic statistics for the SOI
;    These statistics are not used .... just background statistics
;***********************************************************
  soiStat4   = dim_stat4_n(soi, 0) ; 1st 4 moments of original sample
                                   ; for clarity; explicitly extract
  soiAvg     = soiStat4(0)         ; original sample mean
  soiStd     = sqrt(soiStat4(1))   ;    "     sample std dev
  soiSkew    = soiStat4(2)         ; skewness; departure from symmetry
  soiKurt    = soiStat4(3)         ; kurtosis; relative to a normal distribution
  print("SOI KURTOSIS : "+soiKurt)
  exit
  soiStdErr  = soiStd/nsoi

  soiMed     = dim_median_n(soi,0) ; median of original sample

  df         = nsoi-1
  p          = 0.975               ; match default bootstrap 'p' (0.025, 0.975)
  tsoi       = cdft_t(p,df)        ; 2-sided   
  soiLow     = soiAvg-tsoi*soiStd  ; normal low:  2.5% 
  soiHi      = soiAvg+tsoi*soiStd  ; normal hi ; 97.5%

;***********************************************************
;--- Read variable
;    Calculate monthly climatology; Calculate monthly anomalies
;***********************************************************

  diri  = "./"
  fili  = "air.mon.mean.nc"
  pthi  = diri+fili
  f     = addfile(pthi, "r")
  printVarSummary(f->air)
;   y     = short2flt(f->air(0:767,0,:,:)) ; [time | 774] x [lat | 73] x [lon | 144]  
  y = f->air(0:767,:,:)  ; UPDATED NOAA FILE IS FLOAT. ALSO NOTE THAT THE NEW FILE IS [time lat lon], so example has extra dimension(!?)
  yClm  = clmMonTLL(y)                   ; [time |  12] x [lat | 73] x [lon | 144]
  yAnom = calcMonAnomTLL(y, yClm)        ; [time | 774] x [lat | 73] x [lon | 144]

  dimy  = dimsizes(y)
  ntim  = dimy(0)                        ; 768: 194801-201112

  if (ntim.ne.nsoi) then
      print("time mismatch: ntim="+ntim+"  nsoi="+nsoi)
      exit
  end if
  
;***********************************************************
;--- Compute conventional sample correlation('r') at each grid point
;    Assumes bivariate normal distribution
;    'r' distribution is skewed, hence, use Fischer z-transform
;***********************************************************

  r    = escorc_n(soi,yAnom,0,0)    ; [73] x [144]
  r@long_name = "correlation: escorc"

  df   = nsoi-2
  rse  = sqrt(1-r^2)/sqrt(df)       ; [73] x [144]
  rse@long_name = "correlation: escorc: std. error"

  z    = 0.5*log((1+r)/(1-r))       ; [73] x [144] 
  z@long_name = "Fischer z-transform"

  zse  = 1.0/sqrt(nsoi-3)           ; [1]
  zse@long_name = "standard error of z statistic"

;***********************************************************
;--- Add coordinate meta data
;***********************************************************

  copy_VarCoords(y(0,:,:), r)   ; create (copy) spatial coordinates
  printVarSummary(r)            ; [lat | 73] x [lon | 144]
  copy_VarCoords(y(0,:,:), rse) ; create (copy) spatial coordinates
  printVarSummary(rse)          ; [lat | 73] x [lon | 144]
  copy_VarCoords(y(0,:,:), z)   ; create (copy) spatial coordinates
  printVarSummary(z)            ; [lat | 73] x [lon | 144]

;***********************************************************
;--- Conventional (normal) estimates; use z-transformed values
;--- Get t-value for specified 'ptst' : Add meta data
;***********************************************************

  ptst = 0.975                  ; match default bootstrap 'p' (0.025, 0.975)
  zval = cdft_t(ptst,9999)      ; 2-sided; use 'big' number for normal dist   
  zLow = z-zval*zse             ; z-normal low:  2.5% 
  zHi  = z+zval*zse             ; z-normal hi ; 97.5%

                                ; transform z back to 'r' space
  rLow = tanh(zLow)             ; =(exp(2*zlow)-1)/(exp(2*zlow)+1)
  rHi  = tanh(zHi)              ; =(exp(2*zhi )-1)/(exp(2*zhi )+1)
  rLow@long_name = "Correlation:  2.5%"
  rHi@long_name  = "Correlation: 97.5%"

  tval = r*sqrt(df/(1-r^2))     ; conventional t-value
  psoi = student_t(tval,df)     ; probability
  psoi@long_name = "Conventional Probability"

  copy_VarCoords(y(0,:,:), rLow); create (copy) spatial coordinates
  copy_VarCoords(y(0,:,:), rHi )
  copy_VarCoords(y(0,:,:), psoi)

  printVarSummary(rLow)
  printVarSummary(rHi)
  printVarSummary(psoi)

;***********************************************************
;--- BOOTSTRAP
;    Extract from returned 'list' vatiable; Add meta data
;***********************************************************

  N     = ntim                  ; convenience
  n     = N                     ; no subsampling
  nBoot = 500                   ; test with nBoot=10
  opt   = False   
  if (n.ne.N) then
      opt   = True
      opt@sample_size = n     
  end if

  Boot_r     := bootstrap_correl(soi, yAnom, nBoot, (/0,0/), opt)
  rBoot       = Boot_r[0]       ; bootstrapped correlations 
  rBootAvg    = Boot_r[1]       ; average of bootstrapped correlations 
  rBootStd    = Boot_r[2]       ; std dev of bootstrapped correlations 
  delete(Boot_r)                ; no longer needed

  rBootLow    = bootstrap_estimate(rBoot, 0.025, False)   ;  2.5% lower confidence bound 
  rBootMed    = bootstrap_estimate(rBoot, 0.500, False)   ; 50.0% median of bootstrapped estimates
  rBootHi     = bootstrap_estimate(rBoot, 0.975, False)   ; 97.5% upper confidence bound
  
  rBootLow@long_name = "bootstrap r: "+rBootLow@estimate
  rBootMed@long_name = "bootstrap r: Median"
  rBootHi@long_name  = "bootstrap r: "+rBootHi@estimate

  printVarSummary(rBoot)
  printMinMax(rBoot, 0)
  printVarSummary(rBootAvg)
  printMinMax(rBootAvg, 0)

;+++++++++++++++++++++++++++++++++++
; PLOTS
;+++++++++++++++++++++++++++++++++++
  pltDir  = "./"
 ;pltName = "BootCor_SOI_"+N+"_"+n+"_"+nBoot
  pltName = "bootstrap_correl"
  pltPath = pltDir+pltName
  pltType = "png"             ; send graphics to PNG file
  wks  = gsn_open_wks (pltType,pltPath)

;***************************************************************
;--- create histogram for the SOI_SIGNAL
;***************************************************************

  resh              = True
  resh@gsnDraw      = False
  resh@gsnFrame     = False
  resh@gsnHistogramNumberOfBins = 21
  resh@gsFillColor  = "green"

  resh@tiMainString = "Original: N="+N
  hstSoi             = gsn_histogram(wks, soi ,resh)  

;***************************************************************
;--- text object SOI_SIGNAL
;***************************************************************

  txres                       = True
  txres@txFont                = "helvetica-bold"
  txres@txFontHeightF         = 0.0150
  textSoi  = (/"  Mean="+sprintf("%5.1f", soiAvg) +"~C~"+  \
               "   Std="+sprintf("%5.1f", soiStd) +"~C~"+  \
               "StdErr="+sprintf("%5.1f", soiStdErr) +"~C~"+  \
               "  Skew="+sprintf("%5.1f", soiSkew) +"~C~"+  \
               "  Kurt="+sprintf("%5.1f", soiKurt) +"~C~"+  \
               "   Low="+sprintf("%5.1f", soiLow) +"~C~"+  \
               "Median="+sprintf("%5.1f", soiMed) +"~C~"+  \
               "    Hi="+sprintf("%5.1f", soiHi ) /)
  txBoxSoi  = gsn_create_text(wks,textSoi,txres)

  amres  = True
  amres@amParallelPosF   =  0.35             ; move legend to the right
  amres@amOrthogonalPosF = -0.275            ; move the legend up
  annoSoi = gsn_add_annotation(hstSoi, txBoxSoi, amres)  ; Attach string to plot

  draw(hstSoi)
  frame(wks)

;***************************************************************
;--- Global correlations of SOI_SIGNAL and monthly anomalies
;***************************************************************

  rescn                     = True
  rescn@gsnDraw             = False
  rescn@gsnFrame            = False

  rescn@mpFillOn            = False            ; no need   
  rescn@mpCenterLonF        = 180

  rescn@cnLevelSelectionMode= "ManualLevels"   ; manual set levels
  rescn@cnMinLevelValF      = -0.9
  rescn@cnMaxLevelValF      =  0.9
  rescn@cnLevelSpacingF     =  0.1   
  rescn@cnFillOn            = True             ; color fill plot
  rescn@cnLinesOn           = False
  rescn@cnLineLabelsOn      = False
  rescn@cnInfoLabelOn       = False
  rescn@cnFillPalette       = "MPL_RdBu" ; "precip4_diff_19lev"
  if (rescn@cnFillPalette .eq. "precip4_diff_19lev") then
      cmap = read_colormap_file("precip4_diff_19lev")
      rescn@cnFillPalette := cmap(::-1,:)      ; reverse the color map:  Blue -> Red
  end if
  rescn@lbLabelBarOn        = False            ; turn off individual label bars

  rescn@gsnStringFontHeightF= 0.02

;---Create arrays to hold series of plots
  plots    = new(6,graphic)    
  plots(0) = gsn_csm_contour_map(wks,rLow,rescn)
  plots(1) = gsn_csm_contour_map(wks,rBootLow,rescn)
  plots(2) = gsn_csm_contour_map(wks,r   ,rescn)
  plots(3) = gsn_csm_contour_map(wks,rBootMed,rescn)
  plots(4) = gsn_csm_contour_map(wks,rHi ,rescn)
  plots(5) = gsn_csm_contour_map(wks,rBootHi ,rescn)

;---Resources for paneling
  pres                    = True              ; modify the panel plot
  pres@gsnPanelMainString = "SOI-TempAnom Cor: "+ymStrt+"-"+ymLast+ \
                  ": Conventional/BootStrap: nBoot="+nBoot+": N="+N+", n="+n 
  pres@gsnPanelMainFontHeightF      = 0.0175            ; default 0.05
  pres@gsnPanelLabelBar   = True              ; add common colorbar
  gsn_panel(wks,plots,(/3,2/),pres)    