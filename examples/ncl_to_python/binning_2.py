;***************************************************************
; binning_2.ncl
;
; Concepts illustrated:
;   - Create an array that spans the desired area
;   - Read data [ here, create bogus data]
;   - Loop over data and count instances of occurence
;   - Plot the data
;
;***************************************************************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"   
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"    

;**************************************************************************
;--- Create desired grid. Here, 2x2 but can be (say) 1x3 if dlat=1, dlon= 3 
;**************************************************************************
  latS =    0
  latN =   70
  lonW = -120
  lonE =    0

  dlat =  2.0                    
  dlon =  2.0

  nlat = toint((latN-latS)/dlat) + 1
  mlon = toint((lonE-lonW)/dlon) + 1

  lat  = fspan(latS, latN, nlat)
  lon  = fspan(lonW, lonE, mlon)

  lat@units = "degrees_north"
  lon@units = "degrees_east"

  count     = new( (/nlat,mlon/), "float", 1e20) 
  count!0   = "lat"
  count!1   = "lon"
  count&lat =  lat
  count&lon =  lon

  valavg    = count

;********************************************************************
;--- Read data ===> Here, create bogus data
;********************************************************************
  clat = random_normal( 23, 10, 10000)    
  clon = random_normal(-90, 10, 10000)
  cval = random_normal( 75, 20, 10000)

  clon = where(clon.lt.lonW, lonW, clon)  ; deal with bogus outliers 
  clon = where(clon.gt.lonE, lonE, clon)
  clat = where(clat.lt.latS, latS, clat)
  clat = where(clat.gt.latN, latN, clat)

;********************************************************************
;--- Bin count and sum; This assumes a simple rectilinear grid
;********************************************************************
  count  = 0
  valavg = 0

  npts = dimsizes(clat)

  do n=0,npts-1
     if (clat(n).ge.latS .and. clat(n).le.latN .and.  \
         clon(n).ge.lonW .and. clon(n).le.lonE .and.  \
         .not.ismissing(cval(n)) ) then

         jl = toint((clat(n)-latS)/dlat) 
         il = toint((clon(n)-lonW)/dlon) 
         count(jl,il)  = count(jl,il)  + 1
         valavg(jl,il) = valavg(jl,il) + cval(n)

     end if
  end do

 ;count@long_name = "Occurrence Count"
 ;count@units     = ""

  printVarSummary(count)
  print("count: min="+min(count)+"   max="+max(count))

  count  = where(count.eq.0, count@_FillValue,count)   ; don't divide by 0         

;********************************************************************
;--- Average
;********************************************************************
  
  valavg = valavg/count
 ;valavg@long_name = "..."
 ;valavg@units     = "..."

  printVarSummary(valavg)
  print("valavg: min="+min(valavg)+"   max="+max(valavg))

;********************************************************************
;--- Bin frequency (%)
;********************************************************************

  freq  = count
  freq  = (count/npts)*100
 ;freq@long_name = "frequency"
 ;freq@units = "%"

  printVarSummary(freq)
  print("freq: min="+min(freq)+"   max="+max(freq))

;************************************************
; create plot
;************************************************
  freq  = where(freq .eq.0,  freq@_FillValue, freq)   ; 

  plot  = new (3, "graphic")

  wks = gsn_open_wks("png","binning")            ; send graphics to PNG file

  res                       = True     ; plot mods desired
  res@gsnAddCyclic          = False    
  res@gsnDraw               = False    
  res@gsnFrame              = False    

  res@cnFillOn              = True     ; turn on color fill
  res@cnFillPalette         = "BlAqGrYeOrRe"     ; set color map
  res@cnFillMode            = "RasterFill"       ; Raster Mode
  res@cnLinesOn             = False    ; turn of contour lines

  res@lbOrientation         = "vertical"

  res@mpMinLatF             = latS
  res@mpMaxLatF             = latN
  res@mpMinLonF             = lonW
  res@mpMaxLonF             = lonE
  res@mpCenterLonF          = (lonE+lonW)*0.5
  res@mpGridAndLimbOn       = True  
  res@mpGridLineDashPattern = 2             ; Dashed lines
  res@mpGridLatSpacingF     = 5.0
  res@mpGridLonSpacingF     = 10.0

  res@cnLevelSpacingF       = 1.0      ; contour spacing
  res@gsnCenterString       = "Occurrence Count"
  plot(0) = gsn_csm_contour_map(wks,count, res)
  
  res@cnLevelSpacingF       = 0.05     ; contour spacing
  res@gsnCenterString       = "Frequency (%)"
  plot(1) = gsn_csm_contour_map(wks,freq , res)
  
  res@cnLevelSpacingF       =  5.0     ; contour spacing
  res@gsnCenterString       = "Average"
  plot(2) = gsn_csm_contour_map(wks,valavg,res)

  resP                      = True               ; modify the panel plot
  resP@gsnMaximize          = True
;;resP@gsnPanelMainString   = "A common title"
  gsn_panel(wks,plot,(/3,1/),resP)               ; now draw as one plot

