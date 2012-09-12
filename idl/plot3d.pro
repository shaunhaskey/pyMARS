

; IDL MAIN PROGRAM       >>>   plot3d   <<<
; = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
;			        Begun:         M. Schaffer, 2004 Dec 14
;				Last Modified: M. Schaffer, 2007 May 29
;
; PLOT3D reads a data file written by the fortran program SURFM, of
;  poloidal and toroidal Fourier analysis on many magnetic surfaces,
;  and makes color contour plots from the spectra.
;
; Inputs:
;  wfile    = 'path to surfmn.out.idl3d file'
;  title    = 'alphanumeric data' that will be printed above plot
;  subtitle = 'subtitle text' that will be printed below plot
;  n_mode   = sets toroidal mode number of data to be plotted
;  msign    < 0 changes sign of m (x axis) on plot
;  qsign    +1 is normal, -1 changes sign of q curve on plot, 
;	    0 plots + and -; any other number scales q curve on plot
;  xrange   = e.g. [-20,20] sets m range 
;  yrange   = e.g. [0.5,1.0] sets radial variable range
;  zrange   = e.g. [5.0,10.0] sets range of contours; ignored if the two 
;                limits are equal; 1st element must be smaller, 2nd larger
;  c_labls = [0,1,0,1,0,1,0,1,0,1,0]  ; 1 or 0 to label 11 contours or not
;  dxgrid  = spacing between vertical grid lines; no lines if dxgrid le 0
;  dygrid  = spacing between horizontal grid lines; no lines if dygrid le 0
;
; In the following: cl_ = color (0=blk, 255=wh);  th_ = thickness 
;  clc = 255  &  thc = 2	; contour line color & thickhness
;  clg = 000  &  thg = 2.0	; grid line      "   &     "
;  clq = 255  &  thq = 4	; q curve        "   &     "
;
;  lp = 'l' for landscape, 'p' for portrait plot
;  username = 'F. Lastname'
;  psname   = 'some_name.ps'
;  epsname  = 'some_name.eps'
;
; Outputs:
;
; To run:   IDL> .r plot3d
;
; = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
;
;			MAIN  PROGRAM  PLOT3D 
;
; = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


device, decomposed = 0		; gets around the IDL vs Mac color war
				;  when running from a Mac

; Set some default values

 wfile = 'surfmn.out.idl3d'	; name of magnetic data file to be read
 n_mode = 1			; default toroidal mode number
 nst = 0  &  nfpts = 0		; initialize
 irpt = 0  &  iradvar = 0
 khand = 0
 msign = 1.0  &  qsign = 1.0
 dxgrid = 0  &  dygrid = 0
 username = 'username'
 gfile = 'g-file'

 ps = 0
 ddt = systime(0)		; day-of-week date time
 time = strmid(ddt,4,20)	; date time

 @plotcommon.in  		; execute file with common information
 @plot3d.in			; execute input file


 if (xrange[0] ge xrange[1]) or (yrange[0] ge yrange[1]) then begin
  print, ''
  print, '  XRANGE and/or YRANGE incompatibility'
  print, ''
  goto, FINISH
 endif


; GET DATA from file specified by wfile . . . . . . . . . . . . . . . .

; READ header data
 openr, 1, wfile	 	; a fortran-written formatted file
 readf, 1, $			; READ HEADER DATA from 1st record
  format= '(1x,5i8,2x,A)', $
  nst, nfpts, irpt, iradvar, khand, gfile

 print, '     nst   nfpts    irpt iradvar   khand   gfile'
 print,  nst,nfpts,irpt,iradvar,khand, gfile		;<< diagnostic


 imax = nst-1			; assign array range variables
 jmax = 2*nfpts
 kmax = nfpts

 ms = indgen(2*nfpts+1) - nfpts		; array of values of m
 ns = indgen(nfpts+1)			; array of values of n

; READ 1D arrays of radial points and qpsi

 rvals = fltarr(nst)		; values of surface radial variable
 readf, 1, $			; READ RADIAL POINTS from 2nd record.
  format= '(1x,257f9.4)', rvals	;  up to maximum of 257 points

 qvals = fltarr(nst)		; values of EFIT q on the surfaces
 readf, 1, $			; READ q values from 3rd record,
  format= '(1x,257f9.4)', qvals	;  up to maximum of 257 points
  
 
; READ SURFMN mode amplitudes from file and
;  insert into the 3D IDL array adat(i,j,k)
;  All indices start with 0.
;   i = radial coordinate index
;   j = poloidal mode number 'm' index
;   k = toroidal mode number 'n' exactly

 atmp = fltarr(kmax+1)			; array for single data records
 adat = fltarr(imax+1,jmax+1,kmax+1)	; 3d array for data

for i=0,imax  do begin	   
 for j=0,jmax  do begin

  readf, 1, atmp		; read one record

  for k=0,kmax  do begin
   adat[i,j,k] = atmp[k]	; add record to adat
  endfor

 endfor
endfor 


; READ 1D arrays of radial points and qline

 qlvals = fltarr(nst)			; values of qline on the surfaces
 readf, 1, $				; up to maximum of 257 points
  format= '(1x,257f9.4)', qlvals  


close, 1 

; END OF READS from SURFMN.OUT file . . . . . . . . . . . . . . . . . .


; PREPARE DATA FOR PLOTTING . . . . . . . . . . . . . . . . . . . . . .
; Set up 2d array  zdat  of data to contour and plot
; Order of array is zdat(Xdependence,Ydependence)
; We also set up full 2d arrays for X and Y data, to allow for possible
;  nonuniform data in the future.

 if msign lt 0 then msign = -1 else msign = +1

 zdat  = adat(*,*,n_mode)	; zdat(i,j) for toroidal mode n_mode
 szdat = size(zdat)		; size[1], size[2] are 1st, 2nd dimensions
 xdat  = fltarr(szdat(1),szdat(2))
 ydat  = fltarr(szdat(1),szdat(2))

 for i=0,imax  do begin		; fill imax r-columns with ms
  xdat[i,*] = msign*ms
 endfor
 
 for j=0,jmax  do begin		; fill jmax m-rows with rvals
  ydat[*,j] = rvals
 endfor

; Find limits of z data range
;  Find where x and y values are within the requested plot range
 xind = where(((xdat[0,*] ge xrange[0]) and (xdat[0,*] le xrange[1])), xcount)
 yind = where(ydat[*,0] ge yrange[0] and ydat[*,0] le yrange[1], ycount)
 sxind = size(xind)     &     syind = size(yind)

 if sxind[0] le 0 or syind[0] le 0 then begin
  print, 'XRANGE AND/OR YRANGE DO NOT OVERLAP DATA.'
  goto, FINISH
 endif
 
 zreduce1 =     zdat[yind,*]
 zreduce2 = zreduce1[*,xind]
 zmax = abs(max(zreduce2))
 zmin = 0. 
 
 if zmax le 0.0 then begin
  print, 'NO DATA GREATER THAN ZERO IN ZDAT.'
  goto, FINISH
 endif

;; print, ''   &   print, zdat			;<< diagnostic



; = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
; Loop through the plot sequence 3 times, plotting successively to the
; X-window, a PostScript file, and an EPS file. The three plots are 
; identified by ps=0, ps=1, and ps=2, respectively.
; = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

ps = 0		; start loop with print to X-window

FOR ps = 0,2 DO BEGIN

; Set set parameters for plot and output device . . . . . . . . . . . .

!p.multi = [0,2,1]			; 2 columns, 1 row of plots, for
					; main contour plot plus color bar

if ps le 0 then begin		; FOR X-WINDOW
 set_plot, 'x'					; for X-Window plots
 !p.font = -1					; default vector fonts
 !p.charsize = 1.3				; font size multiplier
; use default screen window (which is landscape), but bigger than default size
 window, 0, xsize=850, ysize=600		; open screen window
endif			; end of  if ps le 0 then begin

if ps gt 0 then begin		; FOR POSTSCRIPT
 set_plot, 'ps'					; to plot PS files 
 !p.font=0					; enable PS fonts
 device,/helvetica,/narrow,/bold
 device,bits_per_pixel = 8
  
;  device,/landscape				; for bw PS LANDSCAPE plots
  device,/landscape,/color			; for color PS LANDSCAPE
;  device,/inches, xsize=9.5, ysize=6.7		; plot size (100% range *)
  device,/inches, xsize=9.5, ysize=6.7		; plot size
;  device,/inches, xoffset=0.75, yoffset=0.75	; plot's lower left corner
  device,font_size=12				; font size (pt)
  !p.charsize = 1.0				; font size multiplier
endif			; end of  if ps gt 0 then begin
   ; * Notes:
   ;   xsize = from y-axis label to plot right edge, w/ some extra white
   ;   ysize = from bottom of subtitle to top of title
   ;   relative ranges > 100% or < 0 lie outside xsize & ysize

if ps eq 1 then begin			; open PS file
 device, encapsulated=0, preview=0
 device, filename = psname
endif

if ps ge 2 then begin			; open EPS file
; device, /encapsulated, /preview	; preview doesn't work with 2 frames
 device, /encapsulated
 device, filename = epsname
endif


; PLOT . . . . . . . . . . . . . . . . . . . . . . . . . .

; if lp eq 'p' then begin		; set relative size of plot window
;  xlim = [.10, .95]
;  ylim = [.12, .95]
; endif else begin
;  xlim = [.08, .97]
;  ylim = [.10, .95]
; endelse
;; !p.position = [ xlim[0], ylim[0], xlim[1], ylim[1] ]	; overrides !p.multi

!p.region = [-0.05,0.0, 0.93,1.0]	; 1st frame size; larger than xsize

xtitle = 'Poloidal Mode Number, m.  Neg m are Left, Pos m are Right-Handed'
if msign lt 0 and xrange[0] eq 0 then xtitle = 'm, Left-Handed Modes'
if msign lt 0 and xrange[0] lt 0 then xtitle = '-m, Neg are Rt, Pos are Lft'
if msign gt 0 and xrange[0] eq 0 then xtitle = 'm, Right-Handed Modes'

if iradvar eq 1 then ytitle = 'Minor Radius (m)'
if iradvar eq 2 then ytitle = 'Normalized Radius, r/a'
if iradvar eq 3 then ytitle = 'Normalized Flux'
if iradvar eq 4 then ytitle = 'SQRT (Normalized Poloidal Flux)'

; Some color tables of interest:
; loadct,  0	;  0 is grayscale, one of 41 installed color tables
; loadct,  1	;  1 is blue-white
; loadct,  3	;  3 is red temperature
; loadct,  4	;  4 is R-G-B-Y
; loadct, 13	; 13 is rainbow
; loadct, 17	; 17 is B-pastel-R
; loadct, 33	; 33 is blue-red 2
; loadct, 34	; 34 is rainbow 2        v,b,c,g,y,o,r
; loadct, 39	; 39 is rainbow + white, 0 is black,v,..r,255 is white
; loadct, 40	; 40 is rainbow + black, 0 is black,v,..r,255 is black

 ncols = 256		; IDL color levels run 0 to 255

; Need FLOATING zmx and zmn, otherwise (zmx-zmn)/(nlevs - 1) can be
; an integer division less than 1, which is zero, and this would
; give no levels.
 if zrange[0] ne zrange[1] then begin	; use specified zrange
  zmx = float(zrange[1])  &  zmn = float(zrange[0])
 endif else begin			; else autorange contours
  zmx = zmax   & zmn = zmin	; these are already floating
 endelse

; Specify COLOR contour levels; array levels.
; Z-data range is nlevs levels, so there are nlevs - 1 intervals.

 nlevs = ncols
 levels = findgen(nlevs)*abs((zmx-zmn)/(nlevs - 1))
 levels = levels + zmn

; Specify contour LINE levels; array llevels.
;  Define top contour at 0.999 of zmx, to get a small contour at top.
 clines = 11
 llevels = findgen(clines)*(zmx-zmn)/(clines - 1)
 llevels = llevels + zmn


; START PLOT:

; First do "no data" grayscale contour plot. This gives black text, 
;  frame, ticks, titles, etc.
 loadct,0			; 0 is 'black', 255 'white'

 contour, zdat,xdat,ydat,levels=levels,/closed,/fill,c_colors=0, $
         /nodata, $
  title=title,subtitle=subtitle,xtitle=xtitle,ytitle=ytitle, $
  xrange=xrange,yrange=yrange, $
  xstyle=1,ystyle=1,xthick=2,ythick=2,xticklen=.01,yticklen=-.01

; Now overplot the color contours (c_colors=clevs) but with no text.

; Save the data in h5 for reading by other programs
 fn='spectral_info.h5'
 fid = H5F_CREATE(fn)
 a = {zdat:zdat, xdat:xdat, ydat:ydat}
 datatypeID = H5T_IDL_CREATE(a)
 dataspaceID = H5S_CREATE_SIMPLE(1)
 datasetID = H5D_CREATE(fid, '1', datatypeID, dataspaceID)
 H5D_WRITE, datasetID, a
 H5F_CLOSE, fid

 contourcolors = 13

 loadct, contourcolors

 clevs = indgen(ncols)
 clevs[0] = 1		; 0=blk, 1=viol; blk-viol jump is unpleasant
 
 contour, zdat,xdat,ydat,levels=levels,/closed,/fill, $
  c_colors=clevs,/overplot

; Overplot a few BW contour lines
 loadct,0			; 0 is black, 255 white
 clc = 255			; white		;; now set by plot3d.in
 thc = 2			; contour line thickness
; c_labls = [0,1,0,1,0,1,0,1,0,1,0]	; 1 to label corresponding contour
; c_labls = indgen(clines)		; to label all contours

 slevs = strarr(clines)			; string array for my labels
 sllevs = strtrim(string(llevels),2)	; remove leading & trailing blanks
 slevtmp = strmid(sllevs, 0, 5)		; first 5 characters of elements
 slevs = slevtmp			; 2nd element is the right one

 contour, zdat,xdat,ydat,levels=llevels,/overplot,fill=0,c_colors=clc, $
     c_labels=c_labls,thick=thc  ; autolabel using levels=llevels & c_labls

; contour, zdat,xdat,ydat,levels=llevels,/overplot,fill=0,c_colors=clc, $
;  c_annotation=slevs,c_labels=c_labls	; formatted contour line labels

; Overplot a curve of q vs radius 
; Cubic spline helps smooth curve near separatrix. xs and ys are
; splined abscissa and ordinate pairs.
 loadct,0			; 0 is black, 255 white
; clq = 255			; white		;; now set by plot3d.in
; clq = 0			; black
; thq = 4			; q curve thickness

 if qsign ne 0 then begin	; plot n*q = m at each corresponding radius
  qn = qlvals*n_mode*qsign	; change sign if requested
  if khand lt 0 then qn = -qn	;
  spline_p, qn, ydat(*,0), xs, ys
  oplot, xs, ys, linestyle=5, thick=thq, color=clq
;  oplot, qn,ydat(*,0),linestyle=5,thick=thq,color=clq ; linestyle=5 long dashes
 endif else begin			; else plot both + and - q curves
  qn = qlvals*n_mode
  spline_p, qn, ydat(*,0), xs, ys
  oplot, xs,ys,linestyle=5,thick=thq,color=clq 
  oplot,-xs,ys,linestyle=5,thick=thq,color=clq 
;  oplot, qn,ydat(*,0),linestyle=5,thick=thq,color=clq 
;  oplot,-qn,ydat(*,0),linestyle=5,thick=thq,color=clq 
 end

; oplot, qn,ydat(*,0),psym=2,thick=3,color=clq	; psym=2 for asterisks

; Overplot any requested grid lines
 loadct,0			; 0 is black, 255 white
; clg = 255			; white		;; now set by plot3d.in
; clg = 0			; black
; thg = 2			; grid line thickness

 if dxgrid gt 0 then begin
  gridx = [0,0]  &  gridy = yrange	; do vertical lines from x=0
  while (gridx[0] lt xrange[1]) do begin
   oplot, gridx,yrange,linestyle=0,thick=thg,color=clg
   gridx = gridx + [dxgrid,dxgrid]
  end
  gridx = [0,0] - [dxgrid,dxgrid]  &  gridy = yrange
  while (gridx[0] gt xrange[0]) do begin
   oplot, gridx,yrange,linestyle=0,thick=thg,color=clg
   gridx = gridx - [dxgrid,dxgrid]
  end
 end

 if dygrid gt 0 then begin		; do horizontal lines from top
  gridy = [yrange[1],yrange[1]] - [dygrid,dygrid]  &  gridx = xrange
  while (gridy[0] gt yrange[0]) do begin
   oplot, xrange,gridy,linestyle=0,thick=thg,color=clg
   gridy = gridy - [dygrid,dygrid]
  end
 end


; Add a COLOR BAR. This colorbar code is from Jay Jayakumar
; Make it in a second frame
 !p.region=[0.86,0.0, 1.0,1.0]		; 2nd frame size = colorbar box

 bar=fltarr(3,nlevs)
 for i=0,2 do bar(i,*)=levels

 loadct, contourcolors	; use same color table as contour plot

 clevs = indgen(ncols)
 clevs[0] = 1		; 1= darkest viol; blk-viol jump is unpleasant

 contour, bar,indgen(3),levels,levels=levels,c_colors=clevs,/fill, $
  xstyle=4,ystyle=1,ythick=2,ticklen=1


; Annotation
 loadct,0				; 0 is black, 255 white

 s1 = strtrim(string(n_mode),2)		; n_mode as a trimmed string
 s2 = strtrim(string(zmax),2)		; zmx ... for label
 if (irpt ge 1 and irpt le 3) then begin
  s2tmp = strmid(s2, 0, size(s2)-3)	; remove last 3 digits from zmx
  s2t = s2tmp[1]			; 2nd element is the right one
 endif

 rhnote = ''
 case irpt of

  1: rhnote = 'Br    n = ' + s1 + '   Max Value = ' + s2t + ' G'
  2: rhnote = 'Bpol  n = ' + s1 + '   Max Value = ' + s2t + ' G'
  3: rhnote = 'Btor  n = ' + s1 + '   Max Value = ' + s2t + ' G'
  4: rhnote = 'Br^2  n = ' + s1 + '   Max Val = ' + s2 + ' G^2'
  5: rhnote = 'Bp^2  n = ' + s1 + '   Max Val = ' + s2 + ' G^2'
  6: rhnote = 'Bt^2  n = ' + s1 + '   Max Val = ' + s2 + ' G^2'
  7: rhnote = '|Btot| n = ' + s1 + '   Max Value = ' + s2t + ' G' 

 endcase

 xyouts,0.98,0.5, rhnote ,orientation=-90,alignment=0.5,/normal,font=0

 xyouts,0.03,-0.02, username+' '+time+' '+gfile ,orientation=0,alignment=0.0,/normal, $
                   font=0,charsize=0.6 
 					; /normal uses normalized coords

; Must close output files
 if abs(ps) gt 0 then device,/close_file

ENDFOR		; end FOR ps = 0,2
; = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

FINISH:

 if abs(ps) gt 0 then device,/close_file
 set_plot,'x'					; default to screen
 ps = 0						;    "    "    "


END	; . . . . . . . . . END OF MAIN PROGRAM . . . . . . . . . . .
