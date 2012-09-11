PRO Figure,name,xsize=xsize,ysize=ysize,ps=ps,eps=eps,new=new, $
  landscape=landscape, $
  close=close,title=title 
;+
; NAME:
;   FIGURE
;
; PURPOSE:
;
; CALLING SEQUENCE:
;   > Figure,name,close=close,xsize=xsize,ysize=ysize,ps=ps,eps=eps,new=new 
;
; INPUT PARAMETERS:
;
; OPTIONAL INPUT PARAMETERS:
;   name   STRING   name of window or file. Default is 'idl'.
;  
; KEYWORDS:
;   close
;   title  STRING   Window name if x-device, title if ps/eps-device.
;   new
;   xsize  FLOAT    figure width in inches. Default is 5
;   ysize  FLOAT    figure heigth in inches. Default is 7
;   ps
;   eps
;
; OUTPUTS:
;
; RESTRICTIONS:
;
; MODIFICATION HISTORY:
;   09-01-02  HR Add landscape and title keywords.
;   05-21-03  HR Use COLOR_SETUP routine rather than settingup my own
;                color table. 
;-
  
;; default values
  IF KEYWORD_SET(landscape) THEN BEGIN
    IF NOT KEYWORD_SET(xsize) THEN xsize=7.0
    IF NOT KEYWORD_SET(ysize) THEN ysize=5.0
  ENDIF ELSE BEGIN
    landscape = 0
    IF NOT KEYWORD_SET(xsize) THEN xsize=5.0
    IF NOT KEYWORD_SET(ysize) THEN ysize=7.0
  ENDELSE

  IF N_PARAMS() EQ 0 THEN name='idl'

  IF KEYWORD_SET(close) THEN BEGIN
    IF (   STRCMP(!D.NAME,'PS',/FOLD_CASE) $
        OR STRCMP(!D.NAME,'EPS',/FOLD_CASE)) THEN BEGIN

      ;; close device
      PRINT,'Closing device.'
      DEVICE,/close_file
      SET_PLOT,'x'
      COLOR_SETUP,/Reverse
    ENDIF
   
  ENDIF ELSE BEGIN

    ;; open window or device
    !P.FONT=0    ; use postscript fonts
    !P.THICK=2.64    ; 0.75pt  (1~0.284pt)
;    !P.THICK=3.52    ; 1pt  
    !X.THICK=4.40    ; 1.25pt 
;    !X.THICK=5.28    ; 1.50pt 
    !Y.THICK=4.40    
;    !Y.THICK=5.28    

    CASE 1 OF

    KEYWORD_SET(ps): BEGIN
      outfile = name+".ps"

      PRINT, 'Open file '+outfile
      SET_PLOT,'ps' 
      COLOR_SETUP

      ; The keywords XSIZE and YSIZE always refere to the width and height
      ; of the figure whereas XOFFSET and YOFFSET refer to a portrait page!
 
      IF KEYWORD_SET(landscape) THEN BEGIN
        xoffset = (8.5-ysize)/2.
        yoffset=(11.+xsize)/2. 
      ENDIF ELSE BEGIN
        xoffset = (8.5-xsize)/2.
        yoffset = (11.-ysize)/2.
      ENDELSE

      DEVICE,FILENAME=outfile,ENCAPSULATED=0,/COLOR,landscape=landscape, $
        /INCHES,XSIZE=xsize,YSIZE=ysize,$
        XOFFSET=xoffset,YOFFSET=yoffset, $
        SET_FONT='Helvetica'
       
      ;IF KEYWORD_SET(title) THEN $
      ;  XYOUTS,0.5,0.9,title,/normal,alignment=0.5,CharSize=1.25

    END

    KEYWORD_SET(eps): BEGIN
      outfile = name+".eps"
      PRINT, 'Open file '+outfile
     
      SET_PLOT,'ps'
      COLOR_SETUP 
      DEVICE,FILENAME=outfile,/ENCAPSULATED,/COLOR,landscape=landscape, $
        /INCHES,XSIZE=xsize,YSIZE=ysize, $
        SET_FONT='Helvetica'

      IF KEYWORD_SET(title) THEN $
        XYOUTS,0.5,1,title,/normal,alignment=0.5,CharSize=1.25

    END
    ELSE: BEGIN
      !P.FONT=-1
      !P.THICK=1
      !X.THICK=1
      !Y.THICK=1
      set_plot,'x' 
      COLOR_SETUP,/Reverse
      dpi = 120. 
      dpi = 100.   ; Reduce dots-per-inch for X11 on Mac.


      IF NOT KEYWORD_SET(title) THEN title=name  
    
      cw = (!D.Window EQ -1) ? 0 : !D.Window

      IF KEYWORD_SET(new) THEN $
        WINDOW,cw+1,title=title, $
          xsize=dpi*xsize,ysize=dpi*ysize, $
          retain=2,free=(cw GE 32) $
      ELSE $
        WINDOW,cw,title=title, $
          xsize=dpi*xsize,ysize=dpi*ysize, $
          retain=2,free=(cw GE 32)
    END

    ENDCASE

;    TVLCT, [0,255,0,0,127,0,127],[0,0,255,0,0,127,0],[0,0,0,255,0,0,0,127] 

  ENDELSE ;; of open a figure

END
