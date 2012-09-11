;
FUNCTION read_logchease,fn



IF N_PARAMS() LT 1 THEN fn = 'log_chease'

  ;; Allocate variables
  line=''
  count=0

  npsi= -1
  cs =  !VALUES.F_NAN
  q  =  !VALUES.F_NAN

  OPENR,unit,fn,ERROR=error,/GET_LUN

  IF (error NE 0) THEN BEGIN
    PRINT,'READ_LOGCHEASE: Could not open '+fn
    RETURN,-1
  ENDIF

  WHILE ~EOF(unit) DO BEGIN
    count = count+1
    ;;PRINT,count,FORMAT='(I4)'
    READF,unit,line

    IF STRMATCH(line,'* NPSI = *') THEN BEGIN
      PRINT,'Found NPSI in line ',count,FORMAT='(A,1x,I4)'
      words = STRSPLIT(line,COUNT=counttok,/EXTRACT)
      inpsi = WHERE(STRCMP(words,'NPSI',/FOLD_CASE),found)
      IF (found EQ 1) AND (inpsi[0] LT counttok-2) THEN $
        npsi = LONG(words[inpsi[0]+2])
    ENDIF

    IF STRMATCH(line,' CS  - MESH*') THEN BEGIN
      PRINT,'Found CS in line ',count,FORMAT='(A,1x,I4)'
      cs = FLTARR(npsi+1)
      READF,unit,cs
    ENDIF

    IF STRMATCH(line,' CSM - MESH*') THEN BEGIN
      PRINT,'Found CSM in line ',count,FORMAT='(A,1x,I4)'
      csm = FLTARR(npsi+1)
      READF,unit,csm
    ENDIF

    IF STRMATCH(line,' PSI(CSM)*') THEN BEGIN
      PRINT,'Found PSI in line ',count,FORMAT='(A,1x,I4)'
      psi = FLTARR(npsi+1)
      READF,unit,psi
    ENDIF

    IF STRMATCH(line,' T (CSM)*') THEN BEGIN
      PRINT,'Found T in line ',count,FORMAT='(A,1x,I4)'
      t = FLTARR(npsi+1)
      READF,unit,t
    ENDIF

    IF STRMATCH(line,' T*DT/DPSI(CSM)*') THEN BEGIN
      PRINT,'Found TT'' in line ',count,FORMAT='(A,1x,I4)'
      ttpr = FLTARR(npsi+1)
      READF,unit,ttpr
    ENDIF

    IF STRMATCH(line,' P (CSM)*') THEN BEGIN
      PRINT,'Found P in line ',count,FORMAT='(A,1x,I4)'
      p = FLTARR(npsi+1)
      READF,unit,p
    ENDIF

    IF STRMATCH(line,' DP/DPSI(CSM)*') THEN BEGIN
      PRINT,'Found P'' in line ',count,FORMAT='(A,1x,I4)'
      ppr = FLTARR(npsi+1)
      READF,unit,ppr
    ENDIF

    IF STRMATCH(line,' Q (CSM)*') THEN BEGIN
      PRINT,'Found Q in line ',count,FORMAT='(A,1x,I4)'
      q = FLTARR(npsi+1)
      READF,unit,q
    ENDIF

    IF STRMATCH(line,' DQ/DPSI(CSM)*') THEN BEGIN
      PRINT,'Found Q'' in line ',count,FORMAT='(A,1x,I4)'
      qpr = FLTARR(npsi+1)
      READF,unit,qpr
    ENDIF

    IF STRMATCH(line,' SHEAR(CS)*') THEN BEGIN
      PRINT,'Found SHEAR in line ',count,FORMAT='(A,1x,I4)'
      s = FLTARR(npsi+1)
      READF,unit,s
    ENDIF

    IF STRMATCH(line,' J-PARALLEL (<j.B>Eq.43)*') THEN BEGIN
      PRINT,'Found JPAR in line ',count,FORMAT='(A,1x,I4)'
      jpar = FLTARR(npsi+1)
      READF,unit,jpar
    ENDIF

    IF STRMATCH(line,' ELLIPTICITY(CSM)*') THEN BEGIN
      PRINT,'Found KAPPA in line ',count,FORMAT='(A,1x,I4)'
      kappa = FLTARR(npsi+1)
      READF,unit,kappa
    ENDIF

    IF STRMATCH(line,' RHO(CS)*') THEN BEGIN
      PRINT,'Found RHO in line ',count,FORMAT='(A,1x,I4)'
      rho = FLTARR(npsi+1)
      READF,unit,rho
    ENDIF

  ENDWHILE
  FREE_LUN,unit
  RETURN, {npsi: npsi, cs:cs, csm: csm, psi: psi, $
           t: t, ttpr: ttpr, p:p, ppr: ppr, q: q, qpr: qpr, s: s, $
           jpar: jpar, kappa: kappa, rho: rho}

END

FUNCTION findroot,xin,yin,count=nz

  z  = !VALUES.F_NAN
  nz = 0

  n = N_ELEMENTS(xin)

  CASE N_PARAMS() OF
  1: BEGIN
      y = xin
      x = INDGEN(n)
    END
  2: BEGIN
      x = xin
      y = yin
    END
  ELSE: MESSAGE,'FINDROOT requires 1 or 2 arguments'
  ENDCASE

  iz = WHERE(y[0:n-2]*y[1:n-1] LT 0,nz)

  IF nz GT 0 THEN BEGIN
    z = x[iz] - (x[iz+1]-x[iz])/(y[iz+1]-y[iz]) * y[iz]
  ENDIF

  iz = WHERE(y EQ 0,nz0)
  IF nz0 GT 0 THEN BEGIN
    z = (nz GT 0) ? [z,x[iz]] : x[iz]
    IF nz GT 0 THEN z = z[SORT(z)]
    nz = nz + nz0 
  ENDIF 

  RETURN,z
END

PRO plot_chease,fn,xtype=xtype,xrange=xrange,yrange=yrange,q=q,packing=packing,ps=ps,sym=sym

  charsize = 1.5
  psym     = KEYWORD_SET(sym) ? -1*ABS(sym) : 0

  chease = READ_LOGCHEASE(fn) 

  IF NOT KEYWORD_SET(xtype) THEN xtype = 'rhop'
  CASE xtype OF
  'rhop': BEGIN
      x=chease.cs
      xtitle='cs=sqrt(psin)'
    END
  'cs': BEGIN
      x=chease.cs
      xtitle='cs=sqrt(psin)'
    END
  'rhov': BEGIN
      x=chease.rho
      xtitle='rhov=sqrt(V/Vtot)'
    END
  'psin': BEGIN
      x=chease.chease.psi/chease.psi[chease.npsi]
      xtitle='psin'
    END
  'q': BEGIN
      x=chease.q
      xtitle='q'
    END
  ELSE: MESSAGE,'Unkown xtype'
  ENDCASE

  xm=0.5*(x[1:chease.npsi]+x[0:chease.npsi-1])
  dq=chease.q[1:chease.npsi]-chease.q[0:chease.npsi-1]

  IF KEYWORD_SET(q) THEN BEGIN
    nq=0
    FOR k= CEIL(MIN(chease.q)),FLOOR(MAX(chease.q)) DO BEGIN
      xqk = FINDROOT(x,chease.q-k,COUNT=nqk)
      IF nq GT 0 THEN BEGIN
        xq = [xq,xqk]
        qq = [qq,REPLICATE(k,nqk)]
      ENDIF ELSE BEGIN
        xq = [xqk]
        qq = [REPLICATE(k,nqk)]
      ENDELSE
      nq = nq + nqk
    ENDFOR
    
  ENDIF

  IF KEYWORD_SET(packing) THEN BEGIN
    FIGURE,XSIZE=6,YSIZE=6,PS=ps
    !P.MULTI=[0,1,1]

    PLOT,xm,dq,XRANGE=xrange,YRANGE=yrange,CHARSIZE=charsize,PSYM=psym, $
      XTITLE=xtitle, $
      YTITLE='dq', $
      TITLE=STRING(fn,chease.npsi,FORMAT='(A," (NPSI=",I3,")")')
    IF KEYWORD_SET(q) THEN $
      FOR k=0,N_ELEMENTS(xq)-1 DO OPLOT,REPLICATE(xq(k),2),!Y.CRANGE,LINESTYLE=2
    FIGURE,/CLOSE
  ENDIF ELSE BEGIN



  FIGURE,XSIZE=6,YSIZE=9,PS=ps
  !P.MULTI=[0,2,3]
  PLOT,x,chease.p,XRANGE=xrange,YTITLE='p',CHARSIZE=charsize,PSYM=psym
  IF KEYWORD_SET(q) THEN FOR k=0,N_ELEMENTS(xq)-1 DO OPLOT,REPLICATE(xq(k),2),!Y.CRANGE,LINESTYLE=2

  !P.MULTI=[4,2,3]
  PLOT,x,chease.jpar,XRANGE=xrange,YTITLE='jpar',CHARSIZE=charsize,PSYM=psym
  IF KEYWORD_SET(q) THEN FOR k=0,N_ELEMENTS(xq)-1 DO OPLOT,REPLICATE(xq(k),2),!Y.CRANGE,LINESTYLE=2
  
  !P.MULTI=[2,2,3]
  PLOT,x,chease.q,XRANGE=xrange,YTITLE='q',XTITLE=xtitle,CHARSIZE=charsize,PSYM=psym
  IF KEYWORD_SET(q) THEN FOR k=0,N_ELEMENTS(xq)-1 DO OPLOT,REPLICATE(xq(k),2),!Y.CRANGE,LINESTYLE=2
  
  !P.MULTI=[3,2,3]
  PLOT,x,chease.ttpr,XRANGE=xrange,YTITLE='FF''',CHARSIZE=charsize,PSYM=psym
  IF KEYWORD_SET(q) THEN FOR k=0,N_ELEMENTS(xq)-1 DO OPLOT,REPLICATE(xq(k),2),!Y.CRANGE,LINESTYLE=2

  !P.MULTI=[5,2,3]
  PLOT,x,chease.ppr,XRANGE=xrange,YTITLE='p''',CHARSIZE=charsize,PSYM=psym
  IF KEYWORD_SET(q) THEN FOR k=0,N_ELEMENTS(xq)-1 DO OPLOT,REPLICATE(xq(k),2),!Y.CRANGE,LINESTYLE=2
  
  !P.MULTI=[1,2,3]
  PLOT,xm,dq,XRANGE=xrange,YTITLE='dq',XTITLE=xtitle,CHARSIZE=charsize,PSYM=psym
  IF KEYWORD_SET(q) THEN FOR k=0,N_ELEMENTS(xq)-1 DO OPLOT,REPLICATE(xq(k),2),!Y.CRANGE,LINESTYLE=2
  
  FIGURE,/CLOSE
  END
END
