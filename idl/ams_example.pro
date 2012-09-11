;+
; More detailed example of active MHD spectroscopy
;
; 20101103 JMH Created.
; 20120403 JMH Copy from ~/idl/plots/aps/2010, add arb n-number fit
; 20120504 JMH Fork from v2 some specifics for MPI66M probes
; 20120627 JMH Copy for n=2 AMS shots, add SRA, subs options
;
; NOTES:
;
;-

; SUB-ROUTINES -----------------------------------


; INITS ------------------------------------------

	;; Control
	psplot=0		; Plot to postscript/pdf
	replot=0		; Display pre-calculated results
	debug=0
	verbose=1		; Verbose printed output

	dt=0.5

; 	shot=147170
; 	t0=2000.
; 	t1=4000.
; 	fExt=2.5	; Hz
;	fftClip=[2200.,3800.]

 	shot=146382
 	t0=3000
 	t1=3600.
 	fExt=-10.	; Hz
	fftClip = [3200,3600]
;	fftClip=[2200.,3800.]
;	fftClip=[2700.,3100.]

	nTor = [2,4,0]
	nTor = [2,1,3,0]


;	xmark = [2700,2900]
	xmark = [fftClip,MEAN(fftClip)]

	;; Option to use sinusoidal response analysis instead of FFT
	useSra = 1

    ;; Make copies of probes 180 deg away, phase-shifted by 180 deg.
    ;; Only appropriate for difference probes
	mirrorProbesInPlot=0

	;; Show n-number fit in final plot
	showFit=1
	standingFit=0

	subs=1

	;; Mostly graphics-related setup
	@jmh_inits.inc

	coilGroup='ICOILALL'
	coilVersion='nominal'
	sensorGroup='mpi66m_good_b_2011'
	sensorGroup='MISL'
	sensorVersion='nominal'



	;; Fn and single keywords for RWM_COMPENSATION2. 
	fn='/u/hansonjm/var/data/transfers/tf2012_single.h5'
	single=1


; GET DATA ---------------------------------------

	IF ~replot THEN BEGIN

		sensorPoints = GROUP_POINTS(sensorGroup,sensorVersion, $
			phi=sensorPhi, $
			w=sensorW, $
			nominal=sensorNames)
		ns=N_ELEMENTS(sensorPhi)
                print, sensorPoints
                print, "============ got sensor points ================"

		coilPoints = GROUP_POINTS(coilGroup,coilVersion, $
			phi=coilPhi, $
			nominal=coilNames)
		nc=N_ELEMENTS(coilPhi)
                print, coilPoints
                print, "============ got coil points ================"

		;; Coil data
		coilData=AGADAT(coilPoints,shot,dt,t0,t1, $
			time=time, $
			verbose=verbose, $
			subs=subs, $
			indices=coilInds)
                print, "============ got coil data ================"

		;; Sensor data
		sensorData=AGADAT(sensorPoints,shot,dt,t0,t1, $
			time=time, $
			verbose=verbose, $
			subs=subs, $
			indices=sensorInds)
                print, "============ got sensor data ================"

                filename = 'sensorDataMISL.csv'
                s = Size(sensorData, /Dimensions)
                xsize = s[0]
                lineWidth = 1600000
                comma = ","
   
                ; Open the data file for writing.
                OpenW, lun, filename, /Get_Lun, Width=lineWidth

                ; Write the data to the file.
                sData = StrTrim(sensorData,2)
                sData[0:xsize-2, *] = sData[0:xsize-2, *] + comma
                PrintF, lun, sData

                ; Close the file.
                Free_Lun, lun
; CALCULATIONS -----------------------------------

		nt=N_ELEMENTS(time)

		;; Zero sensors at t0
		zi=WIN2IND(time,[fftClip[0]-5.,fftClip[0]+5.])
		sensorZero=REBIN($
			TRANSPOSE(AMEAN(sensorData[zi[0]:zi[1],*],dimension=1)),nt,ns)
		sensorDataZero=sensorData-sensorZero
                print, "============ removed sensor mean ================"
		
        ;; Clip to analysis window. All FFT analysis done in this
        ;; window.  Compensation is done over the larger window
		ci=WIN2IND(time,fftClip)
		timec=time[ci[0]:ci[1]]
		ntc=N_ELEMENTS(timec)

		;; Subtract direct vacuum coupling from sensors
		sensorDataComp=TD_COMPENSATE(sensorData,coilData,dt,$
			sensorNames, coilNames, $
			fn=fn, $
			single=single)
                print, "============ subtracted sensor coupling ================"

                filename = 'sensorDataCompMISL.csv'
                s = Size(sensorDataComp, /Dimensions)
                xsize = s[0]
                lineWidth = 1600000
                comma = ","
   
                ; Open the data file for writing.
                OpenW, lun, filename, /Get_Lun, Width=lineWidth

                ; Write the data to the file.
                sData = StrTrim(sensorDataComp,2)
                sData[0:xsize-2, *] = sData[0:xsize-2, *] + comma
                PrintF, lun, sData

                ; Close the file.
                Free_Lun, lun

        trend=FLTARR(ntc,ns)
		IF useSra THEN BEGIN
                        print, "============ inside sra ================"

			sensorDataFext = FLTARR(ntc,ns)
			sensorCompFext = FLTARR(ntc,ns)
			sensorDataFftFi = COMPLEXARR(ns)			
			sensorCompFftFi = COMPLEXARR(ns)			
			sensorAmpErr = FLTARR(ns)

			FOR i=0, ns-1 DO BEGIN

                ;; Sinusoidal response analysis option.  Note that we
                ;; invert the sign of the frequency to be compatible
                ;; with the FFT frequency sign convention.

				IF i EQ 0 THEN BEGIN
                                    print, "============ first loop ================"

					sensorDataFftFi[i] = SRAFIT(timec, $
						sensorDataZero[ci[0]:ci[1],i], -fExt/1000., $
						get_mats=sraMats, $
						offset=offset, $
						slope=slope, $
						yfit=yfit, $
						residual=residual)/2.
                ENDIF ELSE BEGIN
                                    print, "============ second loop ================"
					sensorDataFftFi[i] = SRAFIT(timec, $
						sensorDataZero[ci[0]:ci[1],i], $
						set_mats=sraMats, $
						offset=offset, $
						slope=slope, $
						yfit=yfit, $
						residual=residual)/2.
				ENDELSE

                                print, "============  outside if ================"
				trend[0,i] = (timec - timec[0])*slope + offset
				sensorDataFExt[0,i] = yfit - trend[*,i]

				sensorCompFftFi[i] = SRAFIT(timec, $
					sensorDataComp[ci[0]:ci[1],i], $
					set_mats=sraMats, $
					offset=offset, $
					slope=slope, $
					yfit=yfit, $
					residual=residual)/2.

				sensorAmpErr[i] = $
					SQRT(AMEAN(residual^2.) - AMEAN(residual)^2.)/2.

				sensorCompFExt[0,i] = yfit - (timec - timec[0])*slope - offset
			ENDFOR
                        print, "============ finished for loop ================"
                        print, "============ finished sra analysis?  ================"
    	ENDIF ELSE BEGIN
            print, "============ linear trend?  ================"
            ;; Linear trend inside analysis window
            FOR i=0,ns-1 DO BEGIN
                tmp=LINFIT(timec,sensorDataZero[ci[0]:ci[1],i],yfit=yfit)
                trend[0,i]=yfit
            ENDFOR

            ;; Extract resonant pickup in sensors using FFT
            FFT_INIT, timec, fv_full=fvalues, /negf, nyind=nyind

            tmp=MIN(ABS(fvalues-fExt/1000.),fi)
            tmp=MIN(ABS(fvalues+fExt/1000.),fiNeg)
            mask=COMPLEXARR(ntc,ns)
            mask[fi,*]=COMPLEX(1.,0.)
            mask[fiNeg,*]=COMPLEX(1.,0.)

            sensorDataFft=FFT(sensorDataZero[ci[0]:ci[1],*],dimension=1)
            sensorDataFExt=FFT(sensorDataFft*mask,1,dimension=1)

			;; FFT of compensated sensors
			sensorCompFft=FFT(sensorDataComp[ci[0]:ci[1],*],dimension=1)
			sensorCompFExt=FFT(sensorCompFft*mask,1,dimension=1)

			sensorCompFftFi = sensorCompFft[fi,*]
			sensorDataFftFi = sensorDataFft[fi,*]

	        ;; Get amplitude error bars from adjacent FFT bins.  Note we
    	    ;; have factors of 2 and 0.5 that cancel here.
; 			sensorAmpErr=ABS(sensorDataFft[fi+1,*])+$
; 				ABS(sensorDataFft[fi-1,*])
			sensorAmpErr=ABS(sensorCompFft[fi+1,*])+$
				ABS(sensorCompFft[fi-1,*])
		ENDELSE
            print, "============ compensated trend?  ================"

		;; Compensated amplitude and phase at fExt
		sensorCompAmp=ABS(sensorCompFftFi)*2.
		sensorDataAmp=ABS(sensorDataFftFi)*2.

		sensorCompPh=ATAN(sensorCompFftFi,/phase)

        ;; Get phase error bars from amplitude error bars using the
        ;; geometric argument.
		sensorPhErr=2.*ASIN(sensorAmpErr/sensorCompAmp/2. < 1.)

		;; Toroidal mode number fit
		IF standingFit THEN BEGIN
			nfit=NFITMAT(nTor,sensorPhi,sensorW,$
					conda=ca, $
					row_n=row_n) $
				## ABS(sensorCompFftFi)*2.
        ENDIF ELSE BEGIN
            print, "============ startin nfitmat - svd analysis?  ================"

			nfit=NFITMAT(nTor,sensorPhi,sensorW, $
					conda=ca, $
					row_n=row_n) ## sensorCompFftFi*2.
		ENDELSE

;		ASSERT, ca LE 10., 'Condition number too large.'

		nn = N_ELEMENTS(ntor)
		cFit = FLTARR(nn)
		sFit = FLTARR(nn)
		FOR i=0, nn-1 DO BEGIN
			iw = WHERE(nTor[i] EQ row_n, c)
			CASE c OF
				1: cFit[i] = nfit[iw]
				2: BEGIN
					cFit[i] = nfit[iw[0]]
					sFit[i] = nfit[iw[1]]
				END
				ELSE: MESSAGE, 'Number of rows for this n should be 1 or 2'
            ENDCASE
			IF nTor[i] EQ 0 THEN fitZero = cFit[i]
        ENDFOR

		IF standingFit THEN BEGIN
			absFit = SQRT(cFit^2. + sFit^2.)
			absFitTot = TOTAL(absFit)
        ENDIF ELSE BEGIN
			compFit = 0.5*COMPLEX(cFit,sFit)
			absFit = ABS(compFit)
			phFit = ATAN(compFit,/phase)
		ENDELSE



		nPhi=100.
		phi=FINDGEN(nPhi)*360./(nPhi-1.)

        ;; Lowpass filter the coil data with 50 Hz corner frequency
        ;; for plotting
		coilDataLp=coilData
		omg=2.*!PI*0.05
		FOR i=0,2 DO coilDataLp[0,i]=LPFILTER(coilData[*,i],dt,omg)

    ENDIF	; ~replot

; PRINTED OUTPUT ---------------------------------
	
	IF standingFit THEN BEGIN
		PINFO, ['Fit to n = '+STRJOIN(FMTSTR(nTor),', '), $
			'Condition number = '+FMTSTR(ca), $
			'n = '+FMTSTR(nTor)+' amp = '+FMTSTR(absFit*1.E4)+' G'], $
			quiet=quiet
    ENDIF ELSE BEGIN
		PINFO, ['Fit to n = '+STRJOIN(FMTSTR(nTor),', '), $
			'Condition number = '+FMTSTR(ca), $
			'n = '+FMTSTR(nTor)+' amp = '+FMTSTR(absFit*1.E4)+' G' $
				+', ph = '+FMTSTR(phFit*!RADEG)+' deg'], $
			quiet=quiet
	ENDELSE




; PLOTS ------------------------------------------

	CLEAR_WINDOWS

	;; Plot of 3 coil currents overlaid
	JMHPLOT, time, coilDataLP[*,0:2]*1.E-3, $
		plots_per_pg=4, $
		overplot=[0,1,1], $
		titles=coilPoints[0:2]+' '+shot_str, $
		/direct, $
;		yrange=[-0.45,0.7], $
		xtitle='time (ms)', $
		xmark=xmark


	;; Plot of sensors with FFT analysis overlaid
; 	MK_WINDOW, xsize=756, ysize=576
; 	AUTO_PMULTI, ns, /no_set, pmulti=pm
; 	dims=pm[1:2]
; 	positions=MPOSITIONS(dims,charsize=charsize)
; 	FOR i=0,ns-1 DO BEGIN
; 		PLOT, time, sensorDataZero[*,i]*1.E4, $
; 			yrange=[-10,10], $
; 			xrange=PLOT_RANGE([t0,t1],0.05), $
; 			/nodata, $
; 			xtitle='time (ms)', $
; 			xstyle=1, $
; 			ystyle=1, $
; 			charsize=charsize, $
; 			position=positions[*,i], $
; 			noerase=i GT 0
; 		ZEROLINE, color=colors.grey
; 		OPLOT, time, sensorDataZero[*,i]*1.E4, color=colors.blue
; 		OPLOT, timec, (sensorDataFExt[*,i]+trend[*,i])*1.E4, color=colors.red
; 		ADDLINE, xmark, color=colors.grey
; 		LEGEND, sensorPoints[i]+' '+shot_str, box=0, textcolor=colors.blue
; 	ENDFOR
	fExtStr = FMTSTR(fExt)+' Hz'
	xPtrs=[REPLICATE(PTR_NEW(time),ns),REPLICATE(PTR_NEW(timec),ns)]
	yPtrs=PTRARR(ns*2)
	FOR i=0,ns-1 DO yPtrs[i]=PTR_NEW(sensorDataZero[*,i]*1.E4)
	FOR i=0,ns-1 DO yPtrs[i+ns]=PTR_NEW((sensorDataFExt[*,i]+trend[*,i])*1.E4)
	JMHPLOT, xPtrs, yPtrs, $
		title=[sensorPoints+' (G) - '+shot_str,$
			REPLICATE(fExtStr+' component',ns)],$
		interleave=2, $
		xtitle='time (ms)', $
		xmark=xmark
	PTR_FREE, xPtrs, yPtrs


	;; Plot of sensor FFTs
	IF debug THEN BEGIN
		JMHPLOT, fvalues[0:nyind]*1000., ABS([[sensorDataFft[0:nyind,*]], $
			[sensorCompFft[0:nyind,*]]])*2.E4, $
			titles=[sensorNames+' FFT amplitude (G)', $
				REPLICATE('Compensated for vacuum coil pickup (G)',ns)], $
			xtitle='frequency (Hz)', $
			/xlog, $
			interleave=2, $
			xrange=[fExt/5.,fExt*1000.], $
			yrange=PLOT_RANGE(ABS(sensorDataFft[0:nyind,*]),[0.,0.1],$
				/force_zero)*2.E4
	ENDIF


	;; Plot of compensated and uncompensated sensors
; 	MK_WINDOW, xsize=756, ysize=576
; 	positions=MPOSITIONS(dims, charsize=charsize, opad=[0.45,0.1])
; 	FOR i=0,ns-1 DO BEGIN
; 		PLOT, timec, sensorDataFExt[*,i]*1.E4, $
; 			yrange=[-3,5], $
; 			xrange=PLOT_RANGE(timec,0.05), $
; 			/nodata, $
; 			xtitle='time (ms)', $
; 			xstyle=1, $
; 			ystyle=1, $
; 			charsize=charsize, $
; 			position=positions[*,i], $
; 			noerase=i GT 0
; 		ZEROLINE, color=colors.grey
; 		OPLOT, timec, sensorDataFExt[*,i]*1.E4, color=colors.red
; 		OPLOT, timec, sensorCompFExt[*,i]*1.E4, color=colors.cyan
; 		ADDLINE, xmark, color=colors.grey
; 		LEGEND, sensorPoints[i]+' '+shot_str, box=0, textcolor=colors.red
;     ENDFOR

	JMHPLOT, timec, [[sensorDataFExt],[sensorCompFExt]]*1.E4, $
		title=[sensorNames+', '+fExtStr+' component (G)',$
			REPLICATE('Compensated for vacuum coil pickup',ns)], $
		xtitle='time (ms)', $
		interleav=2, $
		xmark=xmark



	;; Error bar plots of amplitude and phase of each sensor
	MK_WINDOW, xsize=756, ysize=576
	positions=MPOSITIONS([1,2], charsize=charsize, opad=[0.45,0.1])
	phiSensor1=sensorPhi
	phiSensor2=WRAP(sensorPhi*!DTOR+!PI)*!RADEG

	PLOT, phi, phi, $
		xrange=[0.,360.], $
		yrange=PLOT_RANGE(sensorCompAmp,[0.,0.3],/force_zero)*1.E4, $
		xstyle=1, $
		ystyle=1, $
		xtitle='phi (deg)', $
		/nodata, $
		position=positions[*,0]
	IF showFit THEN BEGIN

		clrs = NCLRS(nn+1)
		fitTot = FLTARR(nPhi)

		IF standingFit THEN BEGIN
			FOR i=0, nn-1 DO BEGIN
				fiti = cFit[i]*COS(nTor[i]*phi*!DTOR) $
					+ sFit[i]*SIN(nTor[i]*phi*!DTOR)
				IF nTor[i] EQ 0 THEN add = 0 ELSE add = fitZero
				OPLOT, phi, (fiti + add)*1.E4, color=clrs[i]
				fitTot += fiti
			ENDFOR

        ENDIF ELSE BEGIN
			FOR i=0, nn-1 DO BEGIN
				fiti = REPLICATE(absFit[i],nPhi)
;				IF nTor[i] EQ 0 THEN add = 0 ELSE add = fitZero
				add = 0
				OPLOT, phi, (fiti + add)*1.E4, color=clrs[i]
				fitTot += fiti
			ENDFOR
		ENDELSE

		OPLOT, phi, fitTot*1.E4, color=clrs[i]
		LEGEND, [FMTSTR(shot)+', '+FMTSTR(LONG(fftClip[0]))+'-'$
					+FMTSTR(LONG(fftClip[1]))+' ms:', $
				'Fit to n = '+FMTSTR(nTor)+': '$
					+FMTSTR(absFit*1.E4,3)+' G',$
				'Total fit for all ns'], $
			textcolor=[!p.COLOR,clrs], $
			/bottom, $
			box=0, $
			charsize=1.
    ENDIF 
	EBPLOT, phiSensor1, sensorCompAmp*1.E4, sensorAmpErr*1.E4, /hole
	OPLOT, phiSensor1, sensorCompAmp*1.E4, psym=SYMCAT(14), symsize=!P.SYMSIZE
	IF mirrorProbesInPlot THEN BEGIN
		EBPLOT, phiSensor2, sensorCompAmp*1.E4, sensorAmpErr*1.E4, /hole
		OPLOT, phiSensor2, sensorCompAmp*1.E4, psym=4, symsize=!P.SYMSIZE
	ENDIF

	LEGEND, sensorGroup+' amplitude (G) '+shot_str, box=0


	PLOT, phi, phi, $
		xrange=[0.,360.], $
		yrange=[-190.,200.], $
		xstyle=1, $
		ystyle=1, $
		xtitle='phi (deg)', $
		/nodata, $
		position=positions[*,1], $
		/noerase
	ADDLINE, [-90.,0.,90.], /horiz, color=colors.grey
	IF showFit THEN BEGIN
		IF standingFit THEN BEGIN
			clrs = NCLRS(nn)			
        ENDIF ELSE BEGIN
			OPLOT, phi, $
				WRAP(phFit[0]-nTor[0]*phi*!DTOR)*!RADEG, $
				color=clrs[0]
		ENDELSE
	ENDIF
	EBPLOT, phiSensor1, sensorCompPh*!RADEG, sensorPhErr*!RADEG, /hole
	OPLOT, phiSensor1, sensorCompPh*!RADEG, psym=SYMCAT(14), symsize=!P.SYMSIZE
	IF mirrorProbesInPlot THEN BEGIN
		OPLOT, phiSensor2, WRAP(sensorCompPh+!PI)*!RADEG, $
			psym=4, symsize=!P.SYMSIZE
		EBPLOT, phiSensor2, WRAP(sensorCompPh+!PI)*!RADEG, $
			sensorPhErr*!RADEG, /hole
	ENDIF
	LEGEND, sensorGroup+' phase (deg) '+shot_str, box=0


; CLEAN UP ---------------------------------------

	PS_CLEANUP, $
		/pdf, $					; Convert to pdf file
		folder=plotsDir, $		; Put result in this folder
		/log, $					; Append to log in plotsDir
		/burst, $				; Burst if more than 1 page
		/crop, $				; Crop white-space
		quiet=quiet
END
