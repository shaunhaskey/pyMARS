;+
; FILTER_EXAMPLE
;	Example calculation of filter coefficients for discrete-time
;	transfer function representation
;
; 20120605 JMH Created.
;
; NOTES:
;	Requires subroutines in /u/hansonjm/idl/lib
;-

; ------------------------------------------------------------
; Subroutines

; ------------------------------------------------------------
; Inits

	;; Control
 	psplot=1				; Plot to postscript/pdf
 	print=0					; Send to printer (forces psplot, files saved)
 	destination='phaser3'	; Printer name to send to 
 	verbose=1				; Verbose printed output

	sensor='UISL4'
	coil='IU270'
	dt = 1000.E-6	; (sec)
	nt = 1000	
	h5file = '/u/hansonjm/var/data/transfers/tf2012_single.h5'
	nf = 100
	f0 = 0.1
	f1 = 1.E4
	flog = 1

	;; Mostly graphics-related setup
	@jmh_inits.inc


; ------------------------------------------------------------
; Calculations


	;; Create step function coil current waveform (A)
	time = FINDGEN(nt)*dt
	current = FLTARR(nt)
	current[nt/2:*] = 500.

	;; Compute coupling in time-domain
	bs = RWM_COMPENSATION2(sensor,coil,current,dt, $
			fn=h5file, $
			sz=sz, $
			sp=sp, $
			sk=sk, $
			az=az, $
			bz=bz, $
			quiet=quiet)

	;; Also compute transfer function
	freqs = LINE(nf,f0,f1,log=flog)
	xfer = RWM_COMPENSATION2(sensor,coil,freqs, $
				fn=h5file, $
				quiet=quiet)

	


; ------------------------------------------------------------
; Printed output

	;; Transfer function parameters
	PINFO, ['Transfer function parameters for '+sensor+'/'+coil+' coupling.',$
		PAR_LIST({sp:sp,sz:sz,sk:sk,dt:dt,az:az,bz:bz,h5file:h5file})]



; ------------------------------------------------------------
; Plots

	CLEAR_WINDOWS

	;; Coil current and coupling
	JMHPLOT, time*1.e3, [[current*1.E-3],[bs*1.E4]], $
		title=['Example '+coil+' current (kA)','Coupling to '+sensor+' (G)'], $
		xtitle='time (ms)', $
		/no_same_yr

	;; Transfer function
	JMHPLOT, freqs, [[ABS(xfer)*1.E7],[ATAN(xfer,/phase)*!RADEG]], $
		title=sensor+'/'+coil+[' amplitude (G/kA)','phase (deg)'], $
		xtitle='frequency (Hz)', $
		/no_same_yr, $
		ymark=[-90,90], $
		xlog=flog



; ------------------------------------------------------------
; Clean up

 	PS_CLEANUP, $
 		/pdf, $					; Convert to pdf file
 		folder='/u/haskeysr/code/idl/', $		; Put result in this folder
 		/log, $					; Append to log in plotsDir
 		/burst, $				; Burst if more than 1 page
 		/crop, $				; Crop white-space
 		quiet=quiet, $
 		print=print, $
 		destination=destination
END
