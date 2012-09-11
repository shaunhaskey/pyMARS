#+
# NAME: sspqi: stab_setup_psave_qsave_invq.bas
#
# PURPOSE: given geqdsk and aeqdsk, prepare EXPEQ, teq, and inverse equilibrium
#		geqdsk and aeqdsk files for scaled pressure AND PLASMA CURRENT equilibria
#		Record the relevant parameters for these equilibria including:
#		betan, li, li3, q0, qmin, q95, qmax, R0EXP, B0EXP, psihigh or thetac
#
# CALLING SEQUENCE: read "stab_setup.bas"
#   
#
# INPUT PARAMETERS: User must declare case dependent variables in the program.
#   			Should/Could change to a function later.
# 
# OPTIONAL INPUT PARAMETERS:
#
# KEYWORDS:
#   status
#   debug
#   ps
#
# OUTPUTS:
#   
#
# RESTRICTIONS:
#   
# Internal Routines:
#
# External Routines:
#         
# MODIFICATION HISTORY:
#    20100602	Include R0EXP, B0EXP normalization using RMJ and BCENTR
#    20100914	Include options to scale qsave for target q95, keep q0>1, and avoid negative edge current
#    20110901   Write EXPEQ for each pressure and current profile
#-

  #read in necessary functions
  remark "Loading functions"

  read "d3.bas"
  read "sewall.bas"
  read "wexpeq.bas"

  chameleon shotid = "135817" 
  chameleon shtime = "3105"

  #Declare d3.bas inputs
  real fedge = 0.95
  real mymi = 1
  integer npsi = 270
  real mythetac = 0.003
  eq.map=256
  
  # Other options
  integer myfuzzy = 0  #free some boundary points
  integer mysmjpar = 0 #keep edge current positive when qmin<0
  real jedge = 0.985    # psi value marking beginning of edge current region
  integer calldcon = 1 #call dcon
  
  #Set pressure multiplier and safety factor profile
  # psave=pmult*psave0
  real pmin = 0.34
  real pstep = -0.0030
  real npmult = 51
  
  # qsave = ((1/qsave0+qsave0)/qd+qminp)*(1+qmult*psibar**qexp1) 
  integer nqmult = npmult
  real qmin = 0.47
  real qstep = 0.0020
  real qd = 2.0
  real qminp = 0
  real qexp1 = 1.8
 
  # Flag for constrait: qmin > 1: default was qminthresh=1.1, qminexp=1.8
  integer q0gt1 = 1
  real qminthresh = 1.1
  real qmincush = 0.0
  real qminexp = 1.8  

  remark "npsi: "
  remark format(npsi,7)
  remark "map:"
  remark format(map,7)
  remark "Thetac value:"
  remark format(mythetac,9,4,1)
  remark "q(0) greater than 1 (0/1)"
  remark format(q0gt1,1)

  # Declare other variables
  real pmult =   pstep*(iota(npmult)-1) + pmin
  real qmult =   qstep*(iota(nqmult)-1) + qmin 
  chameleon gname = "eqdsk" 
  chameleon fileid = "shot_time"
  chameleon inv_save_file = "shot_time_inv.sav"
  real liga, li3, betaN, wtotn1, wtotn2, wtotn3, dpwmin, wwtotn1
 
  # Read g-file in current directory (there should only be one!)
  gfiles

  # Setup .dat result file readable by READ_MATRIX.pro
  remark "Opening results file"
  integer iofile = basopen("stab_setup_results.dat","w")
  iofile << "; " // gfilelist(1) << " thetac, npsi, map =, qd=, qminp=, qexp1= , qminexp="
  iofile << "; " // format(mythetac,9,4,1) // format(npsi,7)  // format(map,7) // format(qd,9,4,1) //  format(qmin,9,4,1) //  format(qexp1,9,4,1) // format(qminexp,9,4,1)

  # Read in equilibrium
  d3(gfilelist(1),0,fedge,mymi,npsi,mythetac)
  shotid = substr(gfilelist(1),2,6)
  shtime = substr(gfilelist(1),10,4)
  inv_save_file = trim(shotid) // "_" // trim(shtime) // "_inv.sav"
  chameleon teqin   = "t" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq"
  chameleon ieqin   = "i" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq"
  chameleon geqin   = "g" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq"
  chameleon aeqin   = "a" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq"
  chameleon meqin   = "m" // trim(shotid) // ".0" // trim(shtime) 
   
  # Restore the inverse equilibrium
  remark "Restoring sav file"
  restore ^inv_save_file
  call teq_inv(0,-1)
  
  chameleon itstr = "Plasma current target: " // format(eq.plcm,7,3,1)
  remark itstr
  chameleon itstr = "Plasma current actual: " // format(eq.placur,18,8,2)
  remark itstr

  # Call sewall
  sewall

  # Print output file headers

  iofile <<  "   PMULT" // "  QMULT" // "   BETAN" // \
	     "      Q0" // "     QMIN" // "    Q95" // "    QMAX" // "   IP" // \
             "         LI" // "     LI3" // "     WTOTN1" // "  WTOTN2" // "  WTOTN3" // "  WWTOTN1" // \
	     "  DPWMIN" // "        RESIDJ" // "        R0EXP" // "             B0EXP" 

  # Prepare files for each pmulture value
  integer iii=1
  #integer jjj=1
  #do jjj=1,nqmult
  do iii=1,npmult
  	
	# Report iteration
	chameleon itstr = "Iteration: " // format(iii,3) // " out of " // format(npmult,3)
	remark itstr
	chameleon itstr = "Pressure value: " // format(pmult(iii),7,3,1)
	remark itstr
	chameleon itstr = "Current value: " // format(qmult(iii),7,3,1)
	remark itstr

	# Reset wall
	ishape = 6
	dcn.a = 20.0
	
  	# Restore the inverse equilibrium,  
	restore ^inv_save_file
	call teq_inv(0,-1)
	chameleon psave0=psave
        chameleon qsave0=qsave
	chameleon jparsave0=jparsave
	
	# Modify fuzzy boundary points
	if (myfuzzy==1) then
	  alfbd=0.5
	  chameleon lfuz = min(where(zfbd < -100,iota(nfbd),nfbd+1))	#use where to get minimum index for zfbd<-100
	  chameleon hfuz = max(where(zfbd < -100,iota(nfbd),0))		#use where to get maximum index for zfbd<-100
	  alfbd(lfuz:hfuz)= 0
	endif
	
	# Scale pressure and parallel current density with Ip free
	psave = pmult(iii)*psave0
	qsave = ((1/qsave0+qsave0)/qd+qminp)*(1+qmult(iii)*psibar**qexp1) 
	remark "Scaling psave and qsave"
	call teq_inv(0,0)

	chameleon itstr = "Plasma current: " // format(eq.plcm,7,3,1)
	remark itstr
	chameleon itstr = "Plasma current actual: " // format(eq.placur,18,8,2)
  	remark itstr	
	chameleon itstr = "Internal inductance: " // format(eq.li(1),10,3,1)
  	remark itstr
	
	# Account for qmin<1 (DCON gives fixed boundary instability otherwise)
	if (q0gt1 == 1) then
	  chameleon delqmin = qminthresh - min(qsave) 
	  if (delqmin > 0) then	  
	  	real foo = (delqmin+qmincush)*(1-psibar)**qminexp
  	  	real qsave0=qsave
  	  	qsave=qsave0+foo
		call teq_inv(0,0)
	  endif
	endif
	
	# Smooth edge jparsave
	if (mysmjpar==1) then
	  remark "Removing negative edge current"
	  # get lower index in psibar
	  chameleon lpsi = min(where(psibar > jedge,iota(msrf),msrf+1))
	  chameleon jps0m = jparsave0
	  jps0m(1:lpsi)=0.0	  
	  # find negative values
	  chameleon ljpar = min(where(jps0m > jparsave,iota(msrf),msrf+1))
	  jparsave(ljpar:msrf)=jparsave0(ljpar:msrf)
	  # solve holding psave, jparsave and letting flux and current float
	  teq_inv(3,0)
	  jparsave=jparsave-jparsave(msrf)
	  teq_inv(3,0)
	endif

	# Get free boundary stability
	wtotn1=0
	wtotn2=0
	wtotn3=0
	wwtotn1=0	
	if (calldcon==1) then
	  dcn.nn=1
    	  remark "Calling DCON (n=1)" 
    	  call dcon
    	  wtotn1   = et(1)
	  
	  dcn.nn=2
    	  remark "Calling DCON (n=2)" 
    	  call dcon
    	  wtotn2   = et(1)

	  dcn.nn=3
    	  remark "Calling DCON (n=3)" 
    	  call dcon
    	  wtotn3   = et(1)
	endif

	# Set wall
    	dpwmin=sewall()

	if (calldcon==1) then
	  # Get ideal wall stability n=1
	  dcn.nn=1
    	  remark "Calling DCON" 
    	  call dcon
    	  wwtotn1   = et(1)
	endif

	# Write stability files
	wexpeq
	weqdsk("t")
	weqdsk("ag")
	weqdsk("i")
	
	# Get R0EXP, B0EXP for CHEASE normalization: Code taken from wexpeq.bas
  	real rbnd   = 0.01*eq.rls  # R (m) boundary
  	real zrmax  = max(rbnd)
  	real zrmin  = min(rbnd)
  	real r0exp  = 0.5 * (zrmax + zrmin)      # RMJ (m) major radius  
  	real eqro   = eq.ro/1e2                  # (m) nominal major radius
  	real eqbtor = eq.btor/1e4              # (T) field on nominal axis if f'=0 
  	real b0exp  = abs(eqbtor)*eqro/r0exp  	# (T) toroidal field at RMJ 
	
	# Get q95
	real polyvals=fit(psibar,qsave,20)	#polynomial fitting (perhaps there is a canned interpolation routine?)
	real q95=fitvalue(0.95,polyvals)

	# Rename files
	if (pmult(iii)*100 < 10) then  
		chameleon pstr = "_p00" // trim(format(pmult(iii)*100,1,0,1))
	  else if (pmult(iii)*100 < 100) then 
	  	chameleon pstr = "_p0" // trim(format(pmult(iii)*100,2,0,1))
	  else
		chameleon pstr = "_p" // trim(format(pmult(iii)*100,3,0,1))
	  endif
	endif
	
	if (qmult(iii)*100 < 10) then  
		chameleon qstr = "_q00" // trim(format(qmult(iii)*100,1,0,1))
	  else if (qmult(iii)*100 < 100) then 
	  	chameleon qstr = "_q0" // trim(format(qmult(iii)*100,2,0,1))
	  else
		chameleon qstr = "_q" // trim(format(qmult(iii)*100,3,0,1))
	  endif
	endif

	# Setup output file names
	chameleon expeqout = "EXPEQ" // "_" // trim(shotid) // ".0" // trim(shtime) // trim(pstr) // trim(qstr)
	chameleon teqout   = "t" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq" // trim(pstr) // trim(qstr)
	chameleon ieqout   = "i" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq" // trim(pstr) // trim(qstr)
	chameleon geqout   = "g" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq" // trim(pstr) // trim(qstr)
	chameleon aeqout   = "a" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq" // trim(pstr) // trim(qstr)
	chameleon meqout   = "m" // trim(shotid) // ".0" // trim(shtime) // "_inv_teq" // trim(pstr) // trim(qstr)
	# Account for extensions
  	chameleon gfilelen = strlen(gfilelist(1))
  	if (gfilelen > 13) then
  	  chameleon shext = substr(gfilelist(1),14,gfilelen-13)
	  chameleon teqout   = "t" // trim(shotid) // ".0" // trim(shtime) // trim(shext) // "_inv_teq" // trim(pstr) // trim(qstr)
	  chameleon ieqout   = "i" // trim(shotid) // ".0" // trim(shtime) // trim(shext) // "_inv_teq" // trim(pstr) // trim(qstr)
	  chameleon geqout   = "g" // trim(shotid) // ".0" // trim(shtime) // trim(shext) // "_inv_teq" // trim(pstr) // trim(qstr)
	  chameleon aeqout   = "a" // trim(shotid) // ".0" // trim(shtime) // trim(shext) // "_inv_teq" // trim(pstr) // trim(qstr)
	  chameleon meqout   = "m" // trim(shotid) // ".0" // trim(shtime) // trim(shext) // "_inv_teq" // trim(pstr) // trim(qstr)		
  	endif
	
	# Setup unix commands and change filenames
	chameleon expcmd   = "! mv EXPEQ_corsica " // trim(expeqout)
	chameleon teqcmd   = "! mv " // trim(teqin) // " " // trim(teqout)
	chameleon ieqcmd   = "! mv " // trim(ieqin) // " " // trim(ieqout)
	chameleon geqcmd   = "! mv " // trim(geqin) // " " // trim(geqout)
	chameleon aeqcmd   = "! mv " // trim(aeqin) // " " // trim(aeqout)
	chameleon meqcmd   = "! cp " // trim(meqin) // " " // trim(meqout)
	integer expeqstat, teqstat, ieqstat, geqstat, aeqstat, meqstat
	expeqstat = basisexe(expcmd)
	teqstat   = basisexe(teqcmd)
	ieqstat   = basisexe(ieqcmd)
	geqstat   = basisexe(geqcmd)
	aeqstat   = basisexe(aeqcmd)
	meqstat   = basisexe(meqcmd)

	# Write output-file
  	iofile << format(pmult(iii),7,2,1) // format(qmult(iii),7,2,1) // format(ctroy,9,3,1)      // \
		  format(qsave(1),11,3,1)  // format(min(qsave),7,3,1) // format(q95,7,3,1) // format(max(qsave),8,3,1) // \
                  format(eq.placur*1E-6,8,4,1)       // format(eq.li(1),10,3,1)   // format(eq.li(3),7,3,1)   // \
		  format(wtotn1,8,3,1)     // format(wtotn2,8,3,1)     // format(wtotn3,8,3,1) // format(wwtotn1,8,3,1)  // \
		  format(dpwmin,9,3,1)     // format(residj,19,4,2) //  format(r0exp,18,8,2) // format(b0exp,18,8,2) 
        
	
	
  enddo #pmult loop
  #enddo #qmult loop

  #Close result file
  call basclose(iofile)
  remark "STAB_SETUP_PSAVE_QSAVE.BAS: Normal Exit"
