'''
Generates a plot of the field decomposed into poloidal harmonics. A plot with plasma and without is shown from some random configurations
These plots are for comparison with SURFMN plots
'''

from  results_class import *
from RZfuncs import I0EXP_calc
from RZfuncs import I0EXP_calc_real
import numpy as np
import matplotlib.pyplot as pt

import PythonMARS_funcs as pyMARS

N = 6
n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?
I0EXP = I0EXP_calc_real(n,I)
#I0EXP = 1.0e+3*3.**1.5/(2.*np.pi)
#I0EXP = 1.0e+3*0.954 #PMZ ideal
#I0EXP = 1.0e+3*0.863 #PMZ real
#I0EXP = 1.0e+3*0.827 #MPM ideal
#I0EXP = 1.0e+3*0.748 #MPM real
#I0EXP = 1.0e+3*0.412 #MPM n4 real
#I0EXP = 1.0e+3*0.528 #PMZ n4 real

print I0EXP, 1.0e+3 * 3./np.pi

#a = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.p',I0EXP = I0EXP)
#a.plot_Bn(start_surface = 0,end_surface = a.Ns1+24,skip=1)
#a.amp_plot.set_clim([0,20])
#a.fig.canvas.draw()

#c = data('/home/srh112/code/pyMARS/other_scripts/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.p',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/shot146382_single2/qmult1.000/exp1.000/marsrun/RUNrfa.vac', I0EXP=I0EXP)
#d = data('/home/srh112/Desktop/Test_Case/PEST_files/marsrun/RUNrfa.p', I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/shot146398_0deg/qmult1.000/exp1.000/amarsrun/RUNrfa.vac', I0EXP=I0EXP)


#d = data('/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/mars_files/RUNrfa.p/', I0EXP=I0EXP)

##Jan 2
#d = data('/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_lower.p',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/plotk_rzplot/146382/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)

#d = data('/home/srh112/NAMP_datafiles/mars/shot146382_single_n4/qmult1.000/exp1.000/marsrun/RUNrfa.vac', I0EXP=I0EXP)
#d = data('/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/mars_files/RUNrfa.vac/', I0EXP=I0EXP)

#d = data('/home/srh112/NAMP_datafiles/mars/shot146398_0_MARS_SURFMN/qmult1.000/exp1.000/marsrun/RUNrfa.vac', I0EXP=I0EXP)



########These are for checking the psi_N offset between SURMN and MARS-F
#d = data('/home/srh112/NAMP_datafiles/mars/146382_thetac_003/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/146382_thetac_006/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/146382_thetac_010/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/146382_thetac_020/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/146382_thetac_003_res_wall/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/146382_thetac_003_high_freq/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)

#c = data('/home/srh112/code/pyMARS/other_scripts/shot146382_single2/qmult1.000/exp1.000/marsrun/RUNrfa.p',I0EXP=I0EXP)
#d = data('/home/srh112/code/pyMARS/other_scripts/shot146382_single2/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#####Offset check using more dense gridding with 146388 ########
# Note for 146388 usre this surfmn_file = '/home/srh112/NAMP_datafiles/SURFMN/SURF146388.03233.ph000.pmz/surfmn.out.idl3d
#d = data('/home/srh112/NAMP_datafiles/mars/grid_check1/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/grid_check2/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/grid_check3/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/grid_check10/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/shot146382_single2_R0EXT_2/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/shot146382_single2/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#####Offset check using more dense gridding with 146382 ########
#d = data('/home/srh112/NAMP_datafiles/mars/shot146382_REXP_2/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/shot146382_REXP_7/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
d = data('/home/srh112/NAMP_datafiles/mars/shot146382_NVEXP_4/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)


#c = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.p', I0EXP = I0EXP)
#d = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.vac', I0EXP = I0EXP)



#c.get_PEST(facn = facn)
#c.plot1(inc_phase=0,clim_value=[0,1.2], surfmn_file = None)#'/home/srh112/code/python/NAMP_analysis/spectral_info.h5',ss_squared = 1)

d.get_PEST(facn = facn)
#d.plot1(inc_phase=0,clim_value=[0,0.6], surfmn_file = '/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/mars_files/RUNrfa.vac/spectral_info_pmz.h5', ss_squared = 0)



d.plot1(inc_phase=0,clim_value=[0,0.6], surfmn_file = '/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/SURF146382.03230.ph000.pmz/surfmn.out.idl3d', ss_squared = 0, n=2, single_mode_plots2 = [1,3,9])

#d.plot1(inc_phase=0,clim_value=[0,0.6], surfmn_file = '/home/srh112/NAMP_datafiles/SURFMN/SURF146388.03233.ph000.pmz/surfmn.out.idl3d', ss_squared = 0, n=2, single_mode_plots2 = [1,3,9])



# c.plot_Bn(start_surface = 0,end_surface = c.Ns1+24+2,skip=1, modification = d.Bn, field = 'Bn')

# c.amp_plot.set_clim([0,10])
# c.fig.canvas.draw()

# probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
# #probe type 1: poloidal field, 2: radial field
# probe_type   = np.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
# #Poloidal geometry
# Rprobe = np.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
# Zprobe = np.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
# tprobe = np.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*np.pi/360  #DTOR # poloidal inclination
# lprobe = np.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

# Br = c.Br - d.Br
# Bz = c.Bz - d.Bz
# Bphi = c.Bphi - d.Bphi
# grid_r = c.R*c.R0EXP
# grid_z = c.Z*c.R0EXP

# i = 4

# R_tot_array, Z_tot_array, interp_values, Bprobem = coil_responses6(grid_r, grid_z, Br, Bz, Bphi, [probe[i]], [probe_type[i]], [Rprobe[i]],[Zprobe[i]],[tprobe[i]],[lprobe[i]])


# fig = pt.figure()
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax.plot(range(0,len(Br[c.Ns1+23,:])),np.abs(Br[c.Ns1+23,:]), 'b,')
# ax.plot(range(0,len(Br[c.Ns1+22,:])),np.abs(Br[c.Ns1+22,:]), 'k-,')
# ax.plot(range(0,len(Br[c.Ns1+24,:])),np.abs(Br[c.Ns1+24,:]), 'r-,')


# ax.plot(range(0,len(Br[c.Ns1+23,:])),c.Z[c.Ns1+23,:], '-')
# ax.plot(range(0,len(Br[c.Ns1+23,:])),c.R[c.Ns1+23,:], '-')
# ax2.plot(range(0,len(Br[c.Ns1+23,:])),np.angle(Br[c.Ns1+23,:],deg=True), '-')
# fig.canvas.draw()
# fig.show()

# fig = pt.figure()
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212, sharex = ax)


# for surface in [c.Ns1+22,c.Ns1+23,c.Ns1+24]:
#     ax.plot(np.arctan2(grid_z[surface,:],grid_r[surface,:]-np.mean(grid_r[0,:]))*180/np.pi, np.abs(Br[surface,:]), '.', label = 'surf_'+str(surface))
#     ax2.plot(np.arctan2(grid_z[surface,:],grid_r[surface,:]-np.mean(grid_r[0,:]))*180/np.pi, np.angle(Br[surface,:],deg=True), '.', label = 'surf_'+str(surface))
# y_limits = np.array(ax.get_ylim())

# angles = np.arctan2(Z_tot_array, R_tot_array-np.mean(grid_r[0,:]))*180./np.pi
# #angle = np.arctan2(0.4, 2.431-np.mean(grid_r[0,:]))*180./np.pi
# ax2.set_xlabel(r'$\theta - deg - real space$')
# ax2.set_ylabel('Phase (deg)')
# ax.set_ylabel('mag G/kA')
# ax.set_title('Br')
# #ax.plot([angle, -angle], [np.mean(y_limits),np.mean(y_limits)])
# ax.plot(angles.flatten(), np.abs(interp_values).flatten(),'-o',label='pickup')
# #ax.plot(angles, np.abs(interp_values),'.',label='pickup'),'.',label='pickup')
# ax2.plot(angles.flatten(), np.angle(interp_values,deg=True).flatten(),'-o',label='pickup')
# #ax.vlines([angle, -angle],y_limits[0],y_limits[1])
# ax.legend()
# fig.canvas.draw()
# fig.show()
