'''
SH : Nov 21 2012
This is useful for creating PEST plot images
It can make an animation and introduce different phasings

'''

from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt
import copy
import PythonMARS_funcs as pyMARS

N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I); facn = 1.0

#I0EXP = 1.0e+3*3.**1.5/(2.*np.pi)
#I0EXP = 1.0e+3*0.954 #PMZ ideal
I0EXP = 1.0e+3*0.863 #PMZ real
#I0EXP = 1.0e+3*0.827 #MPM ideal
#I0EXP = 1.0e+3*0.748 #MPM real
#I0EXP = 1.0e+3*0.412 #MPM n4 real
#I0EXP = 1.0e+3*0.528 #PMZ n4 real

#print I0EXP, 1.0e+3 * 3./np.pi
dir_loc ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/146382/qmult1.000/exp1.000/marsrun/RUNrfa.vac' 
d = data(dir_loc, I0EXP=I0EXP)
d.get_PEST(facn = facn)
fig,ax = pt.subplots()
color_plot = d.plot_BnPEST(ax, n=n, inc_contours = 1)
color_plot.set_clim([0,1.5])
cbar = pt.colorbar(color_plot, ax = ax)
ax.set_xlabel('m')
ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
cbar.ax.set_ylabel('G/kA')

fig.canvas.draw(); fig.show()

#d = data(,I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_lower.p',I0EXP=I0EXP)

#dir_loc_lower ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_lower.vac'
#dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_upper.vac'

dir_loc_lower ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_lower.p'
dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_upper.p'

dir_loc_lower ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.p'
dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.p'

#dir_loc_lower ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.vac'
#dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.vac'
#dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_upper.p'

d_upper = data(dir_loc_upper, I0EXP=I0EXP)
d_lower = data(dir_loc_lower, I0EXP=I0EXP)
d_upper.get_PEST(facn = facn)
d_lower.get_PEST(facn = facn)
subplot_phasings = 0
if subplot_phasings:
    phasings = [0,120,180,270]
    fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =1, sharey = 1)
    ax[0,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1,0].set_xlabel('m')
    ax[1,1].set_xlabel('m')
    ax = ax.flatten()
    color_plots = []
    for i, phasing in enumerate(phasings):
        print phasing
        combined = copy.deepcopy(d_upper)
        R, Z, B1, B2, B3, Bn, BMn, BnPEST = combine_data(d_upper, d_lower, phasing)
        combined.BnPEST = BnPEST
        color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))
        ax[i].set_title('%d deg'%(phasing))
        color_plots[-1].set_clim([0,1.5])
    ax[0].set_xlim([0,25])
    ax[0].set_ylim([0.4,1])
    #cbar = pt.colorbar(color_plot, ax = ax)
    #ax.set_xlabel('m')
    #ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #cbar.ax.set_ylabel('G/kA')

    fig.canvas.draw(); fig.show()

animation_phasings = 1
if animation_phasings:
    phasings = range(0,360,15)
    phasings = [0]
    #fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =1, sharey = 1)
    #ax[0,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #ax[1,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #ax[1,0].set_xlabel('m')
    #ax[1,1].set_xlabel('m')
    #ax = ax.flatten()
    #color_plots = []
    for i, phasing in enumerate(phasings):
        fig,ax = pt.subplots()
        print phasing
        combined = copy.deepcopy(d_upper)
        R, Z, B1, B2, B3, Bn, BMn, BnPEST = combine_data(d_upper, d_lower, phasing)
        combined.BnPEST = BnPEST

        color_plot = combined.plot_BnPEST(ax, n=n, inc_contours = 1)
        color_plot.set_clim([0,3])
        ax.set_title('MARS-F Total, %d deg I-coil Phasing'%(phasing))
        cbar = pt.colorbar(color_plot, ax = ax)
        #color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))
        #color_plots[-1].set_clim([0,1.5])
        ax.set_xlabel('m')
        ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
        cbar.ax.set_ylabel(r'$\delta B_r$ (G/kA)')
        ax.set_xlim([-29,29])
        ax.set_ylim([0,1])
        fig.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/plas_%03d.png'%(phasing,))
        #fig.canvas.draw(); fig.show()
        fig.clf()
        pt.close('all')
    #cbar = pt.colorbar(color_plot, ax = ax)
    #ax.set_xlabel('m')
    #ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #cbar.ax.set_ylabel('G/kA')


    
