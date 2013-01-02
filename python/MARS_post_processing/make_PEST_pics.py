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
single = 0
if single:
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

dir_loc_lower_t ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.p'
dir_loc_upper_t ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.p'
dir_loc_lower_v ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.vac'
dir_loc_upper_v ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.vac'


#dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_upper.p'

d_upper_t = data(dir_loc_upper_t, I0EXP=I0EXP)
d_lower_t = data(dir_loc_lower_t, I0EXP=I0EXP)
d_upper_v = data(dir_loc_upper_v, I0EXP=I0EXP)
d_lower_v = data(dir_loc_lower_v, I0EXP=I0EXP)
d_upper_t.get_PEST(facn = facn)
d_lower_t.get_PEST(facn = facn)
d_upper_v.get_PEST(facn = facn)
d_lower_v.get_PEST(facn = facn)

subplot_phasings = 1
subplot_plot = 'total'
if subplot_phasings:
    phasings = [0,90,180,270]
    fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =1, sharey = 1)
    ax[0,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1,0].set_xlabel('m')
    ax[1,1].set_xlabel('m')
    ax = ax.flatten()
    color_plots = []
    for i, phasing in enumerate(phasings):
        print phasing
        combined = copy.deepcopy(d_upper_t)
        R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_upper_t, d_lower_t, phasing)
        R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_upper_v, d_lower_v, phasing)

        #R, Z, B1, B2, B3, Bn, BMn, BnPEST = combine_data(d_upper_t, d_lower_t, phasing)
        if subplot_plot=='total':
            combined.BnPEST = BnPEST_t
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST=1, gauss_filter = [0,0.05]))
        elif subplot_plot=='vac':
            combined.BnPEST = BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))
        elif subplot_plot=='plasma':
            combined.BnPEST = BnPEST_t - BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))

        ax[i].set_title('%d deg'%(phasing))
        color_plots[-1].set_clim([0,3.0])
    ax[0].set_xlim([0,25])
    ax[0].set_ylim([0.4,1])
    #cbar = pt.colorbar(color_plot, ax = ax)
    #ax.set_xlabel('m')
    #ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #cbar.ax.set_ylabel('G/kA')

    fig.canvas.draw(); fig.show()

animation_phasings = 0
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
        print phasing
        R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_upper_t, d_lower_t, phasing)
        R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_upper_v, d_lower_v, phasing)

        fig,ax = pt.subplots(ncols = 3, sharex=1, sharey=1)
        combined_t = copy.deepcopy(d_upper_t)
        combined_v = copy.deepcopy(d_upper_t)
        combined_p = copy.deepcopy(d_upper_t)
        combined_t.BnPEST = BnPEST_t
        combined_v.BnPEST = BnPEST_v
        combined_p.BnPEST = BnPEST_t-BnPEST_v
        contour_levels = np.linspace(0,3.0,7)
        color_plot_v = combined_v.plot_BnPEST(ax[0], n=n, inc_contours = 1, contour_levels=contour_levels, increase_grid_BnPEST = 1)
        color_plot_p = combined_p.plot_BnPEST(ax[1], n=n, inc_contours = 1, contour_levels=contour_levels, increase_grid_BnPEST = 1)
        color_plot_t = combined_t.plot_BnPEST(ax[2], n=n, inc_contours = 1, contour_levels=contour_levels,increase_grid_BnPEST = 1)

        color_plots = [color_plot_v, color_plot_p, color_plot_t]
        titles = ['Vacuum','Plasma','Total']
        for tmp_loc in range(0,len(color_plots)):
            color_plots[tmp_loc].set_clim([0,3])
            cbar = pt.colorbar(color_plots[tmp_loc], ax = ax[tmp_loc])
            ax[tmp_loc].set_title('MARS-F %s, %d deg I-coil Phasing'%(titles[tmp_loc], phasing,))
            ax[tmp_loc].set_xlabel('m')
            #ax[tmp_loc].hline(np.sqrt(0.95),-29,29,colors='b')
        ax[0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
        cbar.ax.set_ylabel(r'$\delta B_r^{m,2}$ (G/kA)', fontsize = 14)
        ax[0].set_xlim([-29,29])
        ax[0].set_ylim([0,1])
        fig.set_size_inches([ 17. ,   4.])
        #fig.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/plas_%03d.png'%(phasing,), bbox_inches = 'tight')
        fig.canvas.draw(); fig.show()
        #fig.clf()
        #pt.close('all')
    #cbar = pt.colorbar(color_plot, ax = ax)
    #ax.set_xlabel('m')
    #ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #cbar.ax.set_ylabel('G/kA')


    
