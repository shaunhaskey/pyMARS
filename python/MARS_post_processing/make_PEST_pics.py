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
I0EXP = I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

# #n=2 plots
# d = data('/home/srh112/NAMP_datafiles/mars/shot146382_NVEXP_4/qmult1.000/exp1.000/marsrun/RUNrfa.vac',I0EXP=I0EXP)
# d.get_PEST(facn = facn)
# d.load_SURFMN_data('/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/SURF146382.03230.ph000.pmz/surfmn.out.idl3d', n, horizontal_comparison=0, PEST_comparison=1, single_radial_mode_plots=1,all_radial_mode_plots=0)
# #d.plot1(inc_phase=0,clim_value=[0,0.6], surfmn_file = '/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/SURF146382.03230.ph000.pmz/surfmn.out.idl3d', ss_squared = 0, n=n, single_mode_plots2 = [1,3,9])


single = 1
if single:
    dir_loc ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/146382/qmult1.000/exp1.000/marsrun/RUNrfa.vac' 
    dir_loc ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/146382/qmult1.000/exp1.000/marsrun/RUNrfa.p' 
    dir_loc ='/home/srh112/NAMP_datafiles/mars/146382_thetac_003/qmult1.000/exp1.000/marsrun/RUNrfa.p'
    dir_loc ='/home/srh112/NAMP_datafiles/mars/146382_thetac_006/qmult1.000/exp1.000/marsrun/RUNrfa.p'
    dir_loc ='/home/srh112/NAMP_datafiles/mars/146382_thetac_010/qmult1.000/exp1.000/marsrun/RUNrfa.p'
    dir_loc ='/home/srh112/NAMP_datafiles/mars/146382_thetac_020/qmult1.000/exp1.000/marsrun/RUNrfa.p'
    d = data(dir_loc, I0EXP=I0EXP)
    d.get_PEST(facn = facn)
    fig,ax = pt.subplots()
    color_plot = d.plot_BnPEST(ax, n=n, inc_contours = 1)
    color_plot.set_clim([0,1.5])
    color_plot.set_clim([0,4.5])
    cbar = pt.colorbar(color_plot, ax = ax)
    ax.set_xlabel('m')
    ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    cbar.ax.set_ylabel('G/kA')
    fig.canvas.draw(); fig.show()

1/0
#d = data(,I0EXP=I0EXP)
#d = data('/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_lower.p',I0EXP=I0EXP)

#dir_loc_lower ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_lower.vac'
#dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_upper.vac'

#dir_loc_lower ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_lower.p'
#dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_upper.p'

#various simulation directories to get the components
dir_loc_lower_t ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.p'
dir_loc_upper_t ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.p'
dir_loc_lower_v ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.vac'
dir_loc_upper_v ='/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.vac'

#dir_loc_upper ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/exp1.303/marsrun/RUN_rfa_upper.p'

#Load data including PEST data
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
    fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =True, sharey = True)
    ax[0,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1,0].set_xlabel('m')
    ax[1,1].set_xlabel('m')
    ax = ax.flatten()
    color_plots = []
    for i, phasing in enumerate(phasings):
        print phasing
        combined = copy.deepcopy(d_upper_t)
        
        #Combine the upper and lower data with the appropriate phasing
        R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_upper_t, d_lower_t, phasing)
        R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_upper_v, d_lower_v, phasing)

        #Choose which plot to create
        if subplot_plot=='total':
            combined.BnPEST = BnPEST_t
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST=1, gauss_filter = [0,0.05]))
        elif subplot_plot=='vac':
            combined.BnPEST = BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))
        elif subplot_plot=='plasma':
            combined.BnPEST = BnPEST_t - BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))

        ax[i].set_title(r'$\Delta \phi_{ul} = %d^o$'%(phasing),fontsize = 18)
        color_plots[-1].set_clim([0,3.0])
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
    #fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =True, sharey = True)
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

        fig,ax = pt.subplots(ncols = 3, sharex=True, sharey=True)
        pt.subplots_adjust(wspace = .05)
        move_up = 0.15
        for j_tmp in range(0,len(ax)):
            ax_tmp = ax[j_tmp]
            start_point = ax_tmp.get_position().bounds
            if j_tmp==0:
                x_bot = start_point[0]
                y_bot = start_point[1]
                print x_bot
            if j_tmp==len(ax)-1:
                x_width = (start_point[0]+start_point[2])-x_bot
                print x_width
            start_point = (start_point[0],start_point[1]+move_up,start_point[2],start_point[3]-move_up)
            ax_tmp.set_position(start_point)
        cbar_axes = fig.add_axes([x_bot,y_bot,x_width,move_up-0.1])


        combined_t = copy.deepcopy(d_upper_t)
        combined_v = copy.deepcopy(d_upper_t)
        combined_p = copy.deepcopy(d_upper_t)
        combined_t.BnPEST = BnPEST_t
        combined_v.BnPEST = BnPEST_v
        combined_p.BnPEST = BnPEST_t-BnPEST_v
        contour_levels1 = np.linspace(0,3.0,7)
        contour_levels2 = np.linspace(0,5.0,7)
        color_plot_v = combined_v.plot_BnPEST(ax[0], n=n, inc_contours = 1, contour_levels=contour_levels1, increase_grid_BnPEST = 1)
        color_plot_p = combined_p.plot_BnPEST(ax[1], n=n, inc_contours = 1, contour_levels=contour_levels2, increase_grid_BnPEST = 1)
        color_plot_t = combined_t.plot_BnPEST(ax[2], n=n, inc_contours = 1, contour_levels=contour_levels2, increase_grid_BnPEST = 1)

        color_plots = [color_plot_v, color_plot_p, color_plot_t]
        titles = ['Vacuum','Plasma','Total']
        for tmp_loc in range(0,len(color_plots)):
            color_plots[tmp_loc].set_clim([0,3])
            #cbar = pt.colorbar(color_plots[tmp_loc], ax = ax[tmp_loc])
            #ax[tmp_loc].set_title('MARS-F %s, %d deg I-coil Phasing'%(titles[tmp_loc], phasing,))
            ax[tmp_loc].set_xlabel('m')
            ax[tmp_loc].set_title(titles[tmp_loc])
            #ax[tmp_loc].hline(np.sqrt(0.95),-29,29,colors='b')
        ax[0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
        #cbar.ax.set_ylabel(r'$\delta B_r^{m,2}$ (G/kA)', fontsize = 14)
        ax[0].set_xlim([0,15])
        ax[0].set_ylim([0.4,0.995])

        ax[1].annotate('kink-\nresonant', xy=(8, 0.9), xytext=(8.9, 0.6),arrowprops=dict(facecolor='black', shrink=0.05,ec='white'),color='white')
        ax[1].annotate('pitch-\nresonant', xy=(6.93, 0.952), xytext=(0.3, 0.7),arrowprops=dict(facecolor='black', shrink=0.05,ec='white'),color='white')
        #ax[1].text(3.5,0.95,"kink-resonant response",rotation=70,fontsize=15,horizontalalignment='left')
        #fig.set_size_inches([ 17. ,   4.])
        cbar = pt.colorbar(color_plot_v, cax = cbar_axes ,orientation='horizontal')
        cbar.ax.set_xlabel('G/kA')
        #fig.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/plas_%03d.png'%(phasing,), bbox_inches = 'tight')
        fig.canvas.draw(); fig.show()
        #fig.clf()
        #pt.close('all')
    #cbar = pt.colorbar(color_plot, ax = ax)
    #ax.set_xlabel('m')
    #ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #cbar.ax.set_ylabel('G/kA')

tmp_fig, tmp_ax = pt.subplots()
#cbar = pt.colorbar(color_plots[0], ax = tmp_ax,orientation='horizontal')
cbar = pt.colorbar(color_plots[0], ax = tmp_ax,orientation='horizontal')
cbar.ax.set_xlabel('G/kA')
tmp_fig.canvas.draw(); tmp_fig.show()


#n=4 plots
I0EXP = I0EXP_calc_real(4,I)
n4_data = data('/home/srh112/NAMP_datafiles/mars/shot146382_single_n4/qmult1.000/exp1.000/marsrun/RUNrfa.vac', I0EXP=I0EXP)
n4_data.get_PEST(facn = facn)
n4_data.load_SURFMN_data('/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/SURF146382.03230.ph000.pmz/surfmn.out.idl3d', 4, horizontal_comparison=0, PEST_comparison=1, single_radial_mode_plots=1,all_radial_mode_plots=0)


# fig,ax = pt.subplots(ncols = 3, sharex=True, sharey=True)
# pt.subplots_adjust(wspace = .05)
# move_up = 0.15
# for j in range(0,len(ax)):
#     i = ax[j]
#     start_point = i.get_position().bounds
#     if j==0:
#         x_bot = start_point[0]
#         y_bot = start_point[1]
#         print x_bot
#     if j==len(ax)-1:
#         x_width = (start_point[0]+start_point[2])-x_bot
#         print x_width
#     start_point = (start_point[0],start_point[1]+move_up,start_point[2],start_point[3]-move_up)
#     i.set_position(start_point)

# new_ax = fig.add_axes([x_bot,y_bot,x_width,move_up-0.1])
# cbar = pt.colorbar(color_plots[0], cax = new_ax ,orientation='horizontal')
# cbar.ax.set_xlabel('G/kA')
# fig.canvas.draw();fig.show()
