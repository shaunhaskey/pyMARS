'''
SH : Nov 21 2012
This is useful for creating PEST plot images
It can make an animation and introduce different phasings
'''

from pyMARS.results_class import *
import pyMARS.results_class as res_class
from pyMARS.RZfuncs import I0EXP_calc
import pyMARS.generic_funcs as gen_funcs
import numpy as np
import matplotlib.pyplot as pt
import copy
import PythonMARS_funcs as pyMARS
N = 6; n = 3
I = np.array([1.,-1.,1.,-1,1.,-1.])
I0EXP = I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

#various simulation directories to get the components
base_dir = '/u/haskeysr/mars/raffi_157312_n3RMP_ideal/qmult1.000/exp1.000/RES0.0000_ROTE0.0000/'
base_dir = '/u/haskeysr/mars/raffi_157312_n3RMP/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
base_dir = '/u/haskeysr/mars/raffi_157312_n3RMP_0-1_ROTE/qmult1.000/exp1.000/RES-100000000.0000_ROTE0.4800/'
base_dir = '/u/haskeysr/mars/raffi_157312_n3RMP_0-01_ROTE/qmult1.000/exp1.000/RES-100000000.0000_ROTE0.0480/'
#base_dir = '/u/haskeysr/mars/raffi_157312_n3RMP_0-001_ROTE/qmult1.000/exp1.000/RES-100000000.0000_ROTE0.0048/'
dir_loc_lower_t =base_dir + '/RUN_rfa_lower.p'
dir_loc_upper_t =base_dir + '/RUN_rfa_upper.p'
dir_loc_lower_v =base_dir + '/RUN_rfa_lower.vac'
dir_loc_upper_v =base_dir + '/RUN_rfa_upper.vac'

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
subplot_plot = 'plasma'
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
            #color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST=1, gauss_filter = [0,0.05]))
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST = 1, gauss_filter = None,phase_ref = True, phase_ref_array = BnPEST_v))
        elif subplot_plot=='vac':
            combined.BnPEST = BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))
        elif subplot_plot=='plasma':
            combined.BnPEST = BnPEST_t - BnPEST_v
            #color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1))
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST = 1, gauss_filter = None,phase_ref = True, phase_ref_array = BnPEST_v))
        ax[i].set_title(r'$\Delta \phi_{ul} = %d^o$'%(phasing),fontsize = 18)
        color_plots[-1].set_clim([0,3.0])
    ax[0].set_xlim([0,25])
    ax[0].set_ylim([0.4,1])
    ax[0].set_xlim([6,18])
    ax[0].set_ylim([0.8,1])
    #cbar = pt.colorbar(color_plot, ax = ax)
    #ax.set_xlabel('m')
    #ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #cbar.ax.set_ylabel('G/kA')
    
    fig.canvas.draw(); fig.show()

import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func
file_names = ['/u/haskeysr/mars/raffi_157312_n3RMP/raffi_157312_n3RMP_post_processing_PEST.pickle','/u/haskeysr/mars/raffi_157312_n3RMP_ideal/raffi_157312_n3RMP_ideal_post_processing_PEST.pickle',]
file_names = ['/u/haskeysr/mars/raffi_157312_n3RMP_ideal/raffi_157312_n3RMP_ideal_post_processing_PEST.pickle',
              '/u/haskeysr/mars/raffi_157312_n3RMP/raffi_157312_n3RMP_post_processing_PEST.pickle',
              '/u/haskeysr/mars/raffi_157312_n3RMP_0-1_ROTE/raffi_157312_n3RMP_0-1_ROTE_post_processing_PEST.pickle',
              '/u/haskeysr/mars/raffi_157312_n3RMP_0-01_ROTE/raffi_157312_n3RMP_0-01_ROTE_post_processing_PEST.pickle',
              '/u/haskeysr/mars/raffi_157312_n3RMP_0-001_ROTE/raffi_157312_n3RMP_0-001_ROTE_post_processing_PEST.pickle']
fig_harms, ax_harms = pt.subplots(nrows = len(file_names), sharex = True, sharey = True)
gen_func.setup_publication_image(fig_harms, height_prop = 1./1.618 * 1.75/2*len(file_names), single_col = True)
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [4,0]
sort_name = 'time_list'

for file_name, cur_ax in zip(file_names, ax_harms):
    reference_dB_kink = 'plasma'
    a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)

    dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'sum')
    dBkink = dBres_dBkink.dBkink_calculations(a)
    probe = dBres_dBkink.magnetic_probe(a,' 66M')
    xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)
    tmp_a = np.array(dBres.single_phasing_individual_harms(phasing,field='plasma'))
    tmp_b = np.array(dBres.single_phasing_individual_harms(phasing,field='total'))
    tmp_c = np.array(dBres.single_phasing_individual_harms(phasing,field='vacuum'))
    min_time = 0;max_time = 20075
    min_shot_time = np.min(a.raw_data['shot_time'])
    min_shot_time = min_time
    max_shot_time = np.max(a.raw_data['shot_time'])
    max_shot_time = max_time
    range_shot_time = max_shot_time - min_shot_time
    initial = 0
    for i in range(0,tmp_a.shape[0]):
        x_axis = dBres.raw_data['res_m_vals'][i]
        clr = (a.raw_data['shot_time'][i] - min_shot_time)/float(range_shot_time)
        clr = clr*0.9
        if int(a.raw_data['shot_time'][i])>=min_time and int(a.raw_data['shot_time'][i])<=max_time:
            #cur_ax.plot(x_axis, np.abs(tmp_b[i,:]), color=str(clr), marker = 'x')
            cur_ax.plot(x_axis, np.abs(tmp_b[i]), color=str(clr), marker = 'x')
            #cur_ax.plot(x_axis, np.abs(tmp_c[i]), color='b', marker = '.')
            cur_ax.plot(x_axis, np.abs(tmp_c[i]), color=str(clr), marker = '.')
            if initial==0:
                pass
                #cur_ax.text(13,2.1,'Vacuum')
                #cur_ax.text(13,1.13,'Vacuum + Plasma')
                #cur_ax.plot(x_axis, np.abs(tmp_c[i,:]), color='b', marker = '.')
                #cur_ax.plot(x_axis, np.abs(tmp_c[i]), color='b', marker = '.')
            initial += 1
#ax_harms[0].set_title('{}-{}ms $\eta=${n}'.format(min_time, max_time, eta))
ax_harms[-1].set_xlabel('m')
#ax_harms[-1].set_ylim([0,2.5])
tmp_ylim = ax_harms[-1].get_ylim()
#ax_harms[0].text(6,tmp_ylim[1]*0.85,'(a)')
#ax_harms[1].text(6,tmp_ylim[1]*0.85,'(b)')
for i in ax_harms: i.set_ylabel('Resonant harm amp (G/kA)')
#ax_harms[1].set_ylabel('Resonant harm phase (rad)')
for i in ax_harms:i.grid(True)
fig_harms.tight_layout(pad = 0.1)
#for end in ['svg','eps','pdf']:fig_harms.savefig('harms_{}_{}.{}'.format(eta, const_rot, end))
fig_harms.canvas.draw(); fig_harms.show()



1/0
animation_phasings = 1
filename_list = []
if animation_phasings:
    phasings = range(0,360,15)
    #phasings = [0]
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
        rmax = 3.5
        color_plot_v = combined_v.plot_BnPEST(ax[0], n=n, inc_contours = 1, contour_levels=contour_levels1, increase_grid_BnPEST = 1,cmap = 'RdBu', phase_ref = True, rmax = rmax, phase_ref_array = combined_v.BnPEST*0+1)
        color_plot_p = combined_p.plot_BnPEST(ax[1], n=n, inc_contours = 1, contour_levels=contour_levels2, increase_grid_BnPEST = 1, cmap = 'RdBu', phase_ref = True, rmax = rmax, phase_ref_array = combined_v.BnPEST)
        color_plot_t = combined_t.plot_BnPEST(ax[2], n=n, inc_contours = 1, contour_levels=contour_levels2, increase_grid_BnPEST = 1, cmap = 'RdBu', phase_ref = True, rmax = rmax, phase_ref_array = combined_v.BnPEST)

        color_plots = [color_plot_v, color_plot_p, color_plot_t]
        titles = ['Vac Only','Plasma Only','Total']
        for tmp_loc in range(0,len(color_plots)):
            color_plots[tmp_loc].set_clim([-np.pi,np.pi])
            #cbar = pt.colorbar(color_plots[tmp_loc], ax = ax[tmp_loc])
            #ax[tmp_loc].set_title('MARS-F %s, %d deg I-coil Phasing'%(titles[tmp_loc], phasing,))
            ax[tmp_loc].set_xlabel('m')
            ax[tmp_loc].set_title(titles[tmp_loc] + ' {}deg'.format(phasing))
            #ax[tmp_loc].hline(np.sqrt(0.95),-29,29,colors='b')
        ax[0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
        #cbar.ax.set_ylabel(r'$\delta B_r^{m,2}$ (G/kA)', fontsize = 14)
        ax[0].set_xlim([0,15])
        ax[0].set_ylim([0.4,0.995])

        ax[1].annotate('kink-\nresonant', xy=(8, 0.9), xytext=(8.9, 0.6),arrowprops=dict(facecolor='black', shrink=0.05,ec='white'),color='white')
        ax[1].annotate('pitch-\nresonant', xy=(6.93, 0.952), xytext=(0.3, 0.7),arrowprops=dict(facecolor='black', shrink=0.05,ec='white'),color='white')
        #ax[1].text(3.5,0.95,"kink-resonant response",rotation=70,fontsize=15,horizontalalignment='left')
        #fig.set_size_inches([ 17. ,   4.])
        hue_sat = True
        if not hue_sat:
            cbar = pt.colorbar(color_plot_v, cax = cbar_axes ,orientation='horizontal')
            cbar.ax.set_xlabel('G/kA')
        else:
            cbar = pt.colorbar(color_plot_v, cax = cbar_axes ,orientation='horizontal')
            #gen_funcs.create_cbar_ax(original_ax, pad = 3, loc = "right", prop = 5):
            cbar.ax.cla()
            #cbar = pt.colorbar(color_plot_v, cax = cbar_axes ,orientation='horizontal')
            res_class.hue_sat_cbar(cbar.ax, rmax = rmax)
            gen_funcs.setup_axis_publication(cbar.ax, n_xticks = 5, n_yticks = 2)
            cbar.ax.set_yticks([0,4.5])
        filename_list.append('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/plas_%03d.png'%(phasing))
        fig.savefig(filename_list[-1], bbox_inches = 'tight', dpi = 150)
        #fig.canvas.draw(); fig.show()
        #fig.clf()
        #pt.close('all')
    #cbar = pt.colorbar(color_plot, ax = ax)
    #ax.set_xlabel('m')
    #ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    #cbar.ax.set_ylabel('G/kA')
print 'convert -delay {} -loop 0 {} {}'.format(100, ' '.join(filename_list), 'test.gif')
os.system('convert -delay {} -loop 0 {} {}'.format(20, ' '.join(filename_list), 'changed_phasing.gif'))

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
