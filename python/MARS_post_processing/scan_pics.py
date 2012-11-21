'''
SH : Nov19 2012 - This will go through a q95 scan style pickle output, and 
generate plots of the PEST results (vac, total, or plasma), so they can be
viewed to see what is going on.
'''
import results_class, copy
import RZfuncs
import numpy as np
import matplotlib.pyplot as pt
import PythonMARS_funcs as pyMARS
from scipy.interpolate import griddata
import pickle
import matplotlib.cm as cm

file_name = '/u/haskeysr/mars/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'
N = 6; n = 2; I = np.array([1.,-1.,0.,1,-1.,0.])

project_dict = pickle.load(file(file_name,'r'))
key_list = project_dict['sims'].keys()
q95_list = []; Bn_Li_list = []
for i in key_list:
    q95_list.append(project_dict['sims'][i]['Q95'])

key_list_arranged = []
q95_list_arranged = []; Bn_Li_list_arranged = []; mode_list_arranged = []
q95_list_copy = copy.deepcopy(q95_list)

for i in range(0,len(q95_list)):
    cur_loc = np.argmin(q95_list)
    q95_list_arranged.append(q95_list.pop(cur_loc))
    Bn_Li_list_arranged.append(Bn_Li_list.pop(cur_loc))
    mode_list_arranged.append(mode_list.pop(cur_loc))
    time_list_arranged.append(time_list.pop(cur_loc))
    key_list_arranged.append(key_list.pop(cur_loc))

plot_quantity = 'total'
#I0EXP = I0EXP_calc(N,n,I)
I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])

facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

for i in project_dict['sims'].keys():
    q95_cur = project_dict['sims'][i]['Q95']
    print '===========',i,'==========='
    if plot_quantity=='total' or plot_quantity=='plasma':
        upper_file_loc = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
        lower_file_loc = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']
    elif plot_quantity=='vacuum':
        upper_file_loc = project_dict['sims'][i]['dir_dict']['mars_upper_vac_dir']
        lower_file_loc = project_dict['sims'][i]['dir_dict']['mars_lower_vac_dir']
    elif plot_quantity=='plasma':
        upper_file_loc_vac = project_dict['sims'][i]['dir_dict']['mars_upper_vac_dir']
        lower_file_loc_vac = project_dict['sims'][i]['dir_dict']['mars_lower_vac_dir']
        upper_file_loc = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
        lower_file_loc = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']

    upper = results_class.data(upper_file_loc, I0EXP=I0EXP)
    lower = results_class.data(lower_file_loc, I0EXP=I0EXP)
    upper.get_PEST(facn = facn)
    lower.get_PEST(facn = facn)
    tmp_R, tmp_Z, upper.B1, upper.B2, upper.B3, upper.Bn, upper.BMn, upper.BnPEST = results_class.combine_data(upper, lower, 0)

    if plot_quantity=='plasma':
        upper_vac = results_class.data(upper_file_loc_vac, I0EXP=I0EXP)
        lower_vac = results_class.data(lower_file_loc_vac, I0EXP=I0EXP)
        upper_vac.get_PEST(facn = facn)
        lower_vac.get_PEST(facn = facn)
        tmp_R, tmp_Z, upper_vac.B1, upper_vac.B2, upper_vac.B3, upper_vac.Bn, upper_vac.BMn, upper_vac.BnPEST = results_class.combine_data(upper_vac, lower_vac, 0)

        upper.B1 = upper.B1 - upper_vac.B1
        upper.B2 = upper.B2 - upper_vac.B2
        upper.B3 = upper.B3 - upper_vac.B3
        upper.Bn = upper.Bn - upper_vac.Bn
        upper.BMn = upper.BMn - upper_vac.BMn
        upper.BnPEST = upper.BnPEST - upper_vac.BnPEST

    print plot_quantity, i, q95_cur
    #suptitle = '%s key: %d, q95: %.2f, max_amp: %.2f, psi: %.2f, m_max: %d'%(plot_quantity, i, q95_list_arranged[i], plot_quantity_plas_arranged[i], psi, mode_list_arranged[i])
    suptitle = '%s key: %d, q95: %.2f'%(plot_quantity, i,)

    integer_part = int(q95_cur)
    other_part = int((q95_cur - integer_part)*1000)
    
    upper.plot_BnPEST(ax, n=2, inc_contours = 1)
    fig, ax = pt.subplots()
    color_plot = upper.plot_BnPEST(ax, n=n, inc_contours = 1)
    color_plot.set_clim([0, 2.0])
    cbar = pt.colorbar(color_plot, ax = ax)
    ax.set_xlabel('m')
    ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax.set_title(suptitle)
    cbar.ax.set_ylabel('G/kA')
    fig.savefig('/u/haskeysr/q%02d_%03d_scan.png'%(integer_part, other_part))
    fig.clf()
    pt.close('all')
    #upper.plot1(suptitle = suptitle, inc_phase=0, clim_value=[0,2], ss_squared = 0, fig_show=0, fig_name='/u/haskeysr/q%02d_%03d_scan.png'%(integer_part, other_part))
