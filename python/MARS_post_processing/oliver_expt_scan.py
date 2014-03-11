import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import generic_funcs as gen_func
import copy

file_name = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan/shot_142614_rote_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp/shot_142614_rote_scan_w_damp_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp2/shot_142614_rote_scan_w_damp2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp3/shot_142614_rote_scan_w_damp3_post_processing_PEST.pickle'
file_name='/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30/shot_142614_rote_res_scan_30x30_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_test/shot_142614_rote_res_scan_test_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30_kpar1/shot_142614_rote_res_scan_30x30_kpar1_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/shot_142614_expt_scan_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq/shot_142614_expt_scan_const_eq_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eqV2/shot_142614_expt_scan_const_eqV2_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq_eta_10-10/shot_142614_expt_scan_const_eq_eta_10-10_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq_eta_10-5/shot_142614_expt_scan_const_eq_eta_10-5_post_processing_PEST.pickle'

phasing = 0
n = 3
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [2,0]
sort_name = 'time_list'

a = dBres_dBkink.test1(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
#fig, ax = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax = ax.flatten()
#fig2, ax2_orig = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax2 = ax2_orig.flatten()
phasings_disp = [0,45,90,135,180,225,270,315]
phasings_disp = [0,180]
phasings_disp = [0]
fig, ax = pt.subplots(nrows = 7, sharex = True)
gen_func.setup_publication_image(fig, height_prop = 1./1.618*4, single_col = True)

for i in range(len(phasings_disp)):
    #a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = ax[i], ax_matrix = ax2[i], clim = [0, 0.015])
    #tmp, color_ax = a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = None, ax_matrix = ax2[i], clim = [0, 0.025])
    #tmp, color_ax = a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = None, ax_matrix = None, clim = [0, 0.025])
    a.plot_values_vs_time(phasings_disp[i], ax = ax)
#dBres
tmp_vac_list, tmp_plas_list, tmp_tot_list, tmp_vac_list2, tmp_plas_list2,  tmp_tot_list2 = a.dB_res_single_phasing(i, phase_machine_ntor, n, a.res_vac_list_upper, a.res_vac_list_lower, a.res_plas_list_upper, a.res_plas_list_lower, a.res_tot_list_upper, a.res_tot_list_lower)
#tmp = np.sort([[t, res] for t, res in zip(a.time_list, tmp_plas_list)],axis = 0)
tmp = zip(a.time_list, tmp_plas_list)
tmp.sort()
#tmp = sorted(zip(a.time_list, tmp_plas_list), key = lambda sort_val:sort_val[0]) 
ax[2].plot([t for t, res in tmp], [res for t, res in tmp],'x-')
ax[2].set_ylabel('dBres plas')

#print a.time_list, tmp_tot_list
#tmp = np.sort([[t, res] for t, res in zip(a.time_list, tmp_tot_list)],axis = 0)
#tmp = sorted(zip(a.time_list, tmp_tot_list), key = lambda sort_val:sort_val[0]) 
tmp = zip(a.time_list, tmp_tot_list)
tmp.sort()
ax[3].plot([t for t, res in tmp], [res for t, res in tmp],'x-')
ax[3].set_ylabel('dBres tot')

a.plot_dB_res_ind_harmonics(0)

a.plot_probe_values_vs_time(0,' 66M',field='plas', ax = ax[4])

name_list = ['plot_array_plasma', 'plot_array_vac', 'plot_array_tot', 'plot_array_vac_fixed', 'q95_array', 'phasing_array', 'plot_array_plasma_fixed', 'plot_array_plasma_phase', 'plot_array_vac_phase', 'plot_array_vac_fixed_phase', 'plot_array_plasma_fixed_phase']
tmp1 = dBres_dBkink.dB_kink_phasing_dependence(a.q95_list_copy, a.lower_values_plasma, a.upper_values_plasma, a.lower_values_vac, a.upper_values_vac, a.lower_values_tot, a.upper_values_tot, a.lower_values_vac_fixed, a.upper_values_vac_fixed, phase_machine_ntor, a.upper_values_plas_fixed, a.lower_values_plas_fixed, n, phasing_array = [0])
#tmp = np.sort([[t, kink] for t, kink in zip(a.time_list, tmp1[0].flatten().tolist())],axis = 0)

tmp = zip(a.time_list, tmp1[0].flatten().tolist())
tmp.sort()

ax[5].plot([t for t, kink in tmp], [kink for t, kink in tmp],'x-')
ax[5].set_ylabel('dBkink')

tmp = np.sort([[t, q] for t, q in zip(a.time_list, a.q95_list)],axis = 0)
ax[6].plot([t for t, q in tmp], [q for t, q in tmp],'x-')
ax[6].set_ylabel('q')

for i in ax: 
    vline_times = [1600,1717,2134,1830]
    vline_labels = ['I-coil on', 'CIII image 2', 'CIII image 3', 'plas resp decay']
    for t_tmp, j in zip(vline_times, vline_labels):
        i.axvline(t_tmp)
        i.text(t_tmp,np.mean(i.get_ylim()),j,rotation = 90, verticalalignment='center')
    i.grid(True)

fig.tight_layout(pad = 0.01)
fig.savefig('comparison_oliver_data.pdf')
fig.canvas.draw(); fig.show()

#cbar = pt.colorbar(color_ax, ax = ax2.tolist())
#cbar.set_label('Displacement around x-point')
#fig.canvas.draw(); fig.show()
#fig2.savefig('res_rot_scan_displacement.pdf')
#fig2.canvas.draw(); fig2.show()
#a.eta_rote_matrix(phasing = 0, plot_type = 'plas')
