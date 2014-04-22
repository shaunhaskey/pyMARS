import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func
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

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV1/shot_142614_expt_scan_NC_const_eqV1_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV2/shot_142614_expt_scan_NC_const_eqV2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV3/shot_142614_expt_scan_NC_const_eqV3_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV4/shot_142614_expt_scan_NC_const_eqV4_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV5/shot_142614_expt_scan_NC_const_eqV5_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV6/shot_142614_expt_scan_NC_const_eqV6_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eqV3/shot_142614_expt_scan_const_eqV3_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq_eta_10-10/shot_142614_expt_scan_const_eq_eta_10-10_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq_eta_10-5/shot_142614_expt_scan_const_eq_eta_10-5_post_processing_PEST.pickle'

phasing = 270
n = 3
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [2,0]
sort_name = 'time_list'

fig, ax = pt.subplots(nrows = 8, sharex = True)
V_dict = {1:'1e-7',2:'7e-8',3:'3e-8',4:'1e-8',5:'5e-7',6:'1e-6'}

for V in [1,2,3,4,5]:
    file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV{}/shot_142614_expt_scan_NC_const_eqV{}_post_processing_PEST.pickle'.format(V,V)
    #file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_V{}/shot_142614_expt_scan_NC_V{}_post_processing_PEST.pickle'.format(V,V)
    reference_dB_kink = 'plasma'
    a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)


    dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'sum')
    dBkink = dBres_dBkink.dBkink_calculations(a)
    probe = dBres_dBkink.magnetic_probe(a,' 66M')
    xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)

    tmp_a = np.array(dBres.single_phasing_individual_harms(phasing,field='plasma'))
    tmp_b = np.array(dBres.single_phasing_individual_harms(phasing,field='total'))
    tmp_c = np.array(dBres.single_phasing_individual_harms(phasing,field='vacuum'))

    fig_harms, ax_harms = pt.subplots(nrows = 2, sharex = True)
    min_time = 1600; max_time = 2200
    min_shot_time = np.min(a.raw_data['shot_time'])
    max_shot_time = np.max(a.raw_data['shot_time'])
    range_shot_time = max_shot_time - min_shot_time
    for i in range(0,tmp_a.shape[0]):
        clr = (a.raw_data['shot_time'][i] - min_shot_time)/float(range_shot_time)
        print clr
        if int(a.raw_data['shot_time'][i])>min_time and int(a.raw_data['shot_time'][i])<max_time:
            ax_harms[0].plot(np.abs(tmp_a[i,:]), color=str(clr), marker = 'o')
            ax_harms[1].plot(np.angle(tmp_a[i,:]), color=str(clr), marker = 'o')
            ax_harms[0].text(tmp_a.shape[1]-1, np.abs(tmp_a[i,-1]), str(a.raw_data['shot_time'][i])+'plasma')
            ax_harms[0].plot(np.abs(tmp_b[i,:]), color=str(clr), marker = '.')
            ax_harms[1].plot(np.angle(tmp_b[i,:]), color=str(clr), marker = '.')
            ax_harms[0].text(tmp_a.shape[1]-1, np.abs(tmp_b[i,-1]), str(a.raw_data['shot_time'][i])+ 'total')
            ax_harms[0].plot(np.abs(tmp_c[i,:]), color=str(clr))
            ax_harms[1].plot(np.angle(tmp_c[i,:]), color=str(clr))
            ax_harms[0].text(tmp_a.shape[1]-1, np.abs(tmp_b[i,-1]), str(a.raw_data['shot_time'][i]))
    ax_harms[0].set_title(V_dict[V])
    for i in ax_harms:i.grid(True)
    fig_harms.canvas.draw(); fig_harms.show()

    gen_func.setup_publication_image(fig, height_prop = 1./1.618*4, single_col = True)

    xpoint.plot_single_phasing(phasing, 'shot_time', field = 'plasma',  ax = ax[0], plot_kwargs = {'marker':'x'})
    ax[0].set_ylabel('x-point Disp')
    a.plot_parameters('shot_time', 'ROTE', ax = ax[1], plot_kwargs = {'marker':'x'})
    ax[1].set_ylabel('ROTE')
    dBres.plot_single_phasing(phasing, 'shot_time', field = 'plasma', plot_kwargs = {'marker':'x'}, amplitude = True, ax = ax[2])
    ax[2].set_ylabel('dBres plasma')
    dBres.plot_single_phasing(phasing, 'shot_time', field = 'total', plot_kwargs = {'marker':'x'}, amplitude = True, ax = ax[3])
    ax[3].set_ylabel('dBres total')
    probe.plot_single_phasing(phasing, 'shot_time', field = 'plasma', plot_kwargs = {'marker':'x'}, amplitude = True, ax = ax[4])
    ax[4].set_ylabel('66M plasma')
    dBkink.plot_single_phasing(phasing, 'shot_time', field = 'plasma', plot_kwargs = {'marker':'x'}, amplitude = True, ax = ax[5])
    ax[5].set_ylabel('dBkink plasma')
    a.plot_parameters('shot_time', 'Q95', ax = ax[6], plot_kwargs = {'marker':'x'})
    ax[6].set_ylabel('q95')
    a.plot_parameters('shot_time', 'ETA', ax = ax[7], plot_kwargs = {'marker':'x'})
    ax[7].set_ylabel('ETA')
    ax[0].set_xlim([np.min(a.raw_data['shot_time']),np.max(a.raw_data['shot_time'])])

    for i in ax: 
        vline_times = [1600,1717,2134,1830]
        vline_labels = ['I-coil on', 'CIII image 2', 'CIII image 3', 'plas resp decay']
        for t_tmp, j in zip(vline_times, vline_labels):
            i.axvline(t_tmp)
            i.text(t_tmp,np.mean(i.get_ylim()),j,rotation = 90, verticalalignment='center')
        i.grid(True)

fig.tight_layout(pad = 0.01)
fig.savefig('comparison_oliver_data_allV.pdf')
fig.canvas.draw(); fig.show()

1/0

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
