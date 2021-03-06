import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_funcs
import copy

ul = True
file_name = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan/shot_142614_rote_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp/shot_142614_rote_scan_w_damp_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp2/shot_142614_rote_scan_w_damp2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp3/shot_142614_rote_scan_w_damp3_post_processing_PEST.pickle'
file_name='/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30/shot_142614_rote_res_scan_30x30_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_test/shot_142614_rote_res_scan_test_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30_kpar1/shot_142614_rote_res_scan_30x30_kpar1_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/shot_146382_rote_res_scan_15x15_kpar1_med_rote/shot_146382_rote_res_scan_15x15_kpar1_med_rote_post_processing_PEST.pickle'


file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_med_rote/shot_142614_rote_res_scan_20x20_kpar1_med_rote_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_post_processing_PEST.pickle'; ul = False
#file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_post_processing_PEST.pickle'; ul = False
#file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_ul/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_ul_post_processing_PEST.pickle'; ul = True
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_15x15_kpar1/shot_142614_rote_res_scan_15x15_kpar1_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/shot_142614_expt_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/shot_142614_expt_scan_post_processing_PEST.pickle'

phasing = 0
n = 3
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_dB_kink = 'plasma'
reference_offset = [2,0]
sort_name = 'rote_list'


a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False, ul = ul)

#fig1 in the paper?
#a.plot_single_PEST(['ROTE', 'ETA'], [[5e-3,1e-7]], savefig_fname = '/u/haskeysr/vacuum_plasma_total', clim = [0,1.5], phasing = 270)
vals_to_plot = [[0.10000000000000001, 2.3357214690901212e-07],[0.088586679041008226, 2.3357214690901212e-07],[0.00078475997035146064, 2.3357214690901212e-07]]
vals_to_plot = [[0.10000000000000001, 2.3357214690901212e-07],[0.01, 2.3357214690901212e-07],[0.002, 2.3357214690901212e-07]]

vals_to_plot = [[0.10000000000000001, 2.3357214690901212e-08],[0.01, 2.3357214690901212e-08],[0.002, 2.3357214690901212e-08]]

vals_to_plot = [[0.10000000000000001, 5.e-08],[0.01, 5.e-08],[0.002, 5.e-08]]

valid_keys, actual_values = a.find_relevant_keys(['ROTE','ETA'], [vals_to_plot[2]])
combined, results_dict = a.combine_PEST_Vn(valid_keys[0], 0, 'plasma', get_disp = True, return_all_three = True)
grid_r = combined.R*combined.R0EXP
grid_z = combined.Z*combined.R0EXP
plas_r = grid_r[0:combined.Vn.shape[0],:]
plas_z = grid_z[0:combined.Vn.shape[0],:]
#ax.plot(plas_r[i,:],plas_z[i,:],'-')
disp_quant = combined.Vn[-1,:]
fig,ax = pt.subplots()
ax.plot(np.abs(disp_quant))
fig.canvas.draw(); fig.show()
r_vals = plas_r[-1,:]
z_vals = plas_z[-1,:]
dl = np.sqrt(np.diff(z_vals)**2 + np.diff(r_vals)**2)

a.plot_PEST_scan(['ROTE', 'ETA'], vals_to_plot, savefig_fname = '/u/haskeysr/rote_scan_harms_disp', clim = [0,1.35], inc_Bn=False)

1/0
a.plot_multiple_phasings(['ROTE', 'ETA'], [vals_to_plot[0]], savefig_fname = '/u/haskeysr/vac_phasing', clim = [0,2.0], phasing = [0, 90,180,270], field = 'vac')

for i in range(1):
    a.plot_single_PEST(['ROTE', 'ETA'], [vals_to_plot[0], vals_to_plot[2]], savefig_fname = '/u/haskeysr/vacuum_plasma_total2', clim = [0,1.35], phasing = 0)

a.plot_single_PEST(['ROTE', 'ETA'], [vals_to_plot[0], vals_to_plot[2]], savefig_fname = '/u/haskeysr/vacuum_plasma_total2', clim = [0,1.35], phasing = 0)


#fig2 in the paper at the moment
a.plot_single_displacement(['ROTE', 'ETA'], [vals_to_plot[2]], savefig_fname = '/u/haskeysr/displacement_plot', aspect=True, include_mag = False)
#a.plot_single_displacement(['ROTE', 'ETA'], [[5e-3,5.5e-7]], savefig_fname = '/u/haskeysr/displacement_plot', aspect=True, include_mag = False)


1/0
1/0
#works with the flat resistivity profiles from the early scans
a.plot_single_PEST(['ROTE', 'ETA'], [[5e-4,1e-6]], savefig_fname = '/u/haskeysr/vacuum_plasma_total', clim = [0,1.5], phasing = 270)
a.plot_single_PEST(['ROTE', 'ETA'], [[5e-4,1e-6],[5e-2,1e-6]], savefig_fname = '/u/haskeysr/vacuum_plasma_total2', clim = [0,1.5], phasing = 270)

#works with the flat resistivity profiles from the early scans
a.plot_single_PEST(['ROTE', 'ETA'], [[3.e-4,3.e-8],[2.e-2,3.e-8]], savefig_fname = '/u/haskeysr/vacuum_plasma_total2', clim = [0,1.5], phasing = 270)



a.plot_single_PEST(['ROTE', 'ETA'], [[3.e-4,3.e-8]], savefig_fname = '/u/haskeysr/vacuum_plasma_total2', clim = [0,1.5], phasing = 270)

#Works with the flat resistivity profiles
a.plot_PEST_scan(['ROTE', 'ETA'], [[1e-3,5.5e-8],[1e-4,5.5e-8],[1e-5,5.5e-8]], savefig_fname = '/u/haskeysr/rote_scan_harms_disp', clim = [0,1.5])

a.plot_PEST_scan(['ROTE', 'ETA'], [[1e-1,2.335e-7],[1e-2,2.335e-7],[1e-3,2.335e-7]], savefig_fname = '/u/haskeysr/rote_scan_harms_disp', clim = [0,1.5])

1/0




dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean')
dBkink = dBres_dBkink.dBkink_calculations(a)
probe = dBres_dBkink.magnetic_probe(a,' 66M')
#probe = dBres_dBkink.magnetic_probe(a,'Inner_pol')
probe_r = dBres_dBkink.magnetic_probe(a,'UISL')
#probe_r = dBres_dBkink.magnetic_probe(a,'Inner_rad')



# import matplotlib.gridspec as gridspec
# gs = gridspec.GridSpec(2, 1)#, width_ratios=[7,1])
# fig = pt.figure()

# ax = [pt.subplot(gs[0])]
# ax.append(pt.subplot(gs[1], sharex = ax[0]))

#cbar_ax = [pt.subplot(gs[0,1]), pt.subplot(gs[1,1])]

fig, ax = pt.subplots(nrows = 2, sharex = True)
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*1.5, single_col = True)
for i in ax: gen_funcs.setup_axis_publication(i, n_xticks = 5, n_yticks = 5)
cax_kink = dBkink.plot_phasing_scan('ROTE',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, ax = ax[0], n_contours = 10, contour_kwargs = {'colors':'w'})
cax_res = dBres.plot_phasing_scan('ROTE',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, field = 'total', ax = ax[1], n_contours = 10, contour_kwargs = {'colors':'w'})
#xpoint = dBres_dBkink.x_point_displacement_calcs(a, 0)
#xpoint.plot_phasing_scan('ROTE',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, field = 'plasma', ax = ax[2], n_contours = 15, contour_kwargs = {'colors':'w'})
for i in ax: i.set_ylabel('$\Delta \phi_{ul}$ (deg)')
ax[-1].set_xlabel('ROTE')
cbar = pt.colorbar(cax_kink, cax = gen_funcs.create_cbar_ax(ax[0]), ticks = np.linspace(cax_kink.get_clim()[0], cax_kink.get_clim()[1], 5))
#cbar = pt.colorbar(cax_kink, cax = gen_funcs.create_cbar_ax(ax[0]))
gen_funcs.cbar_ticks(cbar)
cbar.set_label('$\delta B_{kink}$ total')

cbar = pt.colorbar(cax_res, cax = gen_funcs.create_cbar_ax(ax[1]), ticks = np.linspace(cax_res.get_clim()[0], cax_res.get_clim()[1], 5))
#gen_funcs.cbar_ticks(cbar)
cbar.set_label('$\delta B_{res}$ total')
#gs.tight_layout(fig, pad = 0.1)
fig.tight_layout(pad = 0.1)
fig.savefig('res_kink_phasing_rote.pdf')
fig.savefig('res_kink_phasing_rote.eps')
fig.canvas.draw(); fig.show()

