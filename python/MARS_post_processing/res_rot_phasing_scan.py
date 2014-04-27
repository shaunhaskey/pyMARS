import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_funcs
import copy

ul = True; mars_params = None
file_name = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan/shot_142614_rote_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp/shot_142614_rote_scan_w_damp_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp2/shot_142614_rote_scan_w_damp2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp3/shot_142614_rote_scan_w_damp3_post_processing_PEST.pickle'
file_name='/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30/shot_142614_rote_res_scan_30x30_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_test/shot_142614_rote_res_scan_test_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30_kpar1/shot_142614_rote_res_scan_30x30_kpar1_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'



#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_post_processing_PEST.pickle'; ul = False
#file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_post_processing_PEST.pickle'; ul = False
#file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_ul/shot_142614_rote_res_scan_5x5_kpar1_low_rote_single_phase_ul_post_processing_PEST.pickle'; ul = True
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_15x15_kpar1/shot_142614_rote_res_scan_15x15_kpar1_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_20x20_kpar1_med_rote/shot_142614_rote_res_scan_20x20_kpar1_med_rote_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_100_kpar1/shot_142614_rote_scan_100_kpar1_post_processing_PEST.pickle'

phasing = 0
n = 3
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_dB_kink = 'plasma'
reference_offset = [4,0]
sort_name = 'rote_list'


a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False, ul = ul, mars_params = mars_params)


dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean')
dBkink = dBres_dBkink.dBkink_calculations(a)
probe = dBres_dBkink.magnetic_probe(a,' 66M')
#probe = dBres_dBkink.magnetic_probe(a,'Inner_pol')
probe_r = dBres_dBkink.magnetic_probe(a,'UISL')
#probe_r = dBres_dBkink.magnetic_probe(a,'Inner_rad')

#1/0
rot_pts = None
res_pts = None
time_pts = None

expt_file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V3/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V3_post_processing_PEST.pickle'
expt = dBres_dBkink.post_processing_results(expt_file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False, ul = ul, mars_params = mars_params)
time_pts = expt.raw_data['shot_time']
eta_pts = expt.raw_data['ETA']
rote_pts = expt.raw_data['ROTE']
vtor0_pts = expt.raw_data['vtor0']

vals_to_plot = [[0.085, 2.3357214690901212e-07],[0.0088586679041008226, 2.3357214690901212e-07],[0.0013, 2.3357214690901212e-07]]

#1/0
time_plt_pts = [1415, 1735, 2135]
time_plt_pts = [1735, 1895, 2135]

fig, ax = pt.subplots(nrows = 3)#, sharex = True)
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*2.25, single_col = True)
for i in ax: gen_funcs.setup_axis_publication(i, n_xticks = 5, n_yticks = 5)
clim_kink = [0,0.5]
#clim_kink = [0,2.5]
clim_res = [0,2.0]
kink_field = 'plasma'
#kink_field = 'total'
res_field = 'total'
cax_kink = dBkink.plot_phasing_scan('ROTE', xaxis_log = True, field = kink_field, ax = ax[0], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True, clim = clim_kink)
cax_res = dBres.plot_phasing_scan('ROTE', xaxis_log = True, field = res_field, ax = ax[1], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True, clim = clim_res)
#xpoint = dBres_dBkink.x_point_displacement_calcs(a, 0)
#xpoint.plot_phasing_scan('ROTE',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, field = 'plasma', ax = ax[2], n_contours = 15, contour_kwargs = {'colors':'w'})
ax[1].set_xlabel('$\omega_0$')
cbar = pt.colorbar(cax_kink, cax = gen_funcs.create_cbar_ax(ax[0]), ticks = np.linspace(cax_kink.get_clim()[0], cax_kink.get_clim()[1], 5))
#cbar = pt.colorbar(cax_kink, cax = gen_funcs.create_cbar_ax(ax[0]))
gen_funcs.cbar_ticks(cbar)
#cbar.set_label('$\delta B_{kink}$ '+kink_field)
cbar.set_label('$\delta B_{\mathrm{RFA}}$ (G/kA)')

cbar = pt.colorbar(cax_res, cax = gen_funcs.create_cbar_ax(ax[1]), ticks = np.linspace(cax_res.get_clim()[0], cax_res.get_clim()[1], 5))
#gen_funcs.cbar_ticks(cbar)
cbar.set_label('$\delta B_{res}^{\mathrm{' + res_field + '}}$ (G/kA)')
#gs.tight_layout(fig, pad = 0.1)
#xlim = [3.e3,3.e5]


#gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*1.5, single_col = True)
labels = ['$\delta B_{res}^{total}$', '$\delta B_{\mathrm{RFA}}$']
for i, label in zip([dBres, dBkink], labels):
    min_loc = np.argmin(np.abs((i.cur_phasing_scan_x[0,:]) - 0.01))
    rote_val = i.cur_phasing_scan_x[0,min_loc]
    print 'max phasing : ', i.cur_phasing_scan_y[np.argmax(i.cur_phasing_scan_z[:,min_loc]),0]
    ax[2].axvline(i.cur_phasing_scan_y[np.argmax(i.cur_phasing_scan_z[:,min_loc]),0],color='k')
    ax[2].plot(i.cur_phasing_scan_y[:,min_loc], i.cur_phasing_scan_z[:,min_loc], label = label)
for i in [0,90,180,270]: ax[2].axvline(i, linestyle='--')
ax[2].grid(True)
ax[2].legend(loc='best', fontsize = 7)
ax[2].set_xlabel('$\Delta \phi_{ul}$')
ax[2].set_ylabel('Amplitude (G/kA)')
ax[2].set_xlim([-10,360])

xlim = [1.e-3,1.e-1]
for i in [ax[0],ax[1]]:
    i.set_xlim(xlim)
    i.set_ylim([0,360])
    i.axvline(rote_val)
    i.set_ylabel('$\Delta \phi_{ul}$ (deg)')
fig.tight_layout(pad = 0.1)
fig.savefig('res_kink_phasing_rote.pdf')
fig.savefig('res_kink_phasing_rote.eps')
fig.savefig('res_kink_phasing_rote.svg')
fig.canvas.draw(); fig.show()

#self.cur_phasing_scan_z = np.abs(output_array)
#self.cur_phasing_scan_x = rel_axis_grid
#self.cur_phasing_scan_y = phase_grid

#fig, ax = pt.subplots()
#fig.canvas.draw(); fig.show()
