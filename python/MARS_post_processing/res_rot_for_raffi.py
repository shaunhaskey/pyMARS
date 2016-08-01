import os
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

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_20x20_kpar1_med_rote_wide_res/shot_142614_rote_res_scan_20x20_kpar1_med_rote_wide_res_post_processing_PEST.pickle'


file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_res_rot/shot158115_04780_res_rot_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04702_n1_res_rot/shot158115_04702_n1_res_rot_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04702_n2_res_rot/shot158115_04702_n2_res_rot_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_25x20_kpar1_med_rote/shot_142614_rote_res_scan_25x20_kpar1_med_rote_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_100_kpar1/shot_142614_rote_scan_100_kpar1_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_146382_rote_res_scan_15x15_kpar1_med_rote/shot_146382_rote_res_scan_15x15_kpar1_med_rote_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/project1_redone/project1_redone_post_processing_PEST.pickle'; mars_params = []
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/shot_142614_expt_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/shot_142614_expt_scan_post_processing_PEST.pickle'


phasing = 0
n = 2
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_dB_kink = 'plasma'
reference_offset = [4,0]
sort_name = 'rote_list'


a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic=fixed_harmonic, reference_offset=reference_offset, reference_dB_kink=reference_dB_kink, sort_name=sort_name, try_many_phasings=False, ul=ul, mars_params=mars_params)


dBres = dBres_dBkink.dBres_calculations(a, mean_sum='mean')
dBkink = dBres_dBkink.dBkink_calculations(a)
#probe = dBres_dBkink.magnetic_probe(a,' 66M')
probe = dBres_dBkink.magnetic_probe(a,'66M')
probe2 = dBres_dBkink.magnetic_probe(a,'MPID1A')
#probe = dBres_dBkink.magnetic_probe(a,'Inner_pol')
probe_r = dBres_dBkink.magnetic_probe(a,'UISL')
#probe_r = dBres_dBkink.magnetic_probe(a,'Inner_rad')

#1/0
rot_pts = None
res_pts = None
time_pts = None

expt_file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V3/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V3_post_processing_PEST.pickle'
expt_file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_dif_eq_const_rot_prof_spitz/shot_142614_expt_scan_NC_dif_eq_const_rot_prof_spitz_post_processing_PEST.pickle'
expt = dBres_dBkink.post_processing_results(expt_file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic=fixed_harmonic, reference_offset=reference_offset, reference_dB_kink=reference_dB_kink, sort_name=sort_name, try_many_phasings=False, ul=ul, mars_params=mars_params)
time_pts = expt.raw_data['shot_time']
eta_pts = expt.raw_data['ETA']
rote_pts = expt.raw_data['ROTE']
vtor0_pts = expt.raw_data['vtor0']


vals_to_plot = [[0.10000000000000001, 2.3357214690901212e-07],[0.0088586679041008226, 2.3357214690901212e-07],[0.0020691380811147901, 2.3357214690901212e-07]]

vals_to_plot = [[0.10000000000000001, 5.5e-08],[0.0088586679041008226, 5.5e-08],[0.0020691380811147901, 5.5e-08]]

#vals_to_plot = [[0.085, 2.3357214690901212e-07],[0.0088586679041008226, 2.3357214690901212e-07],[0.0013, 2.3357214690901212e-07]]

#1/0
time_plt_pts = [1415, 1735, 2135]
time_plt_pts = [1735, 1895, 2135]
detailed_phasing = False
if detailed_phasing:
    fig, ax = pt.subplots(nrows = 2, sharex = True)
    gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*1.5, single_col = True)
    for i in ax: gen_funcs.setup_axis_publication(i, n_xticks = 5, n_yticks = 5)
    #cax_kink = dBkink.plot_phasing_scan('ROTE',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, ax = ax[0], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True)
    #cax_res = dBres.plot_phasing_scan('ROTE',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, field = 'total', ax = ax[1], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True)
    #cax_kink = dBkink.plot_phasing_scan('vtor0',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, ax = ax[0], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True)
    #cax_res = dBres.plot_phasing_scan('vtor0',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, field = 'total', ax = ax[1], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True)
    clim_kink = [0,1.25]
    clim_res = [0,2.0]
    kink_field = 'plasma'
    res_field = 'total'
    cax_kink = dBkink.plot_phasing_scan('vtor0',filter_names = ['ETA'], filter_values = [3.7926901907322539e-07], xaxis_log = True, field = kink_field, ax = ax[0], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True, clim = clim_kink)
    cax_res = dBres.plot_phasing_scan('vtor0',filter_names = ['ETA'], filter_values = [3.7926901907322539e-07], xaxis_log = True, field = res_field, ax = ax[1], n_contours = 10, contour_kwargs = {'colors':'w'}, plot_ridge = True, clim = clim_res)
    #xpoint = dBres_dBkink.x_point_displacement_calcs(a, 0)
    #xpoint.plot_phasing_scan('ROTE',filter_names = ['ETA'], filter_values = [1.1288378916846883e-06], xaxis_log = True, field = 'plasma', ax = ax[2], n_contours = 15, contour_kwargs = {'colors':'w'})
    for i in ax: i.set_ylabel('$\Delta \phi_{ul}$ (deg)')
    ax[-1].set_xlabel('$\omega_0$ (rad/s)')
    cbar = pt.colorbar(cax_kink, cax = gen_funcs.create_cbar_ax(ax[0]), ticks = np.linspace(cax_kink.get_clim()[0], cax_kink.get_clim()[1], 5))
    #cbar = pt.colorbar(cax_kink, cax = gen_funcs.create_cbar_ax(ax[0]))
    gen_funcs.cbar_ticks(cbar)
    cbar.set_label('$\delta B_{RFA}$ '+kink_field)

    cbar = pt.colorbar(cax_res, cax = gen_funcs.create_cbar_ax(ax[1]), ticks = np.linspace(cax_res.get_clim()[0], cax_res.get_clim()[1], 5))
    #gen_funcs.cbar_ticks(cbar)
    cbar.set_label('$\delta B_{res}$ ' + res_field)
    #gs.tight_layout(fig, pad = 0.1)
    xlim = [3.e3,3.e5]
    ax[-1].set_xlim(xlim)
    fig.tight_layout(pad = 0.1)
    fig.savefig('res_kink_phasing_rote.pdf')
    fig.savefig('res_kink_phasing_rote.eps')
    fig.canvas.draw(); fig.show()


phasings_disp = [0,45,90,135,180,225,270,315]
phasings_disp = [0,90,180,270]
if not ul: phasings_disp = [0]
x_axis = 'ROTE'
#x_axis = 'vtor0'
y_axis = 'ETA'

if x_axis=='ROTE' :
    expt_pts_rot = rote_pts 
elif x_axis=='vtor0':
    expt_pts_rot = vtor0_pts
else:
    raise ValueError()

if y_axis=='ETA' :
    expt_pts_res = eta_pts
else:
    raise ValueError()

#x_axis_label = '$\omega$ (rad/s)'
x_axis_label = '$\omega$ (rad/s)' if x_axis == 'vtor0' else '$\omega_0$'
y_axis_label = '$\eta_0$'
fields = ['total','plasma','plasma', 'plasma']
ylim = [1.e-8,1.e-6]
ylim = [5.e-9,5.e-7]
xlim = [1.e-6,1.e-2]
xlim = None
xlim = [1.e2,3.e5]
xlim = [3.e3,3.e5] if x_axis == 'vtor0' else [1.e-3, 0.12]#[0.0012239067526504149, 0.1]

vlines = None
#vlines = [60000,15000]
hlines = None
#hlines = [3.5e-7,1.75e-8]
labels = ['$\delta B_{res}^{tot}$ (G/kA)', '$\delta B_{RFA}$ (G/kA)', 'Midplane outboard poloidal probe (G/kA)', 'HFS poloidal probe (G/kA)']
fnames = ['dBres','dBkink','probe','probe_phase']
clims = [[0,2.5],[0,2],[0,1.5]]
clims = [[0,5],[0,5],[0,15]]
clims = [[0,1.5],[0,0.8],[0,1.05]]
clims = [[0,1.5],[0,0.4],[0,1.05],[-np.pi,np.pi]]
clims = [[0,0.5],[0,1.5],[0,1.05],[-np.pi,np.pi]]
clims = [[0,1.5],[0,3],[0,2.5],[0,2.5]]
#clims = [[0,1.25],[0,0.3],[0,1.05]]
#clims = [[-np.pi, np.pi],[-np.pi,np.pi],[-np.pi,np.pi]]
data =  [dBres, dBkink, probe, probe2]
cmaps =  ['jet', 'jet', 'jet', 'jet']
funcs = [np.abs, np.abs, np.abs, np.abs]
amplitudes = [True, True, True, True]
for calc_type, cur_clim, title, fname, field, cmap, amp in zip(data, clims, labels, fnames, fields, cmaps, amplitudes):
    replacement_kwargs = {'xtick.labelsize': 7.0,'ytick.labelsize': 7.0}
    if len(phasings_disp)==8:
        fig, ax = pt.subplots(nrows = 2, ncols = 4, sharex = True, sharey = True)
        gen_funcs.setup_publication_image(fig, height_prop = 1./2.0, single_col = False, replacement_kwargs = replacement_kwargs)
    else:
        fig, ax = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
        gen_funcs.setup_publication_image(fig, height_prop = 1./1.1, single_col = True, replacement_kwargs = replacement_kwargs)
    #gen_funcs.setup_publication_image(fig, height_prop = 1./1.618, single_col = False)
    for i, cur_ax in zip(phasings_disp, ax.flatten()):
        cax = calc_type.plot_2D(i,x_axis,y_axis,cmap_res = cmap, field = field, clim = cur_clim, ax = cur_ax, med_filt_value = 1, n_contours = 15, contour_kwargs = {'colors':'k', 'linewidths':0.5}, amplitude = amp,plot_dots = True)
        cur_ax.set_title('$\Delta \phi = {}^o$'.format(i))
        if vlines!=None: 
            for vert_line in vlines: cur_ax.axvline(x=vert_line, color='k')
        if hlines!=None: 
            for hor_line in hlines: cur_ax.axhline(y=hor_line, color='k')
        if time_plt_pts!=None and i == 0:
            #cur_ax.plot(expt_pts_rot, expt_pts_res, 'k.')
            #cur_ax.plot(expt_pts_rot, np.array(expt_pts_res)/10, 'k.')
            for t_tmp in time_plt_pts:  
                cur_ax.plot(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)], 'k.')
                cur_ax.text(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)], t_tmp, fontsize=6)
                if i==0: cur_ax.plot([tmp_ind[0] for tmp_ind in vals_to_plot], [tmp_ind[1] for tmp_ind in vals_to_plot], 'kd')
            #for t_tmp in time_plt_pts: 
            #    cur_ax.plot(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)]/10, 'k.')
            #    cur_ax.text(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)]/10, t_tmp, fontsize=6)

            #for rot_tmp, res_tmp, t_tmp in zip(expt_pts_rot, expt_pts_res, time_pts): cur_ax.text(rot_tmp, res_tmp, t_tmp, fontsize=7)
            #for rot_tmp, res_tmp, t_tmp in zip(expt_pts_rot, expt_pts_res, time_pts): cur_ax.text(rot_tmp, res_tmp/10, t_tmp, fontsize=7)
    for i in ax[:,0]: i.set_ylabel(y_axis_label)
    for i in ax[-1,:]: i.set_xlabel(x_axis_label)
    if xlim!=None: ax[0,0].set_xlim(xlim)
    if ylim!=None: ax[0,0].set_ylim(ylim)
    fig.tight_layout(pad = 0.1)
    cbar = pt.colorbar(cax, ax = ax.flatten().tolist())
    #cbar.set_label('{}-{}'.format(title, field))
    cbar.set_label('{}'.format(title))
    #fig.savefig(fname+'-'+field+'2x2.pdf', bbox_inches = 'tight',pad = 0.1)
    #fig.savefig(fname+'-'+field+'2x2.svg', bbox_inches = 'tight',pad = 0.1)
    #fig.savefig(fname+'-'+field+'2x2.pdf', pad = 0.1)
    #fig.savefig(fname+'-'+field+'2x2.svg', pad = 0.1)
    fig.canvas.draw(); fig.show()

for phasing_val in range(0,360,60):
    res_val = 8.88e-9
    fig, ax = pt.subplots(nrows = 3, sharex = True)
    x_vals = 10**(np.linspace(-3,-1,1000))
    #dBres.plot_slice_through_2D_data(0, x_axis, y_axis, 1.e-8, np.linspace(1.e-3,2.e-1,1000), field='plasma', ax=ax, plot_kwargs=None, amplitude=True, yaxis_log=True, xaxis_log=True)
    for i in ['66M', 'MPID1A', 'MPID1B']:
        tmp_probe =  dBres_dBkink.magnetic_probe(a,i)
        dat = tmp_probe.plot_slice_through_2D_data(phasing_val, x_axis, y_axis, res_val, x_vals, field='plasma', ax=ax[0], plot_kwargs=None, amplitude=True, yaxis_log=False, xaxis_log=True, plot_data = False)
        ax[0].plot(x_vals, dat, '-', label = tmp_probe.probe, marker = '.')
        ax[1].plot(x_vals, (dat/dat[-1]-1)*100, '-', label = tmp_probe.probe, marker = '.')
    #dat = probe2.plot_slice_through_2D_data(phasing_val, x_axis, y_axis, res_val, x_vals, field='plasma', ax=ax[0], plot_kwargs=None, amplitude=True, yaxis_log=False, xaxis_log=True, plot_data = False)
    #ax[0].plot(x_vals, dat/dat[-1], '-', label = probe2.probe, marker = '.')
    #dat = dBres.plot_slice_through_2D_data(phasing_val, x_axis, y_axis, res_val, x_vals, field='plasma', ax=ax[1], plot_kwargs=None, amplitude=True, yaxis_log=False, xaxis_log=True, plot_data = False)
    dat = dBres.plot_slice_through_2D_data(phasing_val, x_axis, y_axis, res_val, x_vals, field='total', ax=ax[1], plot_kwargs=None, amplitude=True, yaxis_log=False, xaxis_log=True, plot_data = False)
    ax[2].plot(x_vals, dat, '-', label='res metric total', marker = '.')
    dat = dBres.plot_slice_through_2D_data(phasing_val, x_axis, y_axis, res_val, x_vals, field='vacuum', ax=ax[1], plot_kwargs=None, amplitude=True, yaxis_log=False, xaxis_log=True, plot_data = False)
    ax[2].plot(x_vals, dat, '-', label='res metric vac', marker = '.')
    for i in ax: i.set_xscale('log')
    ax[0].set_xlim([x_vals[0],x_vals[-1]])
    for ax_tmp, lab in zip(ax,['Probe output G/kA', '% increase', 'G/kA']): ax_tmp.set_ylabel(lab)
    #ax[0].set_ylim([1,ax[0].get_ylim()[1]])
    ax[-1].set_xlabel('rotation vrot/valfven')
    ax[0].set_title('n={}, ul phasing={}deg, resistity~={}Ohm-m'.format(tmp_probe.parent.n, phasing_val, res_val))
    for i in ax: i.legend(loc = 'best')
    for i in ax: i.
    fig.canvas.draw(); fig.show()


#########22nd Oct rotation scan########
phasing = 60
#for phasing in range(0,360,60):
for phasing in [0,180]:
    fig, ax_tot = pt.subplots(nrows=3, ncols=2, sharex=True)
    for j, n in enumerate([1,2]):
        ax = ax_tot[:,j]
        name = 'shot158115_04702_n{}_rot_scan'.format(n)
        #name = 'shot158115_04702_n1_rot_scan'
        file_name = '{}/{}/{}_post_processing_PEST.pickle'.format(os.environ['mars_data'], name, name)
        phase_machine_ntor = 0
        s_surface = 0.92
        fixed_harmonic = 3
        reference_dB_kink = 'plas'
        reference_dB_kink = 'plasma'
        reference_offset = [4,0]
        sort_name = 'rote_list'

        a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic=fixed_harmonic, reference_offset=reference_offset, reference_dB_kink=reference_dB_kink, sort_name=sort_name, try_many_phasings=False, ul=ul, mars_params=mars_params)


        dBres = dBres_dBkink.dBres_calculations(a, mean_sum='mean')
        dBkink = dBres_dBkink.dBkink_calculations(a)
        #probe = dBres_dBkink.magnetic_probe(a,' 66M')
        probe = dBres_dBkink.magnetic_probe(a,'66M')
        probe2 = dBres_dBkink.magnetic_probe(a,'MPID1A')
        #probe = dBres_dBkink.magnetic_probe(a,'Inner_pol')
        probe_r = dBres_dBkink.magnetic_probe(a,'UISL')
        #probe_r = dBres_dBkink.magnetic_probe(a,'Inner_rad')
        for probe_name in ['66M','MPID1A', 'MPID1B']:
            tmp_probe = dBres_dBkink.magnetic_probe(a,probe_name)
            x, y = tmp_probe.plot_single_phasing(phasing,'ROTE',field='plasma',plot_data = False, ax = None)
            filt = x<0.1
            x = x[filt]; y = y[filt]
            ax[0].plot(x, y, '-', marker='.', label=probe_name)
            ax[1].plot(x, (y/y[-1]-1)*100, '-', marker='.', label=probe_name)
        x, y = dBres.plot_single_phasing(phasing,'ROTE',field='total',plot_data = False, ax = None)
        filt = x<0.1
        x = x[filt]; y = y[filt]
        ax[2].plot(x, (y/y[-1]-1)*100, '-', marker='.',label='pitch-res metric')
        x, y = dBkink.plot_single_phasing(phasing,'ROTE',field='total',plot_data = False, ax = None)
        filt = x<0.1
        x = x[filt]; y = y[filt]
        ax[2].plot(x, (y/y[-1]-1)*100, '-', marker='.',label='RFA metric')
        ax[0].set_title('n={}'.format(probe.parent.n))
    for i in ax_tot.flatten(): 
        i.legend(loc = 'best',fontsize='small')
        i.grid(True,which='both')
    ax_tot[0,0].set_ylabel('Probe output G/kA')
    ax_tot[1,0].set_ylabel('Output increase (%)')
    ax_tot[2,0].set_ylabel('Output increase (%)')
    ax_tot[0,0].set_xlim([1.e-3,1e-1])
    for i in ax_tot[-1,:]:i.set_xlabel('Vtor/Va')
    for i in ax_tot.flatten(): i.set_xscale('log')
    fig.suptitle('phasing:{}deg'.format(phasing))
    fig.canvas.draw(); fig.show()




#########22nd Oct rotation scan########
phasing = 60
#for phasing in range(0,360,60):
fig, ax_tot = pt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
for j, n in enumerate([1,2]):
    ax = ax_tot[:,j]
    name = 'shot158115_04702_n{}_rot_scan'.format(n)
    #name = 'shot158115_04702_n1_rot_scan'
    file_name = '{}/{}/{}_post_processing_PEST.pickle'.format(os.environ['mars_data'], name, name)
    phase_machine_ntor = 0
    s_surface = 0.92
    fixed_harmonic = 3
    reference_dB_kink = 'plas'
    reference_dB_kink = 'plasma'
    reference_offset = [4,0]
    sort_name = 'rote_list'

    a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic=fixed_harmonic, reference_offset=reference_offset, reference_dB_kink=reference_dB_kink, sort_name=sort_name, try_many_phasings=False, ul=ul, mars_params=mars_params)
    dBres = dBres_dBkink.dBres_calculations(a, mean_sum='mean')
    dBkink = dBres_dBkink.dBkink_calculations(a)
    M66M = dBres_dBkink.magnetic_probe(a,'66M')
    MPID1A = dBres_dBkink.magnetic_probe(a,'MPID1A')
    for phasing in range(0,360):
        #for probe_name in ['66M','MPID1A', 'MPID1B']:
        #    tmp_probe = dBres_dBkink.magnetic_probe(a,probe_name)
        M66M_x, M66M_y = M66M.plot_single_phasing(phasing,'ROTE',field='plasma',plot_data = False, ax = None)
        MPID1A_x, MPID1A_y = MPID1A.plot_single_phasing(phasing,'ROTE',field='plasma',plot_data = False, ax = None)
        dBres_x, dBres_y = dBres.plot_single_phasing(phasing,'ROTE',field='total',plot_data = False, ax = None)
        dBkink_x, dBkink_y = dBkink.plot_single_phasing(phasing,'ROTE',field='total',plot_data = False, ax = None)
        val = np.array(dBres.parent.raw_data['ROTE'])>0.008
        ax[0].plot(np.abs(dBkink_y)[val], np.abs(M66M_y)[val],'r.',alpha=0.1)
        ax[1].plot(np.abs(dBkink_y)[val], np.abs(MPID1A_y)[val],'b.',alpha=0.1)
        ax[2].plot(np.abs(dBres_y)[val], np.abs(M66M_y)[val],'r.',alpha=0.1)
        ax[3].plot(np.abs(dBres_y)[val], np.abs(MPID1A_y)[val],'b.',alpha=0.1)
        #filt = x<0.1
        #x = x[filt]; y = y[filt]
        #ax[2].plot(x, (y/y[-1]-1)*100, '-', marker='.',label='RFA metric')
        #ax[0].set_title('n={}'.format(probe.parent.n))
    #for i in ax_tot.flatten(): 
    #    i.legend(loc = 'best',fontsize='small')
    #    i.grid(True,which='both')
    # ax_tot[0,0].set_ylabel('Probe output G/kA')
    # ax_tot[1,0].set_ylabel('Output increase (%)')
    # ax_tot[2,0].set_ylabel('Output increase (%)')
    # ax_tot[0,0].set_xlim([1.e-3,1e-1])
    # for i in ax_tot[-1,:]:i.set_xlabel('Vtor/Va')
    # for i in ax_tot.flatten(): i.set_xscale('log')
    # fig.suptitle('phasing:{}deg'.format(phasing))
fig.canvas.draw(); fig.show()


phasing = 60
#for phasing in range(0,360,30):
for phasing in [0,180]:
    m_list = [[3,4,5],[8,9,10]]
    fig_res, ax_res = pt.subplots(nrows=2, sharex=True)
    for j, (n,plot_m) in enumerate(zip([1,2],m_list)):

        #ax = ax_tot[:,j]
        name = 'shot158115_04702_n{}_rot_scan'.format(n)
        #name = 'shot158115_04702_n1_rot_scan'
        file_name = '{}/{}/{}_post_processing_PEST.pickle'.format(os.environ['mars_data'], name, name)
        phase_machine_ntor = 0
        s_surface = 0.92
        fixed_harmonic = 3
        reference_dB_kink = 'plas'
        reference_dB_kink = 'plasma'
        reference_offset = [4,0]
        sort_name = 'rote_list'

        a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic=fixed_harmonic, reference_offset=reference_offset, reference_dB_kink=reference_dB_kink, sort_name=sort_name, try_many_phasings=False, ul=ul, mars_params=mars_params)


        dBres = dBres_dBkink.dBres_calculations(a, mean_sum='mean')
        dBkink = dBres_dBkink.dBkink_calculations(a)

        z = dBres.single_phasing_individual_harms(phasing, field='total')
        m_vals = dBres.raw_data['res_m_vals']
        m_vals = dBres.raw_data['res_m_vals']
        s_vals = dBres.raw_data['res_s_vals']
        q_vals = dBres.raw_data['res_q_vals']
        rote = np.array(dBres.parent.raw_data['ROTE'])
        rote_mesh = np.tile(rote,(len(m_vals[0]),1)).T
        #fig, ax = pt.subplots()
        #im = ax.pcolormesh(np.array(m_vals),rote_mesh,np.abs(np.array(z)),cmap='spectral')
        #im.set_clim([0,3])
        #ax.set_yscale('log')
        #fig.canvas.draw(); fig.show()

        ax = ax_res[j]
        #fig, ax = pt.subplots()
        #np.argmin(np.abs(rote - 0.008))
        #val_rote = rote[:]
        for m in plot_m:
            m_loc = np.argmin(np.abs(m_vals[0] - m))
            ax.plot(rote,np.abs(np.array(z)[:,m_loc]), marker = '.',label = 'm={},q={},psi={:.2f}'.format(m,q_vals[0][m_loc], s_vals[0][m_loc]**2))
        x, y = dBkink.plot_single_phasing(phasing,'ROTE',field='total',plot_data = False, ax = None)

        ax.plot(x,y, marker = '.',label = 'RFA:m={}'.format(np.mean(dBkink.raw_data['total_max_mode_list_lower'])))
        ax.axvline(1.e-3)
        ax.axvline(1.e-1)
        ax.set_ylabel('G/kA')
        ax.set_title('n={},Pitch resonant harmonics, and RFA harmonic vs rotation'.format(dBres.parent.n))
        ax.legend(loc='best',fontsize='small')
        ax.set_xscale('log')
    for i in ax_res:i.grid(True,which='both')
    ax_res[-1].set_xlabel('Vtor/Va)')
    fig_res.suptitle('upper-lower phasing: {}'.format(phasing))
    fig_res.canvas.draw(); fig_res.show()


for i,m in zip(z,m_vals):
    ax.plot(m, np.abs(i))
fig.canvas.draw(); fig.show()


fig, ax = pt.subplots()
for i,m in zip(z,m_vals):
    ax.plot(m, np.abs(i))
fig.canvas.draw(); fig.show()
#1/0
rot_pts = None
res_pts = None
time_pts = None





if len(phasings_disp)==8:
    fig, ax = pt.subplots(nrows = 2, ncols = 4, sharex = True, sharey = True)
    gen_funcs.setup_publication_image(fig, height_prop = 1./2.0, single_col = False, replacement_kwargs = replacement_kwargs)
else:
    fig, ax = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
    gen_funcs.setup_publication_image(fig, height_prop = 1./1.1, single_col = True, replacement_kwargs = replacement_kwargs)


# x_axis = 'ROTE'
# y_axis = 'ETA'
# x_axis = 'vtor0'


#x_axis_label = '$\omega_{0}$'
#y_axis_label = '$\eta_0$'
label = ' displacement at the x-point (mm/kA)'
title = 'displacement'
field = 'plasma'
#gen_funcs.setup_publication_image(fig, height_prop = 1./1.618, single_col = False)
#ylim = [1.e-8,1.e-6]
#xlim = [1.e-6,1.e-2]

#xlim = None
# vlines = None
# vlines = [60000,15000]
# hlines = None
# hlines = [3.5e-7,1.75e-8]
clim = [0,0.035]
clim = [0,2.25]
#gen_funcs.setup_publication_image(fig, height_prop = 1./2.0, single_col = False)
for i, cur_ax in zip(phasings_disp, ax.flatten()):
    xpoint = dBres_dBkink.x_point_displacement_calcs(a, i)
    #x1000 for mm
    cax = xpoint.plot_2D(i,x_axis,y_axis,cmap_res = 'jet', field = 'plasma', clim = clim, ax = cur_ax, n_contours = 15, contour_kwargs = {'colors':'k', 'linewidths':0.5}, multiplier = 1000.)
    cur_ax.set_title('$\Delta \phi = {}^o$'.format(i))
    if vlines!=None: 
        for vert_line in vlines: cur_ax.axvline(x=vert_line, color='k')
    if hlines!=None: 
        for hor_line in hlines: cur_ax.axhline(y=hor_line, color='k')
    if expt_pts_rot!=None and i==0:
        #cur_ax.plot(expt_pts_rot, expt_pts_res, 'k.')
        #cur_ax.plot(expt_pts_rot, np.array(expt_pts_res)/10, 'k.')
        for t_tmp in time_plt_pts: 
            cur_ax.plot(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)], 'k.')
            cur_ax.text(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)], t_tmp, fontsize=6)
            if i==0: cur_ax.plot([tmp_ind[0] for tmp_ind in vals_to_plot], [tmp_ind[1] for tmp_ind in vals_to_plot], 'kd')
        #for t_tmp in time_plt_pts: 
        #    cur_ax.plot(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)]/10, 'k.')
        #    cur_ax.text(expt_pts_rot[time_pts.index(t_tmp)], expt_pts_res[time_pts.index(t_tmp)]/10, t_tmp, fontsize=6)
        #for rot_tmp, res_tmp, t_tmp in zip(expt_pts_rot, expt_pts_res, time_pts): cur_ax.text(rot_tmp, res_tmp, t_tmp, fontsize=7)
        #for rot_tmp, res_tmp, t_tmp in zip(expt_pts_rot, expt_pts_res, time_pts): cur_ax.text(rot_tmp, res_tmp/10, t_tmp, fontsize=7)

for i in ax[:,0]: i.set_ylabel(y_axis_label)
for i in ax[-1,:]: i.set_xlabel(x_axis_label)
if xlim!=None: ax[0,0].set_xlim(xlim)
if ylim!=None: ax[0,0].set_ylim(ylim)
fig.tight_layout(pad=0.1)
cbar = pt.colorbar(cax, ax=ax.flatten().tolist())
cbar.set_label(label)
#fig.savefig(title+'2x2.pdf', bbox_inches = 'tight',pad = 0.1)
#fig.savefig(title+'2x2.svg', bbox_inches = 'tight',pad = 0.1)
fig.savefig(title + '2x2.pdf', pad=0.1)
fig.savefig(title + '2x2.svg', pad=0.1)
fig.canvas.draw(); fig.show()



1/0
fig,ax = pt.subplots(nrows = 4, sharex = True)
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*3., single_col = True)
x_log = True
x_interp = np.linspace(-5,-2,30)
x_interp = np.linspace(-4,-2,30)
x_interp = 10**x_interp
#x_interp = np.linspace(1.e-6,0.01,100)
eta_list = [5.e-7, 5.5e-8, 1.e-8]
#eta_list = [5.5e-8, 1.e-8]
field = 'plasma'
phasing = 0
for eta, clr in zip(eta_list, ['b','g','r']):
    #for cur_ax, func, title in zip(ax, [dBres, dBkink, probe, probe_r], ['dBres', 'dBkink', 'probe p', 'probe r']):
    for cur_ax, func, title in zip(ax, [dBres, dBkink, probe,], ['$\delta B_{res}$ (G/kA)', '$\delta B_{kink}$ (G/kA)', 'poloidal probe (G/kA)']):
        plot_kwargs = {'marker':'o', 'color':clr}
        func.plot_slice_through_2D_data(phasing, x_axis, y_axis, eta, x_interp, field = 'plasma',  ax = cur_ax, plot_kwargs = plot_kwargs, amplitude = True, yaxis_log = False, xaxis_log = x_log)
        plot_kwargs = {'marker':'x', 'color':clr}
        func.plot_slice_through_2D_data(phasing, x_axis, y_axis, eta, x_interp, field = 'total',  ax = cur_ax, plot_kwargs = plot_kwargs, amplitude = True, yaxis_log = False, xaxis_log = x_log)
        plot_kwargs = {'marker':'+', 'color':clr}
        func.plot_slice_through_2D_data(phasing, x_axis, y_axis, eta, x_interp, field = 'vacuum',  ax = cur_ax, plot_kwargs = plot_kwargs, amplitude = True, yaxis_log = False, xaxis_log = x_log)
        cur_ax.set_ylabel(title)
    xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)
    plot_kwargs = {'marker':'o', 'color':clr}
    xpoint.plot_slice_through_2D_data(phasing, x_axis, y_axis, eta, x_interp, field = 'plasma',  ax = ax[-1], plot_kwargs = plot_kwargs, amplitude = True, yaxis_log = False, xaxis_log = x_log)
    ax[-1].set_ylabel('disp x-point (a.u)')
for i in ax: i.grid()
title ='blue, green, red : $\eta_0$={}\n o: plasma, +: vacuum, x: total'.format(', '.join(['{:.1e}'.format(i) for i in eta_list]))
ax[0].set_title(title)
ax[0].set_xlim([np.min(x_interp), np.max(x_interp)])
ax[-1].set_xlabel('$\omega_0$')
for i in ax: gen_funcs.setup_axis_publication(i, n_xticks = None, n_yticks = 5)
fig.tight_layout(pad = 0.05)
fig.savefig('simulated_scan_{}_{}.pdf'.format(phasing, field))
fig.canvas.draw(); fig.show()

1/0


a = dBres_dBkink.test1(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)



fig, ax = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax = ax.flatten()
fig2, ax2_orig = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax2 = ax2_orig.flatten()
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618, single_col = False)
gen_funcs.setup_publication_image(fig2, height_prop = 1./1.618, single_col = False)

phasings_disp = [0,45,90,135,180,225,270,315]
for i in range(len(phasings_disp)):
    a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = ax[i], ax_matrix = ax2[i], clim = [0, 0.015])
    tmp, color_ax = a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = None, ax_matrix = ax2[i], clim = [0, 0.025])
    #tmp, color_ax = a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = None, ax_matrix = None, clim = [0, 0.025])
fig.tight_layout(pad = 0.5)
fig.canvas.draw(); fig.show()


ax2_orig[0,0].set_xlim([1.e-4,1e-1])
for i in ax2_orig[:,0]:i.set_ylabel('eta')
for i in ax2_orig[-1,:]:i.set_xlabel('rote')
fig2.tight_layout(pad=0.01)
cbar = pt.colorbar(color_ax, ax = ax2.tolist())
cbar.set_label('Displacement around x-point')
fig2.tight_layout(pad = 0.1)
fig2.savefig('res_rot_scan_displacement.pdf')
fig2.canvas.draw(); fig2.show()
a.eta_rote_matrix(phasing = 0, plot_type = 'plas')


1/0
def do_everything(file_name, s_surface, phasing,phase_machine_ntor, fixed_harmonic = 5, reference_offset=[2,0], reference_dB_kink='plas',sort_name = 'q95_list'):
    project_dict = pickle.load(file(file_name,'r'))
    key_list = project_dict['sims'].keys()

    n = np.abs(project_dict['details']['MARS_settings']['<<RNTOR>>'])
    q95_list, Bn_Li_list, time_list = dBres_dBkink.extract_q95_Bn(project_dict, bn_li = 1)
    eta_list, rote_list = dBres_dBkink.extract_eta_rote(project_dict)
    res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, res_tot_list_upper, res_tot_list_lower = dBres_dBkink.extract_dB_res(project_dict, return_total = True)

    amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close = dBres_dBkink.extract_dB_kink(project_dict, s_surface)
    fig_harm_select, ax_harm_select = pt.subplots()
    #Create the fixed phasing cases (as set by phasing)
    amps_vac_comp = dBres_dBkink.apply_phasing(amps_vac_comp_upper, amps_vac_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
    amps_plas_comp = dBres_dBkink.apply_phasing(amps_plas_comp_upper, amps_plas_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
    amps_tot_comp = dBres_dBkink.apply_phasing(amps_tot_comp_upper, amps_tot_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)

    #Get the reference which we use to find the maximum harmonic for dBkink
    if reference_dB_kink=='plas':
        reference = dBres_dBkink.get_reference(amps_plas_comp_upper, amps_plas_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)
    elif reference_dB_kink=='tot':
        reference = dBres_dBkink.get_reference(amps_tot_comp_upper, amps_tot_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)

    #Note the returned values are simply a 1D array containing the complex amplitude of the max harmonic
    #Do it for the single cases
    plot_quantity_vac, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_vac_comp, reference_offset = reference_offset)
    ax_harm_select.plot(eta_list,max_loc_list, label='max-harmonic')
    plot_quantity_plas, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_plas_comp, reference_offset = reference_offset)
    plot_quantity_tot, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_tot_comp, reference_offset = reference_offset)

    #Do it for the upper/lower cases
    upper_values_plasma, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_plas_comp_upper, reference_offset = reference_offset)
    lower_values_plasma, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_plas_comp_lower, reference_offset = reference_offset)
    upper_values_tot, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_tot_comp_upper, reference_offset = reference_offset)
    lower_values_tot, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_tot_comp_lower, reference_offset = reference_offset)
    upper_values_vac, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_vac_comp_upper, reference_offset = reference_offset)
    lower_values_vac, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_vac_comp_lower, reference_offset = reference_offset)

    ax_harm_select.legend(loc='best')
    ax_harm_select.set_xlabel('q95')
    ax_harm_select.set_ylabel('m')
    ax_harm_select.set_title('%s used to select m'%(reference_dB_kink))
    
    ax_harm_select.set_ylim([0,np.max(q_val_list)*n+5])
    fig_harm_select.canvas.draw(); fig_harm_select.show()
    #Calculate fixed harmonic dBkink based only on vacuum fields, again upper_values.... are 1D array containing the complex amplitude of fixed harmonic
    upper_values_vac_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_vac_comp_upper, fixed_harmonic)
    lower_values_vac_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_vac_comp_lower, fixed_harmonic)
    upper_values_plas_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_plas_comp_upper, fixed_harmonic)
    lower_values_plas_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_plas_comp_lower, fixed_harmonic)

    #Convert the complex number into an amplitude and phase
    plot_quantity_vac_phase = np.angle(plot_quantity_vac,deg=True).tolist()
    plot_quantity_plas_phase = np.angle(plot_quantity_plas,deg=True).tolist()
    plot_quantity_tot_phase = np.angle(plot_quantity_tot,deg=True).tolist()
    plot_quantity_vac = np.abs(plot_quantity_vac).tolist()
    plot_quantity_plas = np.abs(plot_quantity_plas).tolist()
    plot_quantity_tot = np.abs(plot_quantity_tot).tolist()

    #create copies before everything is arranged
    q95_list_copy = copy.deepcopy(q95_list)
    Bn_Li_list_copy = copy.deepcopy(Bn_Li_list)

    #create the sorted lists
    list_of_item_names = ['eta_list', 'rote_list', 'q95_list', 'Bn_Li_list', 'plot_quantity_plas','plot_quantity_vac', 'plot_quantity_tot', 'plot_quantity_plas_phase', 'plot_quantity_vac_phase', 'plot_quantity_tot_phase', 'mode_list', 'time_list', 'key_list', 'resonant_close']

    list_of_items = zip(*[eval(i) for i in list_of_item_names])
    sort_index = list_of_item_names.index(sort_name)
    print sort_index
    tmp = zip(*sorted(list_of_items, key = lambda sort_val:sort_val[sort_index]))
    output_dict2 = {}
    for loc, i in enumerate(list_of_item_names): output_dict2[i+'_arranged'] = tmp[loc]
    for loc, i in enumerate(list_of_item_names): output_dict2[i] = eval(i)

    name_list = ['plot_array_plasma', 'plot_array_vac', 'plot_array_tot', 'plot_array_vac_fixed', 'q95_array', 'phasing_array', 'plot_array_plasma_fixed', 'plot_array_plasma_phase', 'plot_array_vac_phase', 'plot_array_vac_fixed_phase', 'plot_array_plasma_fixed_phase']
    tmp1 = dBres_dBkink.dB_kink_phasing_dependence(q95_list_copy, lower_values_plasma, upper_values_plasma, lower_values_vac, upper_values_vac, lower_values_tot, upper_values_tot, lower_values_vac_fixed, upper_values_vac_fixed, phase_machine_ntor, upper_values_plas_fixed, lower_values_plas_fixed, n, n_phases = 360)
    for name, var in zip(name_list, tmp1): output_dict2[name]=var

    name_list = ['plot_array_vac_res', 'plot_array_plas_res', 'plot_array_vac_res_ave', 'plot_array_plas_res_ave']
    tmp1 = dBres_dBkink.dB_res_phasing_dependence(output_dict2['phasing_array'], output_dict2['q95_array'], res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, phase_machine_ntor, n)
    for name, var in zip(name_list, tmp1): output_dict2[name]=var

    name_list = ['q95_list_copy', 'max_loc_list', 'upper_values_vac_fixed', 'n', 'lower_values_plasma', 'lower_values_vac']
    for name in name_list: output_dict2[name]=eval(name)

    output_dict = output_dict2
    return output_dict

answers = do_everything(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name)
a = dBres_dBkink.test1(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name)


answers = a.output_dict
xaxis = np.array(answers[sort_name+'_arranged'])
dB_kink_fixed_vac = 1
if dB_kink_fixed_vac:
    a.plot_dB_kink_fixed_vac(clim1=[0,0.6],clim2=[0,1.4],xaxis_type='log',xaxis_label='rote')

dB_res_n2_dB_res_sum = 1
if dB_res_n2_dB_res_sum:
    a.dB_res_n2_dB_res_sum(clim1=None,clim2=None,xaxis_type='log',xaxis_label='rote')
    1/0
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    #color_plot = ax[0].pcolor(np.array(answers['eta_list']), answers['phasing_array'], answers['plot_array_vac_res'], cmap='hot', rasterized=True)
    #color_plot = ax[1].pcolor(np.array(answers['eta_list']), answers['phasing_array'], answers['plot_array_plas_res'], cmap='hot', rasterized=True)
    color_plot = ax[0].pcolor(xaxis, answers['phasing_array'], answers['plot_array_vac_res'], cmap='hot', rasterized=True)
    color_plot.set_clim([0,25])
    color_plot2 = ax[1].pcolor(xaxis, answers['phasing_array'], answers['plot_array_plas_res'], cmap='hot', rasterized=True)
    color_plot2.set_clim([0,50])
    #ax[0].contour(np.array(answers['eta_list']),answers['phasing_array'], answers['plot_array_vac_res'], colors='white')
    #color_plot2 = ax[1].pcolor(np.array(answers['eta_list']), answers['phasing_array'], dB_res_sum, cmap='hot', rasterized=True)
    #ax[1].contour(np.array(answers['eta_list']),answers['phasing_array'], dB_res_sum, colors='white')
    #color_plot.set_clim([0,1])
    #color_plot2.set_clim([0,1])
    title_string1 = 'Total Forcing'
    title_string2 = 'Average Forcing'
    ax[0].set_xlim([np.min(answers['eta_list']), np.max(answers['eta_list'])])
    ax[0].set_xlim([np.min(xaxis), np.max(xaxis)])
    ax[0].set_ylim([0,360])

    #ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)
    ax[0].set_title(r'$\delta B_{res}^{n=2}$',fontsize=20)
    ax[1].set_title(r'$\delta B_{res}^{n=2} + \delta B_{res}^{n=4}$',fontsize=20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    #ax[0].set_ylabel('Phasing (deg)')
    #ax[1].set_ylabel('Phasing (deg)')
    fig2, ax2 = pt.subplots(nrows=2, sharex=True, sharey=True)
    ax2[0].plot(xaxis, answers['plot_array_vac_res'][0,:], '-o',label='0deg res vac')
    ax2[0].plot(xaxis, answers['plot_array_plas_res'][0,:], '-o',label='0deg res plas')
    ax2[0].plot(xaxis, -answers['plot_array_plas_res'][0,:]+answers['plot_array_vac_res'][0,:], '-o',label='0deg total')
    ax2[1].plot(xaxis, answers['plot_array_vac_res'][180,:], '-o', label='180deg vac')
    ax2[1].plot(xaxis, answers['plot_array_plas_res'][180,:], '-o',label='180deg plas')
    ax2[1].plot(xaxis, -answers['plot_array_plas_res'][180,:]+answers['plot_array_vac_res'][180,:], '-o', label='180deg total')
    ax2[1].set_xscale('log')
    ax2[0].set_xscale('log')

    ax2[0].legend(loc='best')
    #ax2.plot(np.array(answers['eta_list']), answers['plot_array_total_res'][0,:], '-o')
    fig2.canvas.draw();fig2.show()
    cbar = pt.colorbar(color_plot, ax = ax[0])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    fig.canvas.draw(); fig.show()

#1/0

res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower = dBresdBkink.extract_dB_res(rot_pickle)
amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close = dBresdBkink.extract_dB_kink(rot_pickle, s_surface)

amps_vac_comp = dBresdBkink.apply_phasing(amps_vac_comp_upper, amps_vac_comp_lower, phasing, n, phase_machine_ntor=phase_machine_ntor)
amps_plas_comp = dBresdBkink.apply_phasing(amps_plas_comp_upper, amps_plas_comp_lower, phasing, n, phase_machine_ntor=phase_machine_ntor)
amps_tot_comp = dBresdBkink.apply_phasing(amps_tot_comp_upper, amps_tot_comp_lower, phasing, n, phase_machine_ntor=phase_machine_ntor)
if reference_dB_kink=='plas':
    reference = dBres_dBkink.get_reference(amps_plas_comp_upper, amps_plas_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)
elif reference_dB_kink=='tot':
    reference = dBres_dBkink.get_reference(amps_tot_comp_upper, amps_tot_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)

for i in rot_pickle['sims'].keys():
    rote_list.append(rot_pickle['sims'][i]['MARS_settings']['<<ROTE>>'])
    res_list.append(rot_pickle['sims'][i]['MARS_settings']['<<ETA>>'])
    total_res.append(rot_pickle['sims'][i]['responses']['total_resonant_response_upper_integral']+rot_pickle['sims'][i]['responses']['total_resonant_response_lower_integral'])
    vac_res.append(rot_pickle['sims'][i]['responses']['vacuum_resonant_response_upper_integral']+rot_pickle['sims'][i]['responses']['vacuum_resonant_response_lower_integral'])
    plas_res.append(total_res[-1] - vac_res[-1])

#for i in res_pickle['sims'].keys():
#    res_list.append(res_pickle['sims'][i]['MARS_settings']['<<ETA>>'])
#    total_res.append(res_pickle['sims'][1]['responses']['total_resonant_response_upper_integral']+res_pickle['sims'][1]['responses']['total_resonant_response_lower_integral'])
fig, ax = pt.subplots()
ax.plot(res_list, total_res, '-o', label='totl')
ax.plot(res_list, vac_res, '-o', label='vac')
ax.plot(res_list, plas_res, '-o', label='plas')
ax.legend(loc='best')
fig.canvas.draw(); fig.show()
