import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_funcs
import copy

ul = True
file_name = '/home/srh112/NAMP_datafiles/mars/project1_redone/project1_redone_post_processing_PEST.pickle'; mars_params = []
file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_146394/equal_spacing_146394_post_processing_PEST.pickle'; mars_params = []
file_name = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_processing_PEST.pickle'; mars_params = []
file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan_n2_146382_NVEXP_4/detailed_q95_scan_n2_146382_NVEXP_4_post_processing_PEST.pickle'; mars_params = []
file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacingV2/equal_spacingV2_post_processing_PEST.pickle'; mars_params = []
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


a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False, ul = ul, mars_params = mars_params)

dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean')
dBkink = dBres_dBkink.dBkink_calculations(a)
probe = dBres_dBkink.magnetic_probe(a,' 66M')
probe_r = dBres_dBkink.magnetic_probe(a,'UISL')

#probe_r = dBres_dBkink.magnetic_probe(a,'Inner_rad')
labels = ['$\delta B_{res}^{tot}$ (G/kA)', '$\delta B_{kink}$ (G/kA)', 'Midplanae outboard poloidal probe (G/kA)']


tmp = probe
#tmp = dBkink
phasing = 0
yaxis = 'BNLI'; xaxis = 'Q95'
n_interp = 100
clim1 = [0,8]
clim2 = [-np.pi,np.pi]
fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True)
gen_funcs.setup_publication_image(fig, height_prop = 1., single_col = True, replacement_kwargs = {'lines.markersize': 2.0})
im = tmp.plot_2D_irregular(phasing, xaxis, yaxis, field = 'plasma',  ax = ax[0], amplitude = True, cmap_res = 'jet', clim = clim1, yaxis_log = False, xaxis_log = False, n_contours = 0, contour_kwargs = None, n_x = n_interp, n_y = n_interp, pt_datapts = True)
im2 = tmp.plot_2D_irregular(phasing, xaxis, yaxis, field = 'plasma',  ax = ax[1], amplitude = False, cmap_res = 'RdBu', clim = clim2, yaxis_log = False, xaxis_log = False, n_contours = 0, contour_kwargs = None, n_x = n_interp, n_y = n_interp, pt_datapts = True)
cbar = pt.colorbar(im,ax = ax[0])
cbar2 = pt.colorbar(im2,ax = ax[1])
ax[0].set_ylim([0.8,4.5])
ax[0].set_xlim([2.5,6.8])
cbar.set_label('amplitude G/kA 66M')
cbar2.set_label('phase rad 66M')
ax[-1].set_xlabel('$q_{95}$')
for i in ax: i.set_ylabel(r'$\beta_N / L_i$')
fig.savefig('66M_142382.pdf')
fig.canvas.draw(); fig.show()

1/0



fnames = ['dBres','dBkink','probe']
clims = [[0,2.5],[0,2],[0,1.5]]
clims = [[0,5],[0,5],[0,15]]
clims = [[0,1.5],[0,0.8],[0,1.05]]
for calc_type, cur_clim, title, fname, field in zip([dBres, dBkink, probe], clims, labels, fnames, fields):
    replacement_kwargs = {'xtick.labelsize': 7.0,'ytick.labelsize': 7.0}
    if len(phasings_disp)==8:
        fig, ax = pt.subplots(nrows = 2, ncols = 4, sharex = True, sharey = True)
        gen_funcs.setup_publication_image(fig, height_prop = 1./2.0, single_col = False, replacement_kwargs = replacement_kwargs)
    else:
        fig, ax = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
        gen_funcs.setup_publication_image(fig, height_prop = 1./1.1, single_col = True, replacement_kwargs = replacement_kwargs)
    #gen_funcs.setup_publication_image(fig, height_prop = 1./1.618, single_col = False)
    for i, cur_ax in zip(phasings_disp, ax.flatten()):
        cax = calc_type.plot_2D(i,x_axis,y_axis,cmap_res = 'jet', field = field, clim = cur_clim, ax = cur_ax, med_filt_value = 1, n_contours = 15, contour_kwargs = {'colors':'k', 'linewidths':0.5})
        cur_ax.set_title('$\Delta \phi = {}^o$'.format(i))
        if vlines!=None: 
            for vert_line in vlines: cur_ax.axvline(x=vert_line, color='k')
        if hlines!=None: 
            for hor_line in hlines: cur_ax.axhline(y=hor_line, color='k')
        if res_pts!=None:
            #cur_ax.plot(rot_pts, res_pts, 'k.')
            #cur_ax.plot(rot_pts, np.array(res_pts)/10, 'k.')
            for t_tmp in time_plt_pts: 
                cur_ax.plot(rot_pts[time_pts.index(t_tmp)], res_pts[time_pts.index(t_tmp)], 'k.')
                cur_ax.text(rot_pts[time_pts.index(t_tmp)], res_pts[time_pts.index(t_tmp)], t_tmp, fontsize=6)
            for t_tmp in time_plt_pts: 
                cur_ax.plot(rot_pts[time_pts.index(t_tmp)], res_pts[time_pts.index(t_tmp)]/10, 'k.')
                cur_ax.text(rot_pts[time_pts.index(t_tmp)], res_pts[time_pts.index(t_tmp)]/10, t_tmp, fontsize=6)
            #for rot_tmp, res_tmp, t_tmp in zip(rot_pts, res_pts, time_pts): cur_ax.text(rot_tmp, res_tmp, t_tmp, fontsize=7)
            #for rot_tmp, res_tmp, t_tmp in zip(rot_pts, res_pts, time_pts): cur_ax.text(rot_tmp, res_tmp/10, t_tmp, fontsize=7)
    for i in ax[:,0]: i.set_ylabel(y_axis_label)
    for i in ax[-1,:]: i.set_xlabel(x_axis_label)
    if xlim!=None: ax[0,0].set_xlim(xlim)
    if ylim!=None: ax[0,0].set_ylim(ylim)
    fig.tight_layout(pad = 0.1)
    cbar = pt.colorbar(cax, ax = ax.flatten().tolist())
    #cbar.set_label('{}-{}'.format(title, field))
    cbar.set_label('{}'.format(title))
    fig.savefig(fname+'-'+field+'2x2.pdf', bbox_inches = 'tight',pad = 0.1)
    fig.savefig(fname+'-'+field+'2x2.eps', bbox_inches = 'tight',pad = 0.1)
    fig.canvas.draw(); fig.show()
