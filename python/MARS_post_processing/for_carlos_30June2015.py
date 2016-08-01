import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_funcs
import copy
ul = True; mars_params = None
file_name = '/home/shaskey/NAMP_datafiles/mars/shot158115_04780_res_rot/shot158115_04780_res_rot_post_processing_PEST.pickle'
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

rot_pts = None
res_pts = None
time_pts = None
def plot_stuff(obj, clim  = None, field = None, zoomin=False, save_data = False, label_prefix = ''):
    if field == None: field = 'plasma'
    if clim==None:clim = [0,1]
    fig, ax = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
    ax_new = ax.flatten()
    clr_ax = []
    contour_kwargs = {'colors':'k'}
    plot_dots_kwargs = {'color':'k'}
    if obj.calc_type == 'probe':
        suptitle = label_prefix + '_' + obj.probe
    else:
        suptitle = label_prefix + '_' + obj.calc_type
    zoomin_txt = 'zoomed' if zoomin else ''
    suptitle += '_'+field
    figname = '{}_{}.png'.format(suptitle, zoomin_txt)
    for i,ang in enumerate(range(0,360,90)):
        tmp = obj.plot_2D(ang,'ROTE','ETA', ax = ax_new[i], plot_dots = True, clim = clim, n_contours = 10, contour_kwargs = contour_kwargs, plot_dots_kwargs=plot_dots_kwargs, field = field, return_data = True)
        clr_ax.append(tmp[0])
        for tmp2, lab in zip(tmp[1],['x','y','z']):
            np.savetxt('{}_{}_{}deg.txt'.format(suptitle,lab,ang), tmp2)
        ax_new[i].set_title('{}'.format(ang))
    for i in clr_ax: print i.get_clim()
    #for i in clr_ax: i.set_clim(clim)
    if zoomin:
        ax_new[0].set_xlim([1.e-3,1.e-1])
        ax_new[0].set_ylim([5.e-9,5.e-7])
    for i in ax[-1,:]:i.set_xlabel('omega')
    for i in ax[:,0]:i.set_ylabel('eta')
    pt.colorbar(clr_ax[0], ax = ax_new.tolist())
    fig.suptitle(suptitle)
    fig.canvas.draw();fig.show()
    fig.savefig(figname)
calc_type = ['plasma','plasma','total','plasma']
for obj,clim,calc_type in zip([probe,probe2,dBres,dBkink],[[0,1.3],[0,1.3],[0,0.5],[0,1.5]], calc_type):
    plot_stuff(obj, clim  = clim, field = calc_type, zoomin=True, label_prefix = 'carlos_1July2015')
