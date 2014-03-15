'''
SH : Nov 21 2012
This is useful for creating PEST plot images
It can make an animation and introduce different phasings
'''


import  results_class
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt
import copy
import cPickle as pickle
import PythonMARS_funcs as pyMARS
import multiprocessing
import itertools, os
import scan_pics_func
N = 6; n = 3
I = np.array([1.,-1.,0.,1,-1.,0.])
I = np.array([1.,-1.,1.,-1,-1.,-1.])
I0EXP = results_class.I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
with file(file_name, 'r') as file_handle: scan_data = pickle.load(file_handle)

def plot(args):
    #print 'hello'                                                                                                           
    print 'hello', os.getpid()
    valid_sim, im_name, I0EXP, facn, subplot_plot, n, inc_contours, clim, fig_title = args
    combs = itertools.product(['upper','lower'],['plasma','vacuum'])
    simuls = {}
    for loc, typ in combs:
        simuls['{}_{}'.format(loc,typ)] = data(valid_sim['dir_dict']['mars_{}_{}_dir'.format(loc, typ)], I0EXP = I0EXP)
    for i in simuls.keys(): simuls[i].get_PEST(facn = facn)
    
#     phasings = [0,90,180,270]
#     fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =True, sharey = True)
#     ax[0,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
#     ax[1,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
#     ax[1,0].set_xlabel('m')
#     ax[1,1].set_xlabel('m')
#     ax = ax.flatten()
#     color_plots = []
#     for i, phasing in enumerate(phasings):
#         print phasing
#         combined = copy.deepcopy(simuls['upper_plasma'])
#         #Combine the upper and lower data with the appropriate phasing
#         R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(simuls['upper_plasma'], simuls['lower_plasma'], phasing)
#         R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(simuls['upper_vacuum'], simuls['lower_vacuum'], phasing)
#         print 'totals are ', fig_title, np.sum(np.abs(BnPEST_t)), np.sum(np.abs(BnPEST_v))
#         #Choose which plot to create
#         cmap = 'spectral'
#         if subplot_plot=='total':
#             combined.BnPEST = BnPEST_t
#             color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours, cmap = cmap))
#             #color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST=1, gauss_filter = [0,0.05]))
#         elif subplot_plot=='vac':
#             combined.BnPEST = BnPEST_v
#             color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours, cmap = cmap))
#         elif subplot_plot=='plasma':
#             combined.BnPEST = BnPEST_t - BnPEST_v
#             color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours, cmap = cmap))

#         ax[i].set_title(r'$\Delta \phi_{ul} = %d^o$'%(phasing),fontsize = 18)
#         color_plots[-1].set_clim()
#     ax[0].set_xlim([0,25])
#     ax[0].set_ylim([0.4,1])
#     fig.suptitle(fig_title)
#     fig.savefig('{}'.format(im_name))
#     fig.clf()
#     #fig.close()

clim = [0,6.0]
base_dir = '/u/haskeysr/tmp_ims/'

keys = np.array(scan_data['sims'].keys())
#keys = [1,2]
print keys.shape
eta_list = []
rote_list = []
for i in keys: eta_list.append(scan_data['sims'][i]['MARS_settings']['<<ETA>>'])
for i in keys: rote_list.append(scan_data['sims'][i]['MARS_settings']['<<ROTE>>'])
#keys = keys[(np.array(eta_list)==1.1288378916846883e-6) * (np.array(rote_list)==1.e-6)]
keys = keys[(np.array(eta_list)==1.1288378916846883e-6)]
#for i in range(len(keys)): keys[i] = 1
print keys.shape

subplot_plot = 'vac'
inc_contours = False
valid_sim_list = []
im_name_list = []
title_list = []
for i in keys: valid_sim_list.append(copy.deepcopy(scan_data['sims'][i]))
for i in keys:
    im_name_list.append('{}/ROTE_{:010d}_ETA_{:010d}_{}.png'.format(base_dir, int(scan_data['sims'][i]['MARS_settings']['<<ROTE>>']*10**10), int(scan_data['sims'][i]['MARS_settings']['<<ETA>>']*10**10), subplot_plot))
    title_list.append('ROTE_{:.3e} ETA {:.3e} {}'.format(scan_data['sims'][i]['MARS_settings']['<<ROTE>>'], scan_data['sims'][i]['MARS_settings']['<<ETA>>'], subplot_plot))
print im_name_list

input_data = zip(valid_sim_list, im_name_list, itertools.repeat(I0EXP), itertools.repeat(facn), itertools.repeat(subplot_plot), itertools.repeat(n), itertools.repeat(inc_contours), itertools.repeat(clim), title_list)

#map(scan_pics_func.plot_scan, input_data)

#pool = multiprocessing.Pool(10, maxtasksperchild = 1)
#pool.map(scan_pics_func.plot_scan, input_data)

