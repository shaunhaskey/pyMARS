'''
SH : Nov 21 2012
This is useful for creating PEST plot images
It can make an animation and introduce different phasings
'''


from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt
import copy
import cPickle as pickle
import PythonMARS_funcs as pyMARS
import multiprocessing
import itertools, os
N = 6; n = 3
I = np.array([1.,-1.,0.,1,-1.,0.])
I = np.array([1.,-1.,1.,-1,-1.,-1.])
I0EXP = I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
with file(file_name, 'r') as file_handle: scan_data = pickle.load(file_handle)

def plot(args):
    #print 'hello'                                                                                                           
    print os.getpid()
    valid_sim, im_name, I0EXP, facn, subplot_plot, n, inc_contours, clim = args
    combs = itertools.product(['upper','lower'],['plasma','vacuum'])
    simuls = {}
    for loc, typ in combs:
        simuls['{}_{}'.format(loc,typ)]data(valid_sim['dir_dict']['mars_{}_{}_dir'.format(loc, typ)], I0EXP = I0EXP)
    for i in simuls: i.get_PEST(facn = facn)
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
        combined = copy.deepcopy(simuls['upper_plasma'])
        #Combine the upper and lower data with the appropriate phasing
        R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(simuls['upper_plasma'], simuls['lower_plasma'], phasing)
        R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(simuls['upper_vacuum'], simuls['lower_vacuum'], phasing)

        #Choose which plot to create
        if subplot_plot=='total':
            combined.BnPEST = BnPEST_t
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours))
            #color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST=1, gauss_filter = [0,0.05]))
        elif subplot_plot=='vac':
            combined.BnPEST = BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours))
        elif subplot_plot=='plasma':
            combined.BnPEST = BnPEST_t - BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours))

        ax[i].set_title(r'$\Delta \phi_{ul} = %d^o$'%(phasing),fontsize = 18)
        color_plots[-1].set_clim()
    ax[0].set_xlim([0,25])
    ax[0].set_ylim([0.4,1])
    fig.savefig('{}.png'.format(im_name))
    fig.clf()
    fig.close()

clim = [0,6.0]
pool = multiprocessing.Pool(4)
base_dir = '/u/haskeysr/tmp_ims/'

keys = scan_data['sims'].keys()

subplot_plot = 'total'
inc_contours = True
valid_sim_list = []
im_name_list = []
for i in keys: valid_sim_list.append(scan_data['sims'][i])
for i in keys:
    im_name_list.append('{}_ROTE_{:d}_ETA_{:d}.png'.format(base_dir, scan_data['sims'][i]['MARS_settings']['<<ROTE>>']*10**10, scan_data['sims'][i]['MARS_settings']['<<ETA>>']*10**10))
print im_name_list

input_data = zip(valid_sim_list, im_name_list, itertools.repeat(I0EXP), itertools.repeat(facn), itertools.repeat(subplot_plot), itertools.repeat(n), itertools.repeat(inc_contours), itertools.repeat(clim))
pool.map(plot, input_data)
