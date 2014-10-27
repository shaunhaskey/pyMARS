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
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

#file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/shot_142614_expt_scan_NC_dif_eq_dif_rot_prof_spitz_thetac_005/shot_142614_expt_scan_NC_dif_eq_dif_rot_prof_spitz_thetac_005_post_processing_PEST.pickle'
file_name = '/u/haskeysr/mars/shot158115_04702_n2_rot_scan/shot158115_04702_n2_rot_scan_post_processing_PEST.pickle'
with file(file_name, 'r') as file_handle: scan_data = pickle.load(file_handle)
I = scan_data['details']['I-coils']['I_coil_current']
N = scan_data['details']['I-coils']['N_Icoils']
n = np.abs(scan_data['details']['MARS_settings']['<<RNTOR>>'])
I0EXP = results_class.I0EXP_calc_real(n,I)
print I, N, n, I0EXP

def plot(args):
    print 'hello', os.getpid()
    valid_sim, im_name, I0EXP, facn, subplot_plot, n, inc_contours, clim, fig_title = args
    combs = itertools.product(['upper','lower'],['plasma','vacuum'])
    simuls = {}
    for loc, typ in combs:
        simuls['{}_{}'.format(loc,typ)] = data(valid_sim['dir_dict']['mars_{}_{}_dir'.format(loc, typ)], I0EXP = I0EXP)
    for i in simuls.keys(): simuls[i].get_PEST(facn = facn)
    

clim = [0,4.0]
base_dir = '/u/haskeysr/tmp_ims/'
base_dir = '/u/haskeysr/expt_ims2/'

keys = np.array(scan_data['sims'].keys())
#keys = [1,2]
print keys.shape
print 'hello world'
eta_list = []
rote_list = []
for i in keys: eta_list.append(scan_data['sims'][i]['MARS_settings']['<<ETA>>'])
for i in keys: rote_list.append(scan_data['sims'][i]['MARS_settings']['<<ROTE>>'])
#for i in keys: time_list.append(scan_data['sims'][i]['shot_time']
#print time_list
#keys = keys[(np.array(eta_list)==1.1288378916846883e-6) * (np.array(rote_list)==1.e-6)]
#keys = keys[(np.array(eta_list)==1.1288378916846883e-6)]
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
    #im_name_list.append('{}/time_{:05d}.png'.format(base_dir, scan_data['sims'][i]['shot_time']))
    title_list.append('ROTE_{:.3e} ETA {:.3e} {}'.format(scan_data['sims'][i]['MARS_settings']['<<ROTE>>'], scan_data['sims'][i]['MARS_settings']['<<ETA>>'], subplot_plot).replace('_','-'))
    #title_list.append('time {:05d}'.format(scan_data['sims'][i]['shot_time']))
print im_name_list

input_data = zip(valid_sim_list, im_name_list, itertools.repeat(I0EXP), itertools.repeat(facn), itertools.repeat(subplot_plot), itertools.repeat(n), itertools.repeat(inc_contours), itertools.repeat(clim), title_list)

map(scan_pics_func.plot_scan, input_data)

#pool = multiprocessing.Pool(10, maxtasksperchild = 1)
#pool.map(scan_pics_func.plot_scan, input_data)

