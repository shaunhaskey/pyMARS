#!/usr/bin/env Python
'''
SH : Nov 21 2012
This is useful for creating PEST plot images
It can make an animation and introduce different phasings
'''
import  results_class
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pt
import copy
import cPickle as pickle
import PythonMARS_funcs as pyMARS
import multiprocessing
import itertools, os, sys
import scan_pics_func

keys = [int(sys.argv[i]) for i in range(1,len(sys.argv))]

N = 6; n = 3
I = np.array([1.,-1.,0.,1,-1.,0.])
I = np.array([1.,-1.,1.,-1,-1.,-1.])
I0EXP = results_class.I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?
file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
with file(file_name, 'r') as file_handle: scan_data = pickle.load(file_handle)

clim = [0,3.0]
base_dir = '/u/haskeysr/tmp_ims/'

subplot_plot = 'total'
inc_contours = False
valid_sim_list = []
im_name_list = []
title_list = []
for i in keys: valid_sim_list.append(copy.deepcopy(scan_data['sims'][i]))
for i in keys:
    #im_name_list.append('{}/ROTE_{:010d}_ETA_{:010d}_{}.png'.format(base_dir, int(scan_data['sims'][i]['MARS_settings']['<<ROTE>>']*10**10), int(scan_data['sims'][i]['MARS_settings']['<<ETA>>']*10**10), subplot_plot))
    im_name_list.append('{}/ROTE_{:010d}_ETA_{:010d}'.format(base_dir, int(scan_data['sims'][i]['MARS_settings']['<<ROTE>>']*10**10), int(scan_data['sims'][i]['MARS_settings']['<<ETA>>']*10**10)))
    title_list.append('ROTE {:.3e} ETA {:.3e}'.format(scan_data['sims'][i]['MARS_settings']['<<ROTE>>'], scan_data['sims'][i]['MARS_settings']['<<ETA>>']))
print im_name_list

input_data = zip(valid_sim_list, im_name_list, itertools.repeat(I0EXP), itertools.repeat(facn), itertools.repeat(subplot_plot), itertools.repeat(n), itertools.repeat(inc_contours), itertools.repeat(clim), title_list)

map(scan_pics_func.plot_scan, input_data)

#pool = multiprocessing.Pool(10, maxtasksperchild = 1)
#pool.map(scan_pics_func.plot_scan, input_data)

