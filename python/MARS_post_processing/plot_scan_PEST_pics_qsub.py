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

qsub_workers = 20
file_name = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/shot_142614_rote_res_scan_20x20_kpar1_low_rote_post_processing_PEST.pickle'
file_name = '/u/haskeysr/mars/shot_142614_expt_scan_NC_const_eqV1/shot_142614_expt_scan_NC_const_eqV1_post_processing_PEST.pickle'
with file(file_name, 'r') as file_handle: scan_data = pickle.load(file_handle)


keys = np.array(scan_data['sims'].keys())
print keys.shape
eta_list = []; rote_list = []
for i in keys: eta_list.append(scan_data['sims'][i]['MARS_settings']['<<ETA>>'])
for i in keys: rote_list.append(scan_data['sims'][i]['MARS_settings']['<<ROTE>>'])
#keys = keys[(np.array(eta_list)==1.1288378916846883e-6) * (np.array(rote_list)==1.e-6)]
#keys = keys[(np.array(eta_list)==1.1288378916846883e-6)]
#for i in range(len(keys)): keys[i] = 1
print keys.shape

execute_func = '/u/haskeysr/code/NAMP_analysis/python/MARS_post_processing/plot_scan_PEST_pics_qsub_exec.py'

def chunks(l, n):
    per_list = int(np.ceil(float(len(l))/n))
    return [l[i:i+per_list] for i in range(0, len(l), per_list)]

key_list = chunks(keys, qsub_workers)

print key_list

for i, cur_key in enumerate(key_list):
    job_string = '#!/bin/bash\n#$ -N '+ 'PostProcImages'+str(i)+'\n#$ -q all.q\n#$ -o sge_output.dat\n#$ -e sge_error.dat\n#$ -cwd\nexport PATH=$PATH:/f/python/linux64/bin\nsource ~/.bashrc\n'
    key_list_string  = ' '.join([str(j) for j in cur_key])
    job_name = 'job_{}.job'.format(i)
    
    execute_command = '{} {} {} > {}\n'.format(execute_func, file_name, key_list_string, job_name + '.log')
    job_string += execute_command
    with file(job_name, 'w') as job_handle: job_handle.write(job_string)
    os.system('qsub ' + job_name)

