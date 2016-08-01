#!/usr/bin/env Python
## This is the option for fancy... needs some work

from PythonMARS_funcs import *
import Chease_Batch_Launcher as ch_launch
import time, os, sys, pickle
import numpy as num

project_name = sys.argv[1]

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
var_name = 'ROTE'
var_name = 'FEEDI'
var_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
var_list = num.arange(0,1.55,0.05)
var_list = num.arange(0.8,1,0.01)
var_list = [0]#[-120]#[-180]#[-300]
simultaneous_jobs = 18
################################

overall_start = time.time()
print 'running mars setup section'

giant_pickle ={}
giant_pickle['sims']={}
current = 0
file_name = open('/scratch/haskeysr/jobs.txt','w')
file_name.write(str(simultaneous_jobs)+'\n')
file_name.close()

for var_value in var_list:
    pickle_file = open(project_dir + '6_'+ project_name + '_' + var_name + '_' + str(var_value) + '_post_setup.pickle','r')
    project_dict = pickle.load(pickle_file)
    pickle_file.close()

    for i in project_dict['sims'].keys():
        giant_pickle['sims'][current] = project_dict['sims'][i]
        current += 1
    giant_pickle['details'] = project_dict['details']

giant_pickle['sims'] = ch_launch.batch_launch_mars(giant_pickle['sims'],simultaneous_jobs)

#pickle_file = open(project_dict['details']['base_dir']+'7_' + project_name + '_' + var_name + '_' + str(var_value) + '_post_mars_run.pickle','w')
pickle_file = open(giant_pickle['details']['base_dir']+'7_' + project_name + '_' + var_name + '_giant_file_post_mars_run.pickle','w')

pickle.dump(giant_pickle,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
