#!/usr/bin/env Python
from PythonMARS_funcs import *
import Chease_Batch_Launcher as ch_launch
import time, os, sys, pickle


project_name = sys.argv[1]

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
var_name = 'ROTE'
var_value = 1.25
simultaneous_jobs = 10
################################


overall_start = time.time()
print 'running mars setup section'
pickle_file = open(project_dir + '6_'+ project_name + '_' + var_name + '_' + str(var_value) + '_post_setup.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

project_dict['sims'] = ch_launch.batch_launch_mars(project_dict['sims'],simultaneous_jobs)

pickle_file = open(project_dict['details']['base_dir']+'7_' + project_name + '_' + var_name + '_' + str(var_value) + '_post_mars_run.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)




