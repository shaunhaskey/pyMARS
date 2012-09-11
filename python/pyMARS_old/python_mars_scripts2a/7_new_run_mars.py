#!/usr/bin/env Python
#This can be the standard option
#This needs modification. Should separate out the parts that are common to 7_,7a,7c

from PythonMARS_funcs import *
import Chease_Batch_Launcher as ch_launch
import time, os, sys, pickle

project_name = sys.argv[1]


################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##################################

def run_mars_function(input_file_name, output_file_name, simultaneous_jobs):
    print 'running mars setup section'
    project_dict['sims'] = ch_launch.batch_launch_mars(project_dict['sims'],simultaneous_jobs)
    return project_dict

overall_start = time.time()

simultaneous_jobs = 23

pickle_file = open(input_file_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

input_file_name = project_dir + '6_'+project_name+'_post_setup.pickle','r')
output_file_name = project_dir +'7_' + project_name + '_post_mars_run.pickle'
project_dict = run_mars_function(input_file_name, output_file_name, simultaneous_jobs)

pickle_file = open(output_file_name,'w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
