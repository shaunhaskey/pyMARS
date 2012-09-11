#!/usr/bin/env Python
from PythonMARS_funcs import *
import Chease_Batch_Launcher as ch_launch
import time, os, sys, pickle


simultaneous_jobs = 20

overall_start = time.time()
print 'running mars setup section'
project_name = sys.argv[1]
pickle_file = open('/u/haskeysr/mars/'+ project_name +'/6_post_mars_setup.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

project_dict['sims'] = ch_launch.batch_launch_mars(project_dict['sims'],simultaneous_jobs)

pickle_file = open(project_dict['details']['base_dir']+'7_post_mars_run.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
