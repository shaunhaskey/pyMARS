#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, pickle

project_name = sys.argv[1]

# at the moment this needs to be run on benten - IS THIS TRUE????
################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##################################


overall_start = time.time()
pickle_file = open(project_dir + '3_'+project_name+'_post_chease.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()




total_finished = 0
start_time = time.time()
total_jobs = len(project_dict['sims'].keys())
for i in project_dict['sims'].keys():
    fxin_create(project_dict['sims'][i])
    fxrun(project_dict['sims'][i])
    project_dict['sims'][i]['fxrun']=1
    total_finished += 1
    
    print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)



pickle_file = open(project_dict['details']['base_dir']+'4_'+ project_name +'_post_fxrun.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()


print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
