#!/usr/bin/env Python
from PythonMARS_funcs import *
import Chease_Batch_Launcher as ch_launch
import pickle, time

project_name = sys.argv[1]

############ SET THESE VALUES BEFORE RUNNING!!!!! #####
simultaneous_jobs = 20
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##############################################################


print 'running chease section'
overall_start = time.time()


pickle_file = open(project_dir + '2_'+project_name+'_setup_directories.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

#Chease Setup
for i in project_dict['sims'].keys():
    print i
    copy_chease_files(project_dict['sims'][i])
    modify_datain(project_dict['sims'][i],project_dict['details']['template_dir'])
    generate_chease_job_file(project_dict['sims'][i])
    #execute_chease(project_dict['sims'][i])
    #project_dict['sims']['chease_run']=1

project_dict['sims'] = ch_launch.batch_launch_chease(project_dict['sims'], simultaneous_jobs)
pickle_file = open(project_dict['details']['base_dir']+'3_' +project_name+ '_post_chease.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()
print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
