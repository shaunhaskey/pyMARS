#!/usr/bin/env Python

## This script will modify datain file for CHEASE and run it on every serial number in the
## datastructure.
## Make sure you run this script on the Venus cluster as it will try to submit jobs!
## Probably best to submit this when not many people are using the network if you are going
## to set simultaneous_jobs to a large number!!!!
##
## ## bash $: 3_run_chease.py project_name
## bash $: nohup 3_run_chease.py project_name > step2_log &
##
##
## Shaun Haskey Sept 28 2011


from PythonMARS_funcs import *
import Chease_Batch_Launcher as ch_launch
import pickle, time

project_name = sys.argv[1]


############ SET THESE VALUES BEFORE RUNNING!!!!! #####
simultaneous_jobs = 26
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##############################################################


overall_start = time.time()

# Open data structure from previous step
pickle_file_name = '2_'+project_name+'_setup_directories.pickle'
pickle_file_name = '9_project1_new_eq_COIL_upper_post_setup_new_low_beta.pickle'

pickle_file = open(project_dir + pickle_file_name,'r')

project_dict = pickle.load(pickle_file)
pickle_file.close()

#Chease Setup
for i in project_dict['sims'].keys():
    print i
    project_dict['sims'][i]['dir_dict']['chease_dir_PEST'] = project_dict['sims'][i]['dir_dict']['chease_dir'].rstrip('/') + '_PEST/'
    os.system('mkdir ' + project_dict['sims'][i]['dir_dict']['chease_dir_PEST'])

    #copy the Chease template
    copy_chease_files(project_dict['sims'][i], PEST=1)
    #modify the datain so it is relevant for this project
    modify_datain(project_dict['sims'][i],project_dict['details']['template_dir'], PEST=1)
    #generate a job file that can be submitted to the cluster
    generate_chease_job_file(project_dict['sims'][i], PEST=1)

#This is the step that launches the batch job
project_dict['sims'] = ch_launch.batch_launch_chease(project_dict['sims'], simultaneous_jobs, PEST=1)


#Dump the data structure for the next step
#pickle_file = open(project_dict['details']['base_dir']+'3_' +project_name+ '_post_chease_PEST.pickle','w')
#pickle.dump(project_dict,pickle_file)
#pickle_file.close()


print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
