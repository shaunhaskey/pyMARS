#!/usr/bin/env Python
# Script to create an fxin file for each equilibria, and to run FourierFX on the RMZM file.
# This step is required acording to Yueqiang.
# Best to run this step on benten - IS THIS necessary?
# Creates fxin file
# Runs Fourierfx on that file
## ## bash $: ./4_setup_run_fxrun.py project_name
## bash $: nohup 4_setup_run_fxrun.py project_name > step4_log &
## You can then monitor what is happening:
## bash $: tail -f step4_log
## Shaun Haskey Sept 28 2011


from PythonMARS_funcs import *
import time, pickle

project_name = sys.argv[1]

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##################################

#Load pickle file from previous step
overall_start = time.time()
pickle_file = open(project_dir + '3_'+project_name+'_post_chease.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()


total_finished = 0
start_time = time.time()
total_jobs = len(project_dict['sims'].keys())
for i in project_dict['sims'].keys():
    project_dict['sims'][i]['dir_dict']['chease_dir_PEST'] = project_dict['sims'][i]['dir_dict']['chease_dir'].rstrip('/') + '_PEST/'

    #Create the fxin file
    fxin_create(project_dict['sims'][i], PEST=1)

    #Run fxrun
    fxrun(project_dict['sims'][i], PEST=1)
    project_dict['sims'][i]['fxrun_PEST']=1
    total_finished += 1
    
    print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)


#Dump pickle file for the next step
pickle_file = open(project_dict['details']['base_dir']+'4b_'+ project_name +'_post_fxrun.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()


print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
