#!/usr/bin/env Python
## This script sets up the MARS RUN file
## No settings are changed from the original setup process.
## The other 6_ files allow variations on what has already been done, and to also setup
## Variation type runs
##
## ## bash $: 6_setup_mars.py project_name
## bash $: nohup 6_setup_mars.py project_name > step6_log &
##
##
## Shaun Haskey Sept 28 2011

from PythonMARS_funcs import *
import time, os, sys, pickle

overall_start = time.time()
project_name = sys.argv[1]


################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##################################

#Open previous data structure 
pickle_file = open(project_dir + '5_'+project_name+'_post_RMZM.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

def setup_mars_func(project_dict):
    for i in project_dict['sims'].keys():
        print i
        #Extract required values from CHEASE log file
        project_dict['sims'][i] = extract_NW(project_dict['sims'][i])

        #Setup MARS vacuum run
        mars_setup_files(project_dict['sims'][i], vac = 1)

        #Calculate the values that need to be normalised to something to do with Alfven speed/frequency
        project_dict['sims'][i] = mars_setup_alfven(project_dict['sims'][i], project_dict['sims'][i]['ICOIL_FREQ'], vac = 1)

        mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 1)
        generate_job_file(project_dict['sims'][i],1) #create Venus cluster job file

        #Setup MARS plasma run
        mars_setup_files(project_dict['sims'][i], vac = 0)
        mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 0)
        generate_job_file(project_dict['sims'][i],0) #create Venus cluster job file

    return project_dict

project_dict = setup_mars_func(project_dict)


#Save the data structure so that it can be read by the next step
pickle_file = open(project_dict['details']['base_dir']+'6_' +project_name +'_post_setup.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()


print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
