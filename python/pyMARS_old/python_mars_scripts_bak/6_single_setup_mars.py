#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, os, sys, pickle
import numpy as num

overall_start = time.time()

#Obtain the project name from the initialisation - could use an environment variable for this in future?
#Needed to open the correct pickle file and working directory
project_name = sys.argv[1]

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
var_name = 'COIL'
var_list = ['lower','upper']
##################################


#Cycle through the var_list settings
for var_value in var_list:
    pickle_file = open(project_dir + '5_'+project_name+'_post_RMZM_low_beta.pickle','r')
    project_dict = pickle.load(pickle_file)
    pickle_file.close()


    total_jobs = len(project_dict['sims'].keys())
    completed = 0

    #FEEDI_string = 
    #Change section in details
    #project_dict['details']['FEEDI'] = FEEDI_string

    
    #Cycle through all the simulations in the dictionary
    for i in project_dict['sims'].keys():
        print i
        #create new directories and update mars_run dirs
        vac_name = 'RUNrfa_' + var_name+str(var_value)+'.vac'
        plasma_name = 'RUNrfa_' + var_name+str(var_value)+'.p'

        project_dict['sims'][i] = mars_directories(project_dict['sims'][i], vac_name, plasma_name)
        project_dict['sims'][i] = extract_NW(project_dict['sims'][i])

        #Mars vac
        mars_setup_files(project_dict['sims'][i], 1) #link necessary files from chease run
        mars_setup_files(project_dict['sims'][i], 0) #as above for vacuum

        #Calculate parameters in terms of Alfven frequency - need to verify this works
        project_dict['sims'][i] = mars_setup_alfven(project_dict['sims'][i], project_dict['sims'][i]['ICOIL_FREQ'], vac = 1)
        project_dict['sims'][i]['FEEDI'] = '(1.0, 0.0),'

        mars_setup_run_file_single(project_dict['sims'][i], project_dict['details']['template_dir'], var_value, vac = 1)
        mars_setup_run_file_single(project_dict['sims'][i], project_dict['details']['template_dir'], var_value, vac = 0)

        generate_job_file(project_dict['sims'][i],1)
        generate_job_file(project_dict['sims'][i],0)

        completed+=1
        print 'completed : %d of %d'%(completed, total_jobs)

    #dump the new pickle file - now has details of where the relevant mars dirs are
    pickle_file = open(project_dict['details']['base_dir']+'6_' + project_name + '_' + var_name + '_' + str(var_value) + '_post_setup_low_beta.pickle','w')
    pickle.dump(project_dict,pickle_file)
    pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
