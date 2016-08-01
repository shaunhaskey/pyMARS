#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, os, sys, pickle
import numpy as num
print 'running mars setup section'

overall_start = time.time()
project_name = sys.argv[1]

var_name2 = 'ICOIL_FREQ'
var_list2 = [0]

#This is a special case to set ROTE to 1., and FEEDI to -300deg
######## This is also where I can change IFEED.... #########
################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
var_name = 'FEEDI'
var_list = [-300]#[-120]#[-180]#[-300] #[0, -60, -120, -180, -240, -300]
######### Need to run this file for ever var_value you want to use
######## and continue with the separate names all the way through
######## Need to automate this a bit better!
##################################

def construct_FEEDI(phase):
    real_part = num.cos(phase/360.*2*num.pi)
    imag_part = num.sin(phase/360.*2*num.pi)

    FEEDI_string =  '(1.0,0.0),(%.5f, %.5f),'%(real_part, imag_part)
    return FEEDI_string


for var_value in var_list:
    pickle_file = open(project_dir + '5_'+project_name+'_post_RMZM.pickle','r')
    project_dict = pickle.load(pickle_file)
    pickle_file.close()

    FEEDI_string = construct_FEEDI(var_value)
    #Change section in details
    project_dict['details']['FEEDI'] = FEEDI_string

    total_jobs = len(project_dict['sims'].keys())
    completed = 0

    for i in project_dict['sims'].keys():
        print i
        
        #create new directories and update mars_run dirs - important modification!!!!
        vac_name = 'RUNrfa_' + var_name+str(var_value)+'_'+ var_name2 + '_' + str(var_list2[0]) + '.vac'
        plasma_name = 'RUNrfa_' + var_name+str(var_value)+'_'+ var_name2 + '_' + str(var_list2[0]) +'.p'

        project_dict['sims'][i] = mars_directories(project_dict['sims'][i], vac_name, plasma_name)
        project_dict['sims'][i] = extract_NW(project_dict['sims'][i])

        #Mars vac
        mars_setup_files(project_dict['sims'][i], 1) #link necessary files
        mars_setup_files(project_dict['sims'][i], 0)

        #Calculate parameters in terms of Alfven frequency
        project_dict['sims'][i]['ICOIL_FREQ'] = var_list2[0] # Modification to make ICOIL freq 0

        project_dict['sims'][i] = mars_setup_alfven(project_dict['sims'][i], project_dict['sims'][i]['ICOIL_FREQ'], vac = 1)

        #MAKE PHASING MODIFICATION - this is all we need to do as this is setup in the mars_setup_run_file function
        project_dict['sims'][i]['FEEDI'] = FEEDI_string

        mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 1)
        mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 0)

        #MAKE ROTATION MODIFICATION
#        project_dict['sims'][i]['ROTE'] = project_dict['sims'][i]['ROTE'] * var_list2[0]
#        print 'ROTE value %.5f'%(project_dict['sims'][i]['ROTE'])

#        mars_setup_run_file_special(project_dict['sims'][i], var_name2, project_dict['sims'][i]['ROTE'], 1)
#        mars_setup_run_file_special(project_dict['sims'][i], var_name2,  project_dict['sims'][i]['ROTE'], 0)
        #--------------end modification


        generate_job_file(project_dict['sims'][i],1)
        generate_job_file(project_dict['sims'][i],0)

        completed+=1
        print 'completed : %d of %d'%(completed, total_jobs)

    output_name = '6_post_mars_setup_' + var_name + str(var_value) + '.pickle'
    pickle_file = open(project_dict['details']['base_dir']+'6_' + project_name + '_' + var_name + '_' + str(var_value) + '_' + var_name2 + '_' + str(var_list2[0]) + '_post_setup.pickle','w')
    pickle.dump(project_dict,pickle_file)
    pickle_file.close()


print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
