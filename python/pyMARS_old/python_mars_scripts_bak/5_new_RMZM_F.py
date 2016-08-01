#!/usr/bin/env Python
# Generate the I-coil locations and wall location so that these values are available
# for MARS-F. The script basically generates a script for Matlab to run because
# Yueqiang's code is in Matlab. Matlab then outputs a file for each serial that includes
# all the relevant results. The next step is to read in these values into the datastructure
# so that they can be put in the MARS run file etc...
# This step needs to be run on Benten for it to work because it NEEDS Matlab
# Note that it adds /u/haskeysr/matlab/RZplot3 to the path variable in Matlab so that it can
# access the relevant files. Change this directory to 

## Future work : implement all of this in Python so that this is simpler

from RZfuncs import *
import time, os, sys, pickle

project_name = sys.argv[1]

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
coilN  = num.array([[2.164, 1.012, 2.374, 0.504],[2.164, -1.012, 2.374, -0.504]])
Nchi = 240
##################################

overall_start = time.time()

#Open previous data structure
pickle_file = open(project_dir + '4_'+project_name+'_post_fxrun_low_beta.pickle','r')
pickle_file = open(project_dir + '9_project1_new_eq_COIL_upper_post_setup.pickle','r')

project_dict = pickle.load(pickle_file)
pickle_file.close()

#monitor progress

def RMZM_func(project_dict, coilN, RMZMFILE, Nchi):
    start_time = time.time()
    total_jobs = len(project_dict['sims'].keys())
    total_finished = 0
    for i in project_dict['sims'].keys():
        RMZMFILE = project_dict['sims'][i]['dir_dict']['chease_dir']+'RMZM_F'
        FCCHI, FWCHI, IFEED = Icoil_MARS_grid_details(coilN,RMZMFILE,Nchi)
        total_finished+=1
        FCCHI_old =  project_dict['sims'][i]['FCCHI']
        FWCHI_old = project_dict['sims'][i]['FWCHI']
        IFEED_old = project_dict['sims'][i]['IFEED']
        print i, ' FCCHI=',FCCHI, ' FWCHI=',FWCHI,' IFEED=',IFEED,' time:',time.time()-overall_start,'s, finished ',total_finished,' of ',total_jobs
        print i, ' FCCHI=',FCCHI_old, ' FWCHI=',FWCHI_old,' IFEED=',IFEED_old,' time:',time.time()-overall_start,'s, finished ',total_finished,' of ',total_jobs
        project_dict['sims'][i]['FCCHI'] = FCCHI
        project_dict['sims'][i]['FWCHI'] = FWCHI
        project_dict['sims'][i]['IFEED'] = IFEED
    return project_dict

project_dict = RMZM_func(project_dict, coilN, RMZMFILE, Nchi)
#Output the updated datastructure - now includes some MARS input parameters
#such as coil locations on the chease grid - this is different for each eq
pickle_file = open('new_file_test.pickle','w')
#pickle_file = open(project_dict['details']['base_dir']+'5_' + project_name + '_post_RMZM_low_beta.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
