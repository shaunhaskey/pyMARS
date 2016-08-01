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

from PythonMARS_funcs import *
import time, os, sys, pickle

project_name = sys.argv[1]
print '########### Warning - This needs to be run on Benten !!!!!'
print '########### or a computer that has Matlab licenses'

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
matlab_RZplot_files = '/u/haskeysr/matlab/RZplot3/'
##################################

overall_start = time.time()

#Open previous data structure
pickle_file = open(project_dir + '4_'+project_name+'_post_fxrun_low_beta.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()


start_time = time.time()
total_jobs = len(project_dict['sims'].keys())
total_finished = 0

#Start the list of Matlab commands - this one allows Matlab to find the
#functions it needs from Yueqiang

mat_commands = "addpath('"+ matlab_RZplot_files + "')\n"


for i in project_dict['sims'].keys():
    print i

    #copy a MacMain.m script to the chease dir and modify to perform the job we require
    project_dict['sims'][i] = modify_RMZM_F2(project_dict['sims'][i])

    mat_commands += 'close all;clear all;\n'
    #Go to correct dir
    mat_commands += 'cd ' + project_dict['sims'][i]['dir_dict']['chease_dir']+'\n'
    #Record output to obtain results later
    mat_commands += "diary('testing_output')\n"
    mat_commands += "diary on\n"
    #Track what has been done
    mat_commands += "disp('Finished %d of %d')\n"%(total_finished, total_jobs)
    #Run the script that does the calculations
    mat_commands += "MacMainD3D_current\n"
    #Save the picture it creates so it can be viewed if required
    mat_commands += "print(gcf,'-dpng','p" +str(project_dict['sims'][i]['PMULT'])+'_q'+str(project_dict['sims'][i]['QMULT']) +".png')\n"
    mat_commands += "diary off\n"

    total_finished += 1
    print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)

mat_commands += "quit\n"

#Write the list of Matlab commands to a file
os.chdir(project_dict['details']['base_dir'])
file = open('matlab_commands.txt','w')
file.write(mat_commands)
file.close()

#Run Matlab
os.chdir(project_dict['details']['base_dir'])
#Not sure if this is the best way to run Matlab - may be better to use -r
os.system('matlab -nodesktop -nodisplay < matlab_commands.txt')

#Retrieve the results from the Matlab run
for i in project_dict['sims'].keys():
    print i
    project_dict['sims'][i] = RMZM_post_matlab(project_dict['sims'][i])


#Output the updated datastructure - now includes some MARS input parameters
#such as coil locations on the chease grid - this is different for each eq
pickle_file = open(project_dict['details']['base_dir']+'5_' + project_name + '_post_RMZM_low_beta.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
