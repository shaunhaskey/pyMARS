#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, os, sys, pickle

project_name = sys.argv[1]


# at the moment this needs to be run on benten - IS THIS TRUE????

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##################################

print 'running mars setup section'
overall_start = time.time()

pickle_file = open(project_dir + '4_'+project_name+'_post_fxrun.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

start_time = time.time()
total_jobs = len(project_dict['sims'].keys())
total_finished = 0

mat_commands = "addpath('/u/haskeysr/matlab/RZplot3/')\n"

for i in project_dict['sims'].keys():
    print i
    project_dict['sims'][i] = modify_RMZM_F2(project_dict['sims'][i])

    mat_commands += 'close all;clear all;\n'
    mat_commands += 'cd ' + project_dict['sims'][i]['dir_dict']['chease_dir']+'\n'
    mat_commands += "diary('testing_output')\n"
    mat_commands += "diary on\n"
    mat_commands += "disp('Finished %d of %d')\n"%(total_finished, total_jobs)
    mat_commands += "MacMainD3D_current\n"
    mat_commands += "print(gcf,'-dpng','p" +str(project_dict['sims'][i]['PMULT'])+'_q'+str(project_dict['sims'][i]['QMULT']) +".png')\n"
    mat_commands += "diary off\n"

    total_finished += 1
    print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)

mat_commands += "quit\n"
os.chdir(project_dict['details']['base_dir'])
file = open('matlab_commands.txt','w')
file.write(mat_commands)
file.close()

#Run Matlab:
os.chdir(project_dict['details']['base_dir'])
os.system('matlab -nodesktop -nodisplay < matlab_commands.txt')

for i in project_dict['sims'].keys():
    print i
    project_dict['sims'][i] = RMZM_post_matlab(project_dict['sims'][i])



pickle_file = open(project_dict['details']['base_dir']+'5_' + project_name + '_post_RMZM.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
