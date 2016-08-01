#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, os, sys, pickle
import numpy as num
overall_start = time.time()

project_name = sys.argv[1]

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
var_name = 'ROTE'
var_name = 'FEEDI'
var_value = 0
var_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5] #ROTE1
var_list = num.arange(0,1.55,0.05) #ROTE2
var_list = num.arange(0.8,1,0.01) #ROTE3
var_list = [0]# [-60]#[-300]#[-120]#[-180]#[-300] #Vary 

################################

plot_number = 1

mat_commands = "addpath('/u/haskeysr/matlab/RZplot3/')\n"
total_finished = 0

for var_value in var_list:
    print '********************* var value : %.3f ****************'%(var_value)
    pickle_file = open(project_dir +'7_' + project_name + '_' + var_name + '_giant_file_post_mars_run.pickle','r')

    pickle_file = open(project_dir + '6_'+ project_name + '_' + var_name + '_' + str(var_value) + '_post_setup.pickle','r')
#    pickle_file = open(project_dir + '7_'+ project_name + '_' + var_name + '_' + str(var_value) + '_post_mars_run.pickle','r')
    project_dict = pickle.load(pickle_file)
    pickle_file.close()


    start_time = time.time()
    total_jobs = len(project_dict['sims'].keys())*2


    for i in project_dict['sims'].keys():
        print i
        #Copy matlab file and change relevant variables - to plasma dir
        for type in ['plasma', 'vac']:

            if type == 'plasma':
                os.chdir(project_dict['sims'][i]['dir_dict']['mars_plasma_dir'])
            else:
                os.chdir(project_dict['sims'][i]['dir_dict']['mars_vac_dir'])

            os.system('cp ~/mars/templates/Extract_Results.m Extract_Results_Current.m')
            os.system('cp ~/mars/templates/View_Results.m View_Results.m')

            if type == 'plasma':
                SDIR_newline = "SDIR='"+ project_dict['sims'][i]['dir_dict']['mars_plasma_dir']+"';"
            else:
                SDIR_newline = "SDIR='"+ project_dict['sims'][i]['dir_dict']['mars_vac_dir']+"';"

            modify_input_file('Extract_Results_Current.m', 'SDIR=', SDIR_newline)
            modify_input_file('View_Results.m', 'SDIR=', SDIR_newline)

            replace_value('View_Results.m','Mac.Nm2', ';', str(1+int(abs(project_dict['sims'][i]['M1'])+abs(project_dict['sims'][i]['M2']))))
            replace_value('Extract_Results_Current.m','Mac.Nm2', ';', str(1+int(abs(project_dict['sims'][i]['M1'])+abs(project_dict['sims'][i]['M2']))))
            replace_value('View_Results.m','Mac.plot_Bn', ';', str(plot_number))


            mat_commands += 'close all;clear all;\n'

            if type == 'plasma':
                mat_commands += 'cd ' + project_dict['sims'][i]['dir_dict']['mars_plasma_dir']+'\n'
            else:
                mat_commands += 'cd ' + project_dict['sims'][i]['dir_dict']['mars_vac_dir']+'\n'

            mat_commands += "diary('output_data.tmp')\n"
            mat_commands += "diary on\n"
            mat_commands += "disp('Finished %d of %d')\n"%(total_finished, total_jobs)
            mat_commands += "Extract_Results_Current\n"
            mat_commands += "diary off\n"
            plot_number += 1
            total_finished += 1
            print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)

mat_commands += "quit\n"
os.chdir(project_dict['details']['base_dir'])
file = open('mat_post_mars_commands.txt','w')
file.write(mat_commands)
file.close()

os.system('matlab -nodesktop -nodisplay < mat_post_mars_commands.txt')
