#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, os, sys, pickle

overall_start = time.time()

project_name = sys.argv[1]
pickle_file = open('/u/haskeysr/mars/'+ project_name +'/7_post_mars_run.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()


start_time = time.time()
total_jobs = len(project_dict['sims'].keys())*2
total_finished = 0

mat_commands = "addpath('/u/haskeysr/matlab/RZplot3/')\n"

for i in project_dict['sims'].keys():
    print i
    #Copy matlab file and change relevant variables - to plasma dir
    for type in ['plasma', 'vac']:

        if type == 'plasma':
            os.chdir(project_dict['sims'][i]['dir_dict']['mars_plasma_dir'])
        else:
            os.chdir(project_dict['sims'][i]['dir_dict']['mars_vac_dir'])

        os.system('cp ~/mars/templates/Extract_Results.m Extract_Results_Current.m')

        if type == 'plasma':
            SDIR_newline = "SDIR='"+ project_dict['sims'][i]['dir_dict']['mars_plasma_dir']+"';"
        else:
            SDIR_newline = "SDIR='"+ project_dict['sims'][i]['dir_dict']['mars_vac_dir']+"';"
            
        modify_input_file('Extract_Results_Current.m', 'SDIR=', SDIR_newline)
        replace_value('Extract_Results_Current.m','Mac.Nm2', ';', str(1+int(abs(project_dict['sims'][i]['M1'])+abs(project_dict['sims'][i]['M2']))))
        #IFEED needs to be changed - how to do this???!!!

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

        total_finished += 1
        print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)

mat_commands += "quit\n"
os.chdir(project_dict['details']['base_dir'])
file = open('mat_post_mars_commands.txt','w')
file.write(mat_commands)
file.close()

#os.chdir(project_dict['details']['base_dir'])

os.system('matlab -nodesktop -nodisplay < mat_post_mars_commands.txt')
