#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, os, sys, pickle
import numpy as num
import results_class
overall_start = time.time()

project_name = sys.argv[1]

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'

input_filename = '9_project1_new_eq_COIL_upper_post_setup_low_beta.pickle'
input_filename = '9_project1_new_eq_COIL_upper_post_setup.pickle'
################################


#pickle_file = open(project_dir + input_filename,'r')
pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

start_time = time.time()


def coil_outputs_B(project_dict,serial_list):
    fails = 0
    passes = 0
    total_finished = 0
    total_jobs = len(serial_list)*2
    for i in serial_list:
        for type in ['plasma', 'vac']:
            if type == 'plasma':
                dir = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
                old_answer = num.array(project_dict['sims'][i]['plasma_response4'])
            else:
                dir = project_dict['sims'][i]['dir_dict']['mars_vac_dir']
                old_answer = num.array(project_dict['sims'][i]['vacuum_response4'])
            print i, type
            new_data = results_class.data(dir,Nchi=240,link_RMZM=0)
            r_array, z_array = post_mars_r_z(dir)
            r_array = r_array * project_dict['sims'][i]['R0EXP']
            z_array = z_array * project_dict['sims'][i]['R0EXP']
            new_data_R = new_data.R*new_data.R0EXP
            new_data_Z = new_data.Z*new_data.R0EXP

            diff_r = num.abs(new_data_R-r_array)/num.abs(r_array)*100
            diff_z = num.abs(new_data_Z-z_array)/num.abs(z_array)*100
            #print 'r_diff :',num.mean(diff_r),' max: ', num.max(diff_r)
            #print 'z_diff :',num.mean(diff_z),' max: ', num.max(diff_z)
            new_answer = num.array(coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi))
            
            #comp_answer = new_answer[0:-2]
            comp_answer = new_answer *1
            abs_dif = num.abs(comp_answer-old_answer)/num.abs(old_answer)*100
            real_dif = num.abs(num.real(comp_answer-old_answer)/num.real(old_answer))*100
            imag_dif = num.abs(num.imag(comp_answer-old_answer)/num.imag(old_answer))*100
            print num.sum(num.abs(abs_dif)),num.sum(num.abs(real_dif)),num.sum(num.abs(imag_dif))
            if num.sum(num.abs(abs_dif))>5 or num.sum(num.abs(real_dif))>5 or num.sum(num.abs(imag_dif))>5:
                fails += 1
                print comp_answer
                print old_answer
                print abs_dif
            else:
                passes += 1
            print 'fails :', fails, ' passes :', passes
            if type == 'plasma':
                project_dict['sims'][i]['plasma_response4'] = new_answer
            else:
                project_dict['sims'][i]['vacuum_response4'] = new_answer

            del new_data
    return project_dict

serial_list = project_dict['sims'].keys()
project_dict = coil_outputs_B(project_dict,serial_list)

output_name = project_name + 'output'
pickle_file = open(output_name,'w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()
