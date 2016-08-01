#!/usr/bin/env Python
import pickle, sys
import numpy as num
from PythonMARS_funcs import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pt

project_name = sys.argv[1]

####################SET BEFORE STARTING##########################
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
var_name = 'ROTE'
var_name = 'FEEDI'
var_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
var_list = num.arange(0,1.55,0.05)
var_list = num.arange(0.8,1,0.01) #ROTE3
var_list = [0]#[-300]#[-120]#[-180]#[-300] #[-300] #Vary 

################################################################


for var_value in var_list:
    pickle_file = open(project_dir + '6_'+ project_name + '_' + var_name + '_' + str(var_value) + '_post_setup.pickle','r')

#    pickle_file = open(project_dir + '7_' + project_name + '_' + var_name + '_' + str(var_value) + '_post_mars_run.pickle')
    project_dict = pickle.load(pickle_file)
    pickle_file.close()

    #extract data:
    Br_val_list = []
    Bz_val_list = []
    Bphi_val_list = []
    R_val_list = []
    Z_val_list = []
    success = 0
    fail = 0

    for i in project_dict['sims'].keys():
        for type in ['plasma', 'vac']:
            try:
                if type == 'plasma':
                    dir = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
                else:
                    dir = project_dict['sims'][i]['dir_dict']['mars_vac_dir']

                r_array, z_array = post_mars_r_z(dir)
                r_array = r_array * project_dict['sims'][i]['R0EXP']
                z_array = z_array * project_dict['sims'][i]['R0EXP']
                Br = extractB(dir,'Br')
                Bz = extractB(dir,'Bz')
                Bphi = extractB(dir,'Bphi')

                if type == 'plasma':
                    project_dict['sims'][i]['plasma_response1'] = coil_responses2(r_array,z_array,Br,Bz,Bphi)
                    project_dict['sims'][i]['plasma_response2'] = coil_responses_single(r_array,z_array,Br,Bz,Bphi)
#                    project_dict['sims'][i]['plasma_response3'] = coil_responses3(r_array,z_array,Br,Bz,Bphi)
                    project_dict['sims'][i]['plasma_response4'] = coil_responses4(r_array,z_array,Br,Bz,Bphi)
                else:
                    project_dict['sims'][i]['vacuum_response1'] = coil_responses2(r_array,z_array,Br,Bz,Bphi)
                    project_dict['sims'][i]['vacuum_response2'] = coil_responses_single(r_array,z_array,Br,Bz,Bphi)
#                    project_dict['sims'][i]['vacuum_response3'] = coil_responses3(r_array,z_array,Br,Bz,Bphi)
                    project_dict['sims'][i]['vacuum_response4'] = coil_responses4(r_array,z_array,Br,Bz,Bphi)

                success+=1
                text = 'succeeded!!! %d\n' %(success)
                print text

            except:
                text = 'sorry failed.... %d\n'%(fail)
                print text
                fail +=1
            file_log = open(project_dir + '/calc_' + var_name + '_' + str(var_value) + '_log','a')
            file_log.write(text)
            file_log.close()
            
    pickle_file = open(project_dict['details']['base_dir']+'9_' + project_name + '_' + var_name + '_' + str(var_value) + '_coil_outputs.pickle','w')
    pickle.dump(project_dict, pickle_file)
    pickle_file.close()

    print 'sucess %d, fail %d'%(success,fail)
