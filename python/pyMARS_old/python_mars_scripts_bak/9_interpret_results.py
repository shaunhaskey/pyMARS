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
################################################################


pickle_file = open(project_dir + '6_'+ project_name + '_post_setup.pickle','r')
#pickle_file = open(project_dir + '7_' + project_name + '_post_mars_run.pickle')
project_dict = pickle.load(pickle_file)
pickle_file.close()

#dir = '/u/haskeysr/mars/project1/shot138344/tc_003/qmult0.60/exp0.63/marsrun/RUNrfa.p/'

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
                #project_dict['sims'][i]['plasma_response3'] = coil_responses3(r_array,z_array,Br,Bz,Bphi)
                project_dict['sims'][i]['plasma_response4'] = coil_responses4(r_array,z_array,Br,Bz,Bphi)

            else:
                project_dict['sims'][i]['vacuum_response1'] = coil_responses2(r_array,z_array,Br,Bz,Bphi)
                project_dict['sims'][i]['vacuum_response2'] = coil_responses_single(r_array,z_array,Br,Bz,Bphi)
                #project_dict['sims'][i]['vacuum_response3'] = coil_responses3(r_array,z_array,Br,Bz,Bphi)
                project_dict['sims'][i]['vacuum_response4'] = coil_responses4(r_array,z_array,Br,Bz,Bphi)

#                print 'vac', project_dict['sims'][i]['vacuum_response']

            R=2.34 #2.34
            Z=0.0
            #Br_val, Bz_val, Bphi_val, R_val, Z_val = find_r_z(r_array, z_array, R, Z, Br, Bz, Bphi)

            #Br_val_list.append(Br_val) 
            #Bz_val_list.append(Bz_val)
            #Bphi_val_list.append(Bphi_val)
            #R_val_list.append(R_val)
            #Z_val_list.append(Z_val)

            success+=1
            #difference_percent =10
            print 'succeeded!!! %d' %(success)#, R,Z req: %.4f,%.4f; R,Z rec:%.4f,%.4f'%(success, R, Z, R_val, Z_val)
            ##project_dict['sims'][i]['RESULTS']=[r_array, z_array, Br, Bz, Bphi]
            file_log = open('/u/haskeysr/calc_log.log','a')
            file_log.write('success !!!  %d\n'%(success))
            file_log.close()

        except:
            print 'sorry failed.... %d'%(fail)
            fail +=1
            file_log = open('/u/haskeysr/calc_log.log','a')
            file_log.write('sorry failed.... %d\n'%(fail))
            file_log.close()


pickle_file = open(project_dict['details']['base_dir']+'9_' + project_name + '_coil_outputs.pickle','w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()

print 'sucess %d, fail %d'%(success,fail)

