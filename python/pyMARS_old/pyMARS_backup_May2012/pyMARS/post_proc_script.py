#!/usr/bin/env Python
import results_class
import pickle,sys
import numpy as num
import PythonMARS_funcs as pyMARS

project_name = sys.argv[1]


def coil_outputs_B(project_dict):
    probe = project_dict['details']['pickup_coils']['probe']
    probe_type = project_dict['details']['pickup_coils']['probe_type']
    Rprobe = project_dict['details']['pickup_coils']['Rprobe']
    Zprobe = project_dict['details']['pickup_coils']['Zprobe']
    tprobe = project_dict['details']['pickup_coils']['tprobe']
    lprobe = project_dict['details']['pickup_coils']['lprobe']
    for i in project_dict['sims'].keys():
        print 'working on serial : ', i
        for type in ['plasma', 'vac']:
            if type == 'plasma':
                dir = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
            else:
                dir = project_dict['sims'][i]['dir_dict']['mars_vac_dir']
            print i, type
            new_data = results_class.data(dir,Nchi=240,link_RMZM=0)
            new_data_R = new_data.R*new_data.R0EXP
            new_data_Z = new_data.Z*new_data.R0EXP
            new_answer = num.array(pyMARS.coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi,probe, probe_type, Rprobe,Zprobe,tprobe,lprobe))

            #comp_answer = new_answer[0:-2]
            comp_answer = new_answer *1
            if type == 'plasma':
                project_dict['sims'][i]['plasma_response4'] = new_answer
            else:
                project_dict['sims'][i]['vacuum_response4'] = new_answer
            del new_data
    return project_dict

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

project_dict = coil_outputs_B(project_dict)

output_name = project_name + 'output'
pickle_file = open(output_name,'w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()
