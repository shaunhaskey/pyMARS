#!/usr/bin/env Python
import pyMARS.results_class as results_class
import pickle,sys
import numpy as num
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs

project_name = sys.argv[1]
upper_and_lower = int(sys.argv[2])

def perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP= 1.0e+3 * 3./num.pi):
    #print 'in perform_calcs'
    print directory, 'I0EXP=',I0EXP

    #I0EXP = RZfuncs.I0EXP_calc(N,n,I)
    new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP)
    #print 'results_class initialised'
    new_data_R = new_data.R*new_data.R0EXP
    new_data_Z = new_data.Z*new_data.R0EXP
    #print 'R and Z data obtained'
    new_answer = num.array(pyMARS_funcs.coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi,probe, probe_type, Rprobe,Zprobe,tprobe,lprobe))
    #print 'finished calculation'
    return new_answer

def coil_outputs_B(project_dict, upper_and_lower=0):
    probe = project_dict['details']['pickup_coils']['probe']
    probe_type = project_dict['details']['pickup_coils']['probe_type']
    Rprobe = project_dict['details']['pickup_coils']['Rprobe']
    Zprobe = project_dict['details']['pickup_coils']['Zprobe']
    tprobe = project_dict['details']['pickup_coils']['tprobe']
    lprobe = project_dict['details']['pickup_coils']['lprobe']
    link_RMZM = 0
    #Nchi = 240
    for i in project_dict['sims'].keys():
        project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
        Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
        print 'working on serial : ', i
        locs = ['upper','lower'] if upper_and_lower else ['']
        for loc in locs:
            for type in ['plasma', 'vacuum']:
                directory = project_dict['sims'][i]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
                project_dict['sims'][i]['{}_{}_response4'.format(type, loc)] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
#         if upper_and_lower == 1:
#             #print 'hello1'
#             directory = project_dict['sims'][i]['dir_dict']['mars_upper_vacuum_dir']
#             #print directory, Nchi, link_RMZM
#             #print 'starting vacuum_upper_response'
#             project_dict['sims'][i]['vacuum_upper_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
#             directory = project_dict['sims'][i]['dir_dict']['mars_lower_vacuum_dir']
#             project_dict['sims'][i]['vacuum_lower_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
#             directory = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
#             project_dict['sims'][i]['plasma_upper_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
#             directory = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']
#             project_dict['sims'][i]['plasma_lower_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
#         else:
#             #print 'hello2'
#             directory = project_dict['sims'][i]['dir_dict']['mars_vac_dir']
#             project_dict['sims'][i]['vacuum_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
#             directory = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
#             project_dict['sims'][i]['plasma_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
            
    return project_dict

#N = 6
#n = 2
#I = num.array([1.,-1.,0.,1,-1.,0.])

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()
print 'opened project_dict %d items'%(len(project_dict.keys()))
project_dict = coil_outputs_B(project_dict, upper_and_lower = upper_and_lower)
print 'finished calc'

output_name = project_name + 'output'
pickle_file = open(output_name,'w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()
print 'output file'
