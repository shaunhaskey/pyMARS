#!/usr/bin/env Python
import pyMARS.results_class as results_class
import pickle,sys
import numpy as np
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs


project_name = '/u/haskeysr/mars/shot153585_03795_q95_scan_josh_q95_scan/shot153585_03795_q95_scan_josh_q95_scan_post_processing_PEST.pickle'
pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()
print 'opened project_dict %d items'%(len(project_dict.keys()))


link_RMZM = 0
#for i in project_dict['sims'].keys():
results = {}
for i in [1,2]:
    results[i] = {}
    n = np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
    I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
    Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
    print 'working on serial : ', i
    locs = ['upper','lower']
    for loc in locs:
        print directory, 'I0EXP=',I0EXP
        for type in ['plasma', 'vacuum']:
            directory = project_dict['sims'][i]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
            new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
            new_data_R = new_data.R*new_data.R0EXP
            new_data_Z = new_data.Z*new_data.R0EXP
            results[i]['{}_{}'.format(type, loc)] = 1


# project_dict = coil_outputs_B(project_dict, upper_and_lower = upper_and_lower)
# print 'finished calc'

# output_name = project_name + 'output'
# pickle_file = open(output_name,'w')
# pickle.dump(project_dict, pickle_file)
# pickle_file.close()
# print 'output file'
