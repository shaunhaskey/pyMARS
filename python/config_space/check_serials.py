'''
check that the serials in two different pickle files match up - i.e you are comparing the same eq
with upper and lower coils only
'''

import pickle, time
import numpy as num

name = '9_project1_new_eq_COIL_upper_post_setup_new_low_beta2.pickle'
name2 = '9_project1_new_eq_COIL_lower_post_setup_new_low_beta2.pickle'
project_dict = pickle.load(open(name))
project_dict2 = pickle.load(open(name2))
fails = 0
success = 0
duplicate = 0
for current in project_dict['sims'].keys():
    a = project_dict['sims'][current]['QMULT'],project_dict['sims'][current]['PMULT']
    b = project_dict2['sims'][current]['QMULT'],project_dict2['sims'][current]['PMULT']
    print a,b
    if a!=b:
        print 'fail!'
        fails +=1
    else:
        print 'success!'
        success +=1
    for current2 in project_dict['sims'].keys():
        c = project_dict['sims'][current2]['QMULT'],project_dict['sims'][current2]['PMULT']
        if c==b:
            print 'duplicate!'
            duplicate +=1

print 'fails %d, success %d, duplicate %d'%(fails,success,duplicate)
