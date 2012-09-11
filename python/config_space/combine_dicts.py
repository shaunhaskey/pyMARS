'''
Take a dictionary with only upper and only lower values in it and combine them
Note this is outdated now that the MARS running scripts can do an upper and lower
in the same simulation
'''

import pickle
import copy


name_upper = '9_project1_new_eq_COIL_upper_post_setup_new_low_beta2.pickle'
name_lower = '9_project1_new_eq_COIL_lower_post_setup_new_low_beta2.pickle'
name_comb = 'n2_combined.pickle'

project_dict_upper = pickle.load(open(name_upper))
project_dict_lower = pickle.load(open(name_lower))

project_dict_comb = copy.deepcopy(project_dict_upper)

for current_serial in project_dict_upper['sims'].keys():
    if project_dict_upper['sims'][current_serial]['EXPEQ_name'] == project_dict_upper['sims'][current_serial]['EXPEQ_name']:
        print 'pass',
    else:
        print 'fail'

    tmp_upper = copy.deepcopy(project_dict_upper['sims'][current_serial])
    tmp_lower = copy.deepcopy(project_dict_lower['sims'][current_serial])
    del project_dict_comb['sims'][current_serial]
    project_dict_comb['sims'][current_serial]={}
    project_dict_comb['sims'][current_serial]['upper']=tmp_upper
    project_dict_comb['sims'][current_serial]['lower']=tmp_lower

comb_file = open(name_comb,'w')
pickle.dump(project_dict_comb, comb_file)
comb_file.close()

