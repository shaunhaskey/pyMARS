import pickle, sys
import numpy as num
from PythonMARS_funcs import *

project_name = 'project1_new_eq'

####################SET BEFORE STARTING##########################
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'

input_filename1 = '9_project1_new_eq_COIL_upper_post_setup_new_low_beta.pickle'
input_filename2 = '9_project1_new_eq_COIL_upper_post_setup_new.pickle'
output_filename = '9_project1_new_eq_COIL_upper_post_setup_new_low_beta2.pickle'

################################################################
pickle_file1 = open(project_dir + input_filename1,'r')
pickle_file2 = open(project_dir + input_filename2,'r')

project_dict1 = pickle.load(pickle_file1)
project_dict2 = pickle.load(pickle_file2)
pickle_file1.close()
pickle_file2.close()


key_list = project_dict2['sims'].keys()
start_serial = num.max(key_list) + 4
print 'maximum key : ',start_serial
serial = start_serial
for i in project_dict1['sims'].keys():
    project_dict2['sims'][serial]=project_dict1['sims'][i]
    print 'added ', i,' to serial ', serial
    serial += 1

pickle_file = open(project_dict2['details']['base_dir']+output_filename,'w')

pickle.dump(project_dict2, pickle_file)
pickle_file.close()

