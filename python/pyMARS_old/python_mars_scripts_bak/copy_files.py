import numpy as num
import time, os, sys, string, re, csv, pickle
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata



################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/project1_new_eq/'
pickle_file_name = '6_project1_new_eq_COIL_upper_post_setup.pickle'
new_project_dir = '/u/haskeysr/mars/project1_new_eq/'
folders = 'RUNrfa_COILupper.p RUNrfa_COILupper.vac RUNrfa_COILlower.p RUNrfa_COILlower.vac '
##################################

#Open previous data structure 
pickle_file = open(pickle_file_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

original_base = project_dict['details']['base_dir']

total = len(project_dict['sims'].keys())
count = 0
for current in project_dict['sims'].keys():
    original_mars_dir = project_dict['sims'][current]['dir_dict']['mars_dir']
    os.chdir(original_mars_dir)
    new_dir = new_project_dir + original_mars_dir[len(original_base):]
    string_command = 'cp -r '+folders+new_dir
    os.system(string_command)
    print 'finished %d of %d'%(count,total)
    count += 1
