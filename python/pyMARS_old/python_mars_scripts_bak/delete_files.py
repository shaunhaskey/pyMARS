import numpy as num
import time, os, sys, string, re, csv, pickle
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata

################ SET THESE BEFORE RUNNING!!!!########
project_dir = '/scratch/haskeysr/mars/project1_new_eq/'
pickle_file_name = '9_project1_new_eq_COIL_upper_post_setup_new_low_beta2.pickle'
#folders = 'RUNrfa_FEEDI-120.p RUNrfa_FEEDI-120.vac RUNrfa_FEEDI-240.p RUNrfa_FEEDI-240.vac RUNrfa_FEEDI-60.p RUNrfa_FEEDI-60.vac'
folders = 'OUTRMAR OUTVMAR'
##################################

#Open previous data structure 
pickle_file = open(pickle_file_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

total = len(project_dict['sims'].keys())
count = 0
fails = 0
for current in project_dict['sims'].keys():
    original_chease_dir = project_dict['sims'][current]['dir_dict']['chease_dir']
    chease_pest_dir = original_chease_dir[:-1]+'_PEST/'
    print chease_pest_dir
    try:
        os.chdir(chease_pest_dir)
        string_command = 'rm '+ folders
        os.system(string_command)
    except:
        fails += 1
        print 'dir not found'
        
    print 'finished %d of %d,fails %d'%(count,total,fails)
    count += 1
