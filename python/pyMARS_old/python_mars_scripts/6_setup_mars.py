#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, os, sys, pickle

print 'running mars setup section'

overall_start = time.time()
project_name = sys.argv[1]
pickle_file = open('/u/haskeysr/mars/'+ project_name +'/5_post_RMZM.pickle','r')

project_dict = pickle.load(pickle_file)
pickle_file.close()

#name_of_variable = 'ROTE'
#variable_values = [1,2,3,4,5]

#project_dict['sims'][i]['mars']={}


for i in project_dict['sims'].keys():
    print i
    project_dict['sims'][i] = extract_NW(project_dict['sims'][i])

    #Mars vac
    mars_setup_files(project_dict['sims'][i], vac = 1)
    project_dict['sims'][i] = mars_setup_alfven(project_dict['sims'][i], project_dict['sims'][i]['ICOIL_FREQ'], vac = 1)

    mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 1)
    generate_job_file(project_dict['sims'][i],1)

    #Plasma setup
    mars_setup_files(project_dict['sims'][i], vac = 0)
    mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 0)
    generate_job_file(project_dict['sims'][i],0)
    
pickle_file = open(project_dict['details']['base_dir']+'6_post_mars_setup.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()


print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
