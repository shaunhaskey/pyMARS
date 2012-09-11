from PythonMARS_funcs import *
import time, os, sys, pickle
import numpy as num
import results_class

master_pickle_name = '9_project1_new_eq_COIL_upper_post_setup_new_low_beta2.pickle'
master_pickle_name = '9_project1_new_eq_COIL_lower_post_setup_new_low_beta2.pickle'

project_name = 'project1_new_eq'
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
master_pickle = pickle.load(open(project_dir + master_pickle_name,'r'))

pickle_file_list = []

start_time = time.time()
workers = 30
serial_list = master_pickle['sims'].keys()
increment_base = len(serial_list)/workers
marker1 = 0
worker = 1
odd = 1
for i in range(0,workers):
    if odd ==1:
        increment=increment_base + 1
        odd=0
    else:
        increment = increment_base
        odd =1
    if worker!=workers:
        marker2 = marker1+increment
    else:
        marker2 = len(serial_list)+1
    current_serials = serial_list[marker1:marker2]
    print len(serial_list),marker1,marker2
    tmp_pickle = {}
    tmp_pickle['details']=master_pickle['details']
    tmp_pickle['sims']={}
    for jjj in current_serials:
        tmp_pickle['sims'][jjj]=master_pickle['sims'][jjj]
    print 'temp pickle length : ', len(tmp_pickle['sims'].keys())

    pickle_file_name = 'tmp_'+str(i)+'.pickle'
    project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
    pickle_file = project_dir + pickle_file_name

    pickle.dump(tmp_pickle,open(pickle_file,'w'))
    print 'dumped ', pickle_file
    marker1=marker2

    job_name = 'step9_'+str(i)+'.job'

    job_string = '#!/bin/bash\n#$ -N blah\n#$ -q all.q\n#$ -o sge_output.dat\n#$ -e sge_error.dat\n#$ -cwd\nexport PATH=$PATH:/f/python/linux64/bin\n'
    job_file = open(job_name,'w')
    log_file_name = 'log_test'+str(i)+'.log'
    execute_command = '/u/haskeysr/python_mars_scripts/8_new_post_proc.py '+pickle_file+' > '+log_file_name+'\n'
    job_string += execute_command
    job_file.write(job_string)
    job_file.close()
    os.system('rm ' +pickle_file+'output')
    pickle_file_list.append(pickle_file)
    #os.system('qsub ' + job_name)
    worker += 1


time.sleep(10)
for i in pickle_file_list:
    while os.path.exists(i+'output')!=True:
        print 'waiting for ',i
        time.sleep(5)
    print 'finished ', i


combined_answer = {}
combined_answer['details']= master_pickle['details']
combined_answer['sims']={}
for i in pickle_file_list:
    current_tmp = pickle.load(open(i))
    #print current_tmp.keys()
    #print current_tmp['sims'].keys()
    current_serials = current_tmp['sims'].keys()
    for jjj in current_serials:
        combined_answer['sims'][jjj]=current_tmp['sims'][jjj]


for i in pickle_file_list:
    os.system('rm ' + i)
    os.system('rm ' + i +'output')
    



pickle_file_comb = project_dir + 'combined_lower.pickle'
pickle.dump(combined_answer,open(pickle_file_comb,'w'))
print 'total time : ', (time.time()-start_time)/60.,'mins'
