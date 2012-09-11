#!/usr/bin/python
from PythonMARS_funcs import *
import time
import os
import sys
import pickle
shot_input = sys.argv[1]

#--- List of user defined variables ---------
base_dir = '/u/haskeysr/mars/tutorial/'
template_dir = '/u/haskeysr/mars/tutorial/'
efit_master = '/u/haskeysr/mars/tutorial/shot135762/efit'

corsica_run = 0
chease_run = 0
fxrun_run = 0
mars_vac_run = 0
mars_plasma_run = 0

master={}
master['shot']=int(shot_input)
master['PMULT']=1.0
master['QMULT']=1.0
master['shot_time']=1805
master['thetac']=0.005
master['EXPEQ_name']='EXPEQ_135762.01805_p090'

start = time.time()

#----------- Copy EFIT files and generate directories --------
print 'Creating all required Directories'
master = generate_directories(master,base_dir)
issue_command('cp ', efit_master + '/* ' + master['dir_dict']['efit_dir'])
print 'Finished creating Directories'
directory_time = time.time() - start

#----------- Run Corsica --------------------------
if corsica_run == 1:
    print 'Corsica Section'
    modify_stab_file(master, template_dir)
    run_corsica(master)
corsica_time = time.time() - directory_time - start
print 'Start Chease section'

#----------- Run Chease -------------------
if chease_run ==1:
    master = extract_R0_B0(master)
    copy_chease_files(master, base_dir)
    modify_datain(master,base_dir)
    execute_chease(master)
    master['chease_run']=1
chease_time = time.time() - corsica_time - start

#-----------Run fxrun --------------------
if fxrun_run == 1:
    master = extract_R0_B0(master)
    master = extract_NW(master)
    fxin_create(master)
    fxrun(master)
    master['fxrun']=1
fxrun_time = time.time() - chease_time - start

#----------Run mars_vac ------------------
if mars_vac_run == 1:
    master = extract_NW(master)
    master = extract_R0_B0(master)
    mars_vac_setup_files(master)
    mars_vac_setup_run_file(master, template_dir)
    mars_vac_run_func(master)
    master['mars_vac_run']=1
mars_vac_time = time.time() - fxrun_time - start

#----------Run mars plasma ------------
if mars_plasma_run == 1:
    master = extract_NW(master)
    master = extract_R0_B0(master)
    mars_plasma_setup_files(master)
    mars_plasma_setup_run_file(master,template_dir)
    mars_plasma_run_func(master)
    master['mars_plasma_run']=1
mars_plasma_time = time.time() - mars_vac_time - start

os.chdir(base_dir)
f = open('test_dump.pickle','w')
pickle.dump(master,f)
f.close()


print 'Directory Time :%.2fs'%(directory_time)
print 'Corsica Time :%.2fs'%(corsica_time)
print 'Chease Time :%.2fs'%(chease_time)
print 'fxrun Time :%.2fs'%(fxrun_time)
print 'Mars Vac Time :%.2fmins'%(mars_vac_time/60)
print 'Mars Plasma Time :%.2fmins'%(mars_plasma_time/60)
print 'Total Time : %.2fmins'%((time.time()-start)/60)
