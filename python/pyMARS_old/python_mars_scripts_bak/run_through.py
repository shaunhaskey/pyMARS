#!/usr/bin/env Python
#from PythonMARS_funcs import *
#Step 1 : Create the base directories, and run Corsica
#Step 2 : Create the whole directory structure based on Corsica output
#Step 3 : Run chease on the Corsica output
#Step 4 : Modify the Chease output (as required by Yueqiang)
#Step 5 : Something to do with RMZM
#Step 6 : Setup for the MARS run
#Step 7 : Run MARS
#Step 8 : Post process MARS


import PythonMARS_funcs as pyMARS
import control_funcs as cont_funcs
import Chease_Batch_Launcher as ch_launch
import pickle, time, os
import numpy as num
import Corsica_funcs as Corsica_funcs

#########################################################
#########################################################
############ SET THESE VALUES BEFORE RUNNING!!!!! #######

#starting and finishing step variables
start_from_step = 1
end_at_step = 10

project_name = 'testing_new_code2'
base_directory = '/scratch/haskeysr/mars/'
efit_file_location = '/u/haskeysr/mars/eq_from_matt/efit_files/'
template_directory = '/u/haskeysr/mars/templates/'
post_proc_script = '/u/haskeysr/python_mars_scripts/post_proc_script.py'


#Cluster related variables
CHEASE_simultaneous_jobs = 6
MARS_simultaneous_jobs = 10
post_proc_simultaneous_jobs = 5

include_chease_PEST_run = 1

os.system('mkdir /scratch/haskeysr/mars/'+ project_name)
project_dict={}
project_dict['details']={}
project_dict['details']['thetac'] = 0.003 #somewhat redundant, used to setup directory structure
project_dict['details']['shot'] = 138344 #somewhat redundant, used to setup directory structure
project_dict['details']['M1'] = -29 #poloidal harmonics
project_dict['details']['M2'] = 29 #poloidal harmonics
project_dict['details']['shot_time'] = 2306 #
project_dict['details']['NTOR'] = 2 #toroidal mode number
project_dict['details']['FEEDI']= '(1.0,0.0),(1.0, 0.0),' #'(1.0,0.0),(-0.5, 0.86603),' #Icoil phasing
project_dict['details']['ICOIL_FREQ'] = 20 #Icoil frequency

# Bn_Div_Li_range = [0.0, 3]
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'

# filters out equilibria that DCON finds unstable (1=filter, 0=don't filter)
# also set allowable q95 and Beta_N/Li range
filter_WTOTN1 = 0; filter_WTOTN2 = 0; filter_WTOTN3 = 0; filter_WWTOTN1 = 0
q95_range = [0.,10.]#[2, 7]
Bn_Div_Li_range = [0.,5.]#[0.75, 3]

#I-coil details
coilN  = num.array([[2.164, 1.012, 2.374, 0.504],[2.164, -1.012, 2.374, -0.504]])
Nchi = 240

#corsica_list = [['ml10', 1, 1, 3, -0.005, 0.03]]
corsica_list = [['ml10', 0.50, 0.00, 30, -0.005, 0.03]]
#fxrun_executable =
#corsica_executable =
#chease_executable =
#mars_executable = 
#########################################################
########################################################

project_dict['details']['base_dir'] = base_directory + project_name +'/' #this directory must already exist
project_dict['details']['template_dir'] = template_directory
project_dict['details']['efit_master'] = efit_file_location
corsica_base_dir = project_dict['details']['base_dir']+ '/corsica_temp/' #this is a temporary location for corsica files, is removed when done



#####################################################################
##*****STEP 1 - Copy EFIT files + CORSICA(??) ********************###
# Need a neater way to deal with CORSICA + give it its own step?    #
if start_from_step == 1:
    overall_start = time.time()
    project_dict['sims']={}
    project_dict['details'] = cont_funcs.generate_master_dir(project_dict['details'],project_dict)
    os.system('cp ' + project_dict['details']['efit_master'] + '* ' + project_dict['details']['efit_dir'])

    #Output data structure for the next step
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_initial_setup.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


    #corsica_base_dir = '/scratch/haskeysr/corsica_test9/'
    print 'start corsica setup'

    for i in corsica_list:
        Corsica_funcs.corsica_run_setup(corsica_base_dir, i)
    print 'finished corsica setup'
    workers = 1
    worker_list = []
    proportion = int(round(len(corsica_list)/workers))
    print proportion
    script_file_list = []
    for i in range(0,workers):
        if i == (workers-1):
            send_list = corsica_list[proportion*i:len(corsica_list)]
        else:
            send_list = corsica_list[proportion*i:proportion*(i+1)]
        print i
        print send_list
        Corsica_funcs.run_corsica(corsica_base_dir, send_list, 'corsica_script'+str(i)+'.sh')
        script_file_list.append('corsica_script' + str(i)+'.sh')
    for i in script_file_list:
        print 'running ' + i + ' script'
        Corsica_funcs.execute_scripts(corsica_base_dir, i)
    for i in corsica_list:
        print 'copying files across'
        os.system('cp ' + corsica_base_dir + i[0] + '/EXPEQ* stab_setup* ' + project_dict['details']['efit_dir'])
    os.system('rm -r ' +corsica_base_dir)



#####################################################################
##***************************STEP 2 - Copy EFIT files ?********************####

if start_from_step <=2 and end_at_step>=2:
    if start_from_step ==2:
        project_dict = pyMARS.read_data(project_dir +project_name+'_initial_setup.pickle')

    file_location = project_dict['details']['efit_dir']+'/stab_setup_results.dat'
    base_dir = project_dict['details']['base_dir']

    #Read the stab_results file and create the serial numbers in the data structure for each equilibria
    project_dict['sims'] = pyMARS.read_stab_results(file_location)

    #Filter according to settings at top of file
    project_dict = cont_funcs.remove_certain_values(project_dict, q95_range, Bn_Div_Li_range, filter_WTOTN1, filter_WTOTN2, filter_WTOTN3, filter_WWTOTN1)

    #time.sleep(10) #So someone can read what happened
    #generate the directory structure for the project
    project_dict = cont_funcs.generate_directories_func(project_dict, base_dir)

    #Dump the data structure so it can be read by the next step if required
    print 'dumping data to pickle file'
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir']+ project_name + '_setup_directories.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


#####################################################################
##***************************STEP 3 - CHEASE RUN********************#
## Can be incorporated into the previous step... also need to include somethign to do with PEST
if start_from_step <=3 and end_at_step>=3:
    overall_start = time.time()
    if start_from_step ==3:
        project_dict = pyMARS.read_data(project_dict['details']['base_dir'] + project_name+'_setup_directories.pickle')

    #setup and run the chease jobs, these are submitted to the venus cluster
    #set CHEASE_simultaneous_jobs to set how many jobs are run at the same time on the cluster
    project_dict = cont_funcs.chease_setup_run(project_dict,CHEASE_simultaneous_jobs)

    #Run fxrun on the CHEASE output (as required by Yueqiang)
    project_dict = cont_funcs.setup_run_fxrun_func(project_dict)


    #This is the step that launches the batch job
    if include_chease_pest_run ==1:
        cont_funcs.chease_PEST_setup(project_dict)
        project_dict['sims'] = ch_launch.batch_launch_chease(project_dict['sims'], CHEASE_simultaneous_jobs, PEST=1)


    #Dump the data structure so it can be read by the next step if required
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_chease.pickle')
    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)



########################################################################
##***************************STEP 5 - RMZM ********************#########
## This step used to be performed by Matlab and RZplot from Yueqiang ###
## Need to add some switch to allow it as an option                   ##

if start_from_step <=5 and end_at_step>=5:
    overall_start = time.time()
    if start_from_step == 5:
        project_dict = pyMARS.read_data(project_dir + project_name+'_post_chease.pickle')
    RMZM_name = 'RMZM_F'
    project_dict = cont_funcs.RMZM_func(project_dict, coilN, RMZM_name, Nchi)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_RMZM.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


#########################################################################
##*************************** MARS setup ********************############
if start_from_step <=6 and end_at_step>=6:
    overall_start = time.time()
    if start_from_step == 6:
        project_dict = pyMARS.read_data(project_dir + project_name+'_post_RMZM.pickle')

    project_dict = cont_funcs.setup_mars_func(project_dict)

    #Save the data structure so that it can be read by the next step
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_setup.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


####################################################################
##*************************** Run MARS ********************########
if start_from_step <=7 and end_at_step>=7:
    overall_start = time.time()
    if start_from_step == 7:
        project_dict = pyMARS.read_data(project_dir + project_name + '_post_setup.pickle')

    project_dict = cont_funcs.run_mars_function(project_dict, MARS_simultaneous_jobs)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir']+ project_name+'_post_mars_run.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)



####################################################################
##***************************STEP 8 - Post Processing **********####
if start_from_step <=8 and end_at_step>=8:
    overall_start = time.time()
    if start_from_step == 8:
        print 'reading pickle_file'
        project_dict = pyMARS.read_data(project_dir + project_name + '_post_mars_run.pickle')

    serial_list = project_dict['sims'].keys()
    project_dict = cont_funcs.post_processing(project_dict, post_proc_simultaneous_jobs, post_proc_script)

    #project_dict = cont_funcs.coil_outputs_B(project_dict,serial_list)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_processing.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
