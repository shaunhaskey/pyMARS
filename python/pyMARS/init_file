#!/usr/bin/env Python

#for these scripts to work, PythonMARS_funcs.py, control_funcs.py, Batch_Launcher.py
#RZfuncs.py must be in the same directory, or in the $PYTHONPATH environment variable
#those files contain the functions that do all the work.
#
#must be run on the venus at the moment as it relies of being able to submit batch jobs
#
#Step 1 : Create the base directories, and run Corsica
#Step 2 : Create the whole directory structure based on Corsica output
#Step 3 : Run chease on the Corsica output
#Step 4 : Modify the CHEASE output (as required by Yueqiang)
#Step 5 : Find I-coil locations on the CHEASE grid
#Step 6 : Setup for the MARS run
#Step 7 : Run MARS
#Step 8 : Post process MARS
#
#
#To do:
#- Track what is going on with mesh settings for the various codes and affect on post processing
#- Why doesn't CORSICA work on VENUS - works with qrsh, need a better way than piping commands??
#- Better way of creating the mesh of points on CORSICA
#- Allow option to revert to Yiueqiang's RZPlot for post proc and coil locations
#- Allow to be run on other computers without using the batch pushes
#- Automatically read in the shot number and time from the efit file
#- Calculation of coil phasing
#- Include an upper and lower calculation for phasing studies

#Shaun Haskey Jan 18 2012 shaunhaskey@gmail.com


import PythonMARS_funcs as pyMARS
import control_funcs as cont_funcs
import Batch_Launcher as batch_launch
import pickle, time, os
import numpy as num


#########################################################
#########################################################
############ SET THESE VALUES BEFORE RUNNING!!!!! #######

#PROCESS CONTROL starting and finishing step variables
start_from_step = 1
end_at_step = 1

#directory details
project_name = 'testing_new_code8'
base_directory = '/scratch/haskeysr/mars/'
efit_file_location = '/u/haskeysr/mars/eq_from_matt/efit_files/'
template_directory = '/u/haskeysr/mars/templates/'
post_proc_script = '/u/haskeysr/python_mars_scripts/post_proc_script.py'

shot_time = 2306
shot_number = 138344

#Cluster related variables
CHEASE_simultaneous_jobs = 10
MARS_simultaneous_jobs = 10
post_proc_simultaneous_jobs = 6
CORSICA_workers = 4

# filter out equilibria that DCON finds unstable (1=filter, 0=don't filter)
# also set allowable q95 and Beta_N/Li range, must have set <<calldcon>>=1 in corsica
# for the DCON filters to be meaningful
filter_WTOTN1 = 0; filter_WTOTN2 = 0; filter_WTOTN3 = 0; filter_WWTOTN1 = 0
q95_range = [0.,10.]
Bn_Div_Li_range = [0.,5.]

# Also run chease in PEST mode, useful for viewing results in PEST co-ords
# but not necessary for MARS run
include_chease_PEST_run = 1

#I-coil details, used to place I-coils onto the CHEASE grid
coilN  = num.array([[2.164, 1.012, 2.374, 0.504],[2.164, -1.012, 2.374, -0.504]])
I_coil_frequency = 20 #Hz


#Post processing coil details need to be included here
#instead of using the defaults which are stored in PythonMARS_funcs.py
# probe type 1: poloidal field, 2: radial field
probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe



#Corsica settings, note, each <<XXX>> must exist in the template as a placeholder to be
#replaced. You can add other values to this list if you want, just include the new placeholder
#in the template
corsica_settings = [{'<<pmin>>':'0.5',
                    '<<qmin>>':'0.5',
                    '<<pstep>>':'0.2',
                    '<<qstep>>':'0.2',
                    '<<npmult>>':'3',
                    '<<nqmult>>':'2',
                    '<<calldcon>>':'0',
                     '<<thetac>>':0.003}]


#CHEASE settings, note, each <<XXX>> must exist in the template as a placeholder to be
#replaced. You can add other values to this list if you want, just include the new placeholder
#in the template
CHEASE_settings = {'<<NCHI>>': 240,
                   '<<NPSI>>': 180,
                   '<<NT>>': 60,
                   '<<NS>>': 60,
                   '<<NV>>': 200,
                   '<<REXT>>': 7.0}


#MARS settings
#Will update to include an upper and lower calculation as a single run (?)
MARS_settings = {'<<M1>>': -29,
                 '<<M2>>': 29,
                 '<<FEEDI>>': '(1.0,0.0),(1.0, 0.0)',
                 '<<RNTOR>>' : -2,
                 '<<ROTE>>': 0}


# Cleaning up to save space, these files will be removed, make sure a file you need isn't listed!
MARS_rm_files = 'OUTDATA  JPLASMA VPLASMA PLASMA JACOBIAN'
CHEASE_rm_files = 'NUPLO INP1_FORMATTED'
CHEASE_PEST_rm_files = 'OUTRMAR OUTVMAR INP1_FORMATTED NUPLO'


####################### END USER SETUP VARIABLES ########
#########################################################



initial_start_time = time.time()

project_dir = base_directory + project_name +'/'

#Create the project dictionary
project_dict={}
project_dict['details']={}
project_dict['details']['base_dir'] = base_directory + project_name +'/'
os.system('mkdir ' + project_dict['details']['base_dir'])
project_dict['details']['template_dir'] = template_directory
project_dict['details']['efit_master'] = efit_file_location
project_dict['details']['CHEASE_settings'] = CHEASE_settings
project_dict['details']['CHEASE_settings_PEST'] = CHEASE_settings
project_dict['details']['MARS_settings'] = MARS_settings
project_dict['details']['corsica_settings'] = corsica_settings

project_dict['details']['pickup_coils']={}
project_dict['details']['pickup_coils']['probe'] = probe
project_dict['details']['pickup_coils']['probe'] = probe_type
project_dict['details']['pickup_coils']['Rprobe'] = Rprobe
project_dict['details']['pickup_coils']['Zprobe'] = Zprobe
project_dict['details']['pickup_coils']['tprobe'] = tprobe
project_dict['details']['pickup_coils']['lprobe'] = lprobe


project_dict['details']['shot'] = shot_number #mostly redundant at the moment should get this off the efit file 
project_dict['details']['shot_time'] = shot_time #important to identify the EXPEQ filenames - need to improve this based on efit name
project_dict['details']['ICOIL_FREQ'] = I_coil_frequency #Icoil frequency

corsica_base_dir = project_dict['details']['base_dir']+ '/corsica_temp/' #this is a temporary location for corsica files

corsica_list = [['ml10'],['ml11'],['ml12'],['ml13'],['ml14'],['ml15'],['ml16'],['ml17'] ] #leftover from attempt to run multiple CORSICA jobs at once...


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
    for i in range(0,len(corsica_list)):
        cont_funcs.corsica_run_setup2(corsica_base_dir, corsica_list[i],corsica_settings[0])
    print 'finished corsica setup, starting corsica runs'
                  
    cont_funcs.corsica_batch_run_qrsh(corsica_list, project_dict, corsica_base_dir, workers = CORSICA_workers)


#####################################################################
##***************************STEP 2 - Generate Directory Structure ############
if start_from_step <=2 and end_at_step>=2:
    overall_start = time.time()
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

if start_from_step <=3 and end_at_step>=3:
    overall_start = time.time()
    if start_from_step ==3:
        project_dict = pyMARS.read_data(project_dict['details']['base_dir'] + project_name+'_setup_directories.pickle')

    #setup and run the chease jobs, these are submitted to the venus cluster
    #set CHEASE_simultaneous_jobs to set how many jobs are run at the same time on the cluster

    job_num_filename = project_dict['details']['base_dir']+'CHEASE_simul_jobs.txt'
    job_num_file = open(job_num_filename,'w')
    job_num_file.write('%d\n'%(CHEASE_simultaneous_jobs))
    job_num_file.close()
    
    project_dict = cont_funcs.chease_setup_run(project_dict,job_num_filename, PEST=0, fxrun = 1, rm_files = CHEASE_rm_files)

    #Run fxrun on the CHEASE output (as required by Yueqiang)
    #Why does this step work within Ipython but not from bash on venus??
    #something to do with library linking - what is happening?

    #project_dict = cont_funcs.setup_run_fxrun_func(project_dict)

    #This is the step that launches the batch job
    if include_chease_PEST_run ==1:
        project_dict = cont_funcs.chease_setup_run(project_dict,job_num_filename, PEST=1, fxrun = 1, rm_files = CHEASE_PEST_rm_files)
        #DO I NEED TO RUN FourierX ON THE PEST JOB??? ask Matt or Yueqiang

    #Dump the data structure so it can be read by the next step if required
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_chease.pickle')
    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)



########################################################################
##***************************STEP 4 - RMZM ********************#########
## This step used to be performed by Matlab and RZplot from Yueqiang ###
## Currently using my version of RZplot in Python - need to benchmark code more ##
## Need to add some switch to allow the Matlab option###################

if start_from_step <=4 and end_at_step>=4:
    overall_start = time.time()
    if start_from_step == 4:
        project_dict = pyMARS.read_data(project_dir + project_name+'_post_chease.pickle')
    RMZM_name = 'RMZM_F'

    project_dict = cont_funcs.RMZM_func(project_dict, coilN, RMZM_name)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_RMZM.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


#########################################################################
##*************************** Step 5 - MARS setup ********************############
if start_from_step <=5 and end_at_step>=5:
    overall_start = time.time()
    if start_from_step == 5:
        project_dict = pyMARS.read_data(project_dir + project_name+'_post_RMZM.pickle')

    project_dict = cont_funcs.setup_mars_func(project_dict)

    #Save the data structure so that it can be read by the next step
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_setup.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


####################################################################
##*************************** Step 6 - run MARS ********************########
if start_from_step <=6 and end_at_step>=6:
    overall_start = time.time()
    if start_from_step == 6:
        project_dict = pyMARS.read_data(project_dir + project_name + '_post_setup.pickle')

    job_num_filename = project_dict['details']['base_dir']+'MARS_simul_jobs.txt'
    job_num_file = open(job_num_filename,'w')
    job_num_file.write('%d\n'%(MARS_simultaneous_jobs))
    job_num_file.close()

    project_dict = cont_funcs.run_mars_function(project_dict, job_num_filename, rm_files = MARS_rm_files)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir']+ project_name+'_post_mars_run.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)



####################################################################
##***************************STEP 7 - Post Processing **********####
## **Set this up to have the pickup coil details passed to it** ##
if start_from_step <=7 and end_at_step>=7:
    overall_start = time.time()
    if start_from_step == 7:
        print 'reading pickle_file'
        project_dict = pyMARS.read_data(project_dir + project_name + '_post_mars_run.pickle')

    serial_list = project_dict['sims'].keys()
    project_dict = cont_funcs.post_processing(project_dict, post_proc_simultaneous_jobs, post_proc_script)

    #project_dict = cont_funcs.coil_outputs_B(project_dict,serial_list)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_processing.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)

print 'completion time : %.4f mins' %((time.time()-initial_start_time)/60)
