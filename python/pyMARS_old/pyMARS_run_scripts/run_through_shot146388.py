#!/usr/bin/env Python

#for these scripts to work, PythonMARS_funcs.py, control_funcs.py, Batch_Launcher.py
#RZfuncs.py must be in the same directory, or in the $PYTHONPATH environment variable
#those files contain the functions that do all the work
#
#must be run on the venus at the moment as it relies of being able to submit batch jobs
#
#Step 1 : Create the base directories, and run Corsica
#Step 2 : Create the whole directory structure based on Corsica output
#Step 3 : Run chease on the Corsica output, then run FourierX
#Step 4 : Find I-coil locations on the CHEASE grid
#Step 5 : Setup for the MARS run
#Step 6 : Run MARS
#Step 7 : Post process MARS - (give coil outputs)
#
#
#To do:
#- Track what is going on with mesh settings for the various codes and affect on post processing
#- Why doesn't CORSICA work on VENUS - works with qrsh, need a better way than piping commands??
# - problem seems to be something to do with $DISPLAY ??
#- Better way of creating the mesh of points on CORSICA
#- Allow option to revert to Liu's Matlab RZPlot for post proc and coil locations
#- Allow to be run on other computers without using the batch pushes - tested in one version
# - need to try running it on Benten
#- Calculate I0EXP based on Chu memo
#- Calculation of coil phasing - need to test
#- Include an upper and lower calculation for phasing studies
#   -seems to work
#- Make all the directory structure relative so the directory can be moved easily   -

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
start_from_step = 2
end_at_step = 7


#directory and file details
project_name = 'shot146388' #this directory will be created in base_directory'
base_directory = '/scratch/haskeysr/mars/'
efit_file_location = '/u/haskeysr/mars/eq_from_matt/efit_files/' #efit files will be copied from here
efit_file_location = '/u/haskeysr/new_from_matt/shot146388/' #efit files will be copied from here
template_directory = '/u/haskeysr/mars/templates/'  #where templates are...
post_proc_script = '/u/haskeysr/pyMARS_dist/post_proc_script.py' #weird way of doing things... to do with the cluster

MARS_execution_script = '/u/haskeysr/bin/runmarsf'
CHEASE_execution_script = '/u/haskeysr/bin/runchease'

# Template names
CORSICA_template_name = 'sspqi_sh3.bas'
CHEASE_template_name = 'datain_template'
MARS_template_name = 'RUN_template'

#Cluster related variables
cluster_job = 1 #whether or not to use the cluster
CHEASE_simultaneous_jobs = 15
MARS_simultaneous_jobs = 15
post_proc_simultaneous_jobs = 10
CORSICA_workers = 1 #LEAVE THIS AT 1, here for future use of CORSICA on cluster - can't get it to work yet....

#Whether to use Liu's Matlab RZplot or my Python versions. You need so set a few more variables for Liu's
#Note this must be set to 1 if you are running on the cluseter as you can't use Matlab on the cluster
#This is enforced further down
RMZM_python = 1

if cluster_job==1 or os.getenv('HOST')=='venusa' or os.getenv('HOST')=='venusb':
    print 'running a cluster job or running on venus, must use Python RZplot funcs'
    print 'Matlab not available'
    RMZM_python = 1
    

# filter out equilibria that DCON finds unstable (1=filter, 0=don't filter)
# also set allowable q95 and Beta_N/Li range, must have set <<calldcon>>=1 in corsica
# for the DCON filters to be meaningful
filter_WTOTN1 = 0; filter_WTOTN2 = 0; filter_WTOTN3 = 0; filter_WWTOTN1 = 0
q95_range = [0.,10.]
Bn_Div_Li_range = [0.,10.]

# Also run chease in PEST mode, useful for viewing results in PEST co-ords
# but not necessary for MARS run
include_chease_PEST_run = 1

#I-coil details, used to place I-coils onto the CHEASE grid
coilN  = num.array([[2.164, 1.012, 2.374, 0.504],[2.164, -1.012, 2.374, -0.504]])
I_coil_frequency = 10 #Hz


#Used to calculate I0EXP from Chu's memo on a correction factor from MARS coil representation
#to reality
N_Icoils = 6
I_coil_current = num.array([1.,-1.,0.,1,-1.,0.])


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
p_mult_min = 0.05
p_mult_max = 1.83
p_mult_number = 14
p_mult_increment = (p_mult_max-p_mult_min)/(p_mult_number-1)
q_mult_min = 0.09
q_mult_max = 2.61
q_mult_number = 14
q_mult_increment = (q_mult_max - q_mult_min)/(q_mult_number-1)



single_runthrough=0
if single_runthrough==1:
    p_mult_min = 1
    p_mult_increment = 0
    q_mult_increment = 0
    q_mult_min = 1
    p_mult_number = 1
    q_mult_number = 1

corsica_settings = [{'<<pmin>>':'%3f'%(p_mult_min),
                     '<<qmin>>':'%.3f'%(q_mult_min),
                     '<<pstep>>':'%.3f'%(p_mult_increment),
                     '<<qstep>>':'%.3f'%(q_mult_increment),
                     '<<npmult>>':'%d'%(q_mult_number),
                     '<<nqmult>>':'%d'%(p_mult_number),
                     '<<npsi>>': '270',
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
#Will update to include an upper and lower calculation as a single run
MARS_phasing = 0 #in deg

#Need convert the phasing into a string form for MARS RUN file
FEEDI_string =  pyMARS.construct_FEEDI(MARS_phasing)
print FEEDI_string
upper_and_lower = 1

MARS_settings = {'<<M1>>': -29,
                 '<<M2>>': 29,
                 '<<FEEDI>>': FEEDI_string,
                 '<<RNTOR>>' : -2,
                 '<<ROTE>>': 0}


# Cleaning up to save space, these files will be removed, make sure a file you need isn't listed!
MARS_rm_files = 'OUTDATA  JPLASMA VPLASMA PPLASMA JACOBIAN'
MARS_rm_files2 = 'OUTRMAR OUTVMAR' #80MB saving per eq
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
project_dict['details']['pickup_coils']['probe_type'] = probe_type
project_dict['details']['pickup_coils']['Rprobe'] = Rprobe
project_dict['details']['pickup_coils']['Zprobe'] = Zprobe
project_dict['details']['pickup_coils']['tprobe'] = tprobe
project_dict['details']['pickup_coils']['lprobe'] = lprobe

project_dict['details']['I-coils'] = {}
project_dict['details']['I-coils']['N_Icoils'] = N_Icoils
project_dict['details']['I-coils']['I_coil_current'] = I_coil_current



project_dict['details']['ICOIL_FREQ'] = I_coil_frequency #Icoil frequency

corsica_base_dir = project_dict['details']['base_dir']+ '/corsica_temp/' #this is a temporary location for corsica files

#leftover from attempt to run multiple CORSICA jobs at once...
corsica_list = [['ml10']] 


print '#####################################################################'
print '##*****STEP 1 - Copy EFIT files + CORSICA(??) ********************###'
# Need a neater way to deal with CORSICA + give it its own step?    #
if start_from_step == 1:
    overall_start = time.time()
    project_dict['sims']={}
    project_dict['details'] = cont_funcs.generate_master_dir(project_dict['details'],project_dict)
    os.system('cp ' + project_dict['details']['efit_master'] + '* ' + project_dict['details']['efit_dir'])
    dir_list = os.listdir(project_dict['details']['efit_dir'])

    #get shot number and time from the efit filename
    for i in range(0,len(dir_list)):
        if dir_list[i].find('.') == 7 and dir_list[i][0]=='g':
            print dir_list[i]
            shot_number = int(dir_list[i][1:7])
            if dir_list[i].find('_')== -1:
                shot_time = int(dir_list[i][8:])
            else:
                shot_time = int(dir_list[i][8:dir_list[i].find('_')])
            print dir_list[i]
            print shot_number, shot_time

    project_dict['details']['shot'] = shot_number #mostly redundant at the moment should get this off the efit file 
    project_dict['details']['shot_time'] = shot_time #important to identify the EXPEQ filenames - need to improve this based on efit name
                         
    #Output data structure for the next step
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_initial_setup.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
    #corsica_base_dir = '/scratch/haskeysr/corsica_test9/'
    print 'start corsica setup'

    #p_string,q_string,len_p_list = cont_funcs.corsica_qmult_pmult()

    #creating lots of directories for the various runs of CORSICA
    for i in range(0,len(corsica_list)):
        cont_funcs.corsica_run_setup(corsica_base_dir, project_dict['details']['efit_dir'],project_dict['details']['template_dir'] + CORSICA_template_name, corsica_list[i], corsica_settings[0])
    print 'finished corsica setup, starting corsica runs'
                  
    cont_funcs.corsica_batch_run(corsica_list, project_dict, corsica_base_dir, workers = CORSICA_workers)
    #cont_funcs.corsica_batch_run_qrsh(corsica_list, project_dict, corsica_base_dir, workers = CORSICA_workers)


print '#####################################################################'
print '##***************************STEP 2 - Generate Directory Structure ############'
if start_from_step <=2 and end_at_step>=2:
    overall_start = time.time()
    if start_from_step ==2:
        project_dict = pyMARS.read_data(project_dir +project_name+'_initial_setup.pickle')

    file_location = project_dict['details']['efit_dir']+'/stab_setup_results.dat'

    #Read the stab_results file and create the serial numbers in the data structure for each equilibria
    project_dict['sims'] = pyMARS.read_stab_results(file_location)

    #Filter according to settings at top of file
    project_dict = cont_funcs.remove_certain_values(project_dict, q95_range, Bn_Div_Li_range, filter_WTOTN1, filter_WTOTN2, filter_WTOTN3, filter_WWTOTN1)

    #generate the directory structure for the project
    project_dict = cont_funcs.generate_directories_func(project_dict, project_dict['details']['base_dir'])

    #Dump the data structure so it can be read by the next step if required
    print 'STEP 2 : dumping data to pickle file'
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir']+ project_name + '_setup_directories.pickle')

    print 'Total Time for step 2 (DIR_Setup) : %.2f'%((time.time()-overall_start)/60)


print '#####################################################################'
print '##***************************STEP 3 - CHEASE RUN********************#'
if start_from_step <=3 and end_at_step>=3:
    overall_start = time.time()
    if start_from_step ==3:
        project_dict = pyMARS.read_data(project_dict['details']['base_dir'] + project_name+'_setup_directories.pickle')

    #setup and run the chease jobs, these are submitted to the venus cluster
    #set CHEASE_simultaneous_jobs to set how many jobs are run at the same time on the cluster
    #This allows the number of running jobs to be set during a run by editing the CHEASE_simul_jobs.txt file
    job_num_filename = project_dict['details']['base_dir']+'CHEASE_simul_jobs.txt'
    job_num_file = open(job_num_filename,'w')
    job_num_file.write('%d\n'%(CHEASE_simultaneous_jobs))
    job_num_file.close()
    
    #Run fxrun on the CHEASE output (as required by Yueqiang)
    #Why does this step work within Ipython but not from bash on venus??
    #something to do with library linking - what is happening?
    #This is the step that launches the batch job
    project_dict = cont_funcs.chease_setup_run(project_dict,job_num_filename, CHEASE_execution_script,PEST=0, fxrun = 1, rm_files = CHEASE_rm_files, cluster_job = cluster_job)

    if include_chease_PEST_run ==1:
        project_dict = cont_funcs.chease_setup_run(project_dict,job_num_filename, CHEASE_execution_script, CHEASE_template = CHEASE_template_name, PEST=1, fxrun = 1, rm_files = CHEASE_PEST_rm_files, cluster_job = cluster_job)
        #Does FourierX need to be run ON THE PEST JOB??? ask Matt or Yueqiang

    #Dump the data structure so it can be read by the next step if required
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_chease.pickle')
    print 'Total Time for step 3 (CHEASE) : %.2f'%((time.time()-overall_start)/60)


print '########################################################################'
print '##*************** STEP 4 - I-coil grid location *** *          #########'
## This step used to be performed by Matlab and RZplot from Liu ###
## Currently using my version of RZplot in Python - need to benchmark code more ##
## Need to add some switch to allow the Matlab option###################

if start_from_step <=4 and end_at_step>=4:
    overall_start = time.time()
    if start_from_step == 4:
        project_dict = pyMARS.read_data(project_dir + project_name+'_post_chease.pickle')
    RMZM_name = 'RMZM_F' #This is here so that you can choose between pest and this?
    if RMZM_python ==1:
        project_dict = cont_funcs.RMZM_func(project_dict, coilN, RMZM_name)
    else:
        project_dict = cont_funcs.RMZM_func_matlab(project_dict, '/u/haskeysr/matlab/RZplot3/','/u/haskeysr/matlab/RZplot3/MacMainD3D_Master.m')
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_RMZM.pickle')

    print 'Total Time for step 4 : %.2f'%((time.time()-overall_start)/60)


print '#########################################################################'
print '##*************************** Step 5 - MARS setup**************############'
if start_from_step <=5 and end_at_step>=5:
    overall_start = time.time()
    if start_from_step == 5:
        project_dict = pyMARS.read_data(project_dir + project_name+'_post_RMZM.pickle')


    project_dict = cont_funcs.setup_mars_func(project_dict, upper_and_lower = upper_and_lower, MARS_template_name = MARS_template_name)

    #Save the data structure so that it can be read by the next step
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_setup.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


print '####################################################################'
print '##*************************** Step 6 - run MARS ********************########'
if start_from_step <=6 and end_at_step>=6:
    overall_start = time.time()
    if start_from_step == 6:
        project_dict = pyMARS.read_data(project_dir + project_name + '_post_setup.pickle')

    job_num_filename = project_dict['details']['base_dir']+'MARS_simul_jobs.txt'
    job_num_file = open(job_num_filename,'w')
    job_num_file.write('%d\n'%(MARS_simultaneous_jobs))
    job_num_file.close()

    project_dict = cont_funcs.run_mars_function(project_dict, job_num_filename, MARS_execution_script,rm_files = MARS_rm_files, rm_files2 = MARS_rm_files2, cluster_job = cluster_job, upper_and_lower = upper_and_lower)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir']+ project_name+'_post_mars_run.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)



print '####################################################################'
print '##***************************STEP 7 - Post Processing **********####'
## **Set this up to have the pickup coil details passed to it** ##
if start_from_step <=7 and end_at_step>=7:
    overall_start = time.time()
    if start_from_step == 7:
        print 'reading pickle_file'
        project_dict = pyMARS.read_data(project_dir + project_name + '_post_mars_run.pickle')

    serial_list = project_dict['sims'].keys()
    project_dict = cont_funcs.post_processing(project_dict, post_proc_simultaneous_jobs, post_proc_script, upper_and_lower = upper_and_lower, cluster_job = cluster_job)

    #project_dict = cont_funcs.coil_outputs_B(project_dict,serial_list)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_processing.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)

print 'completion time : %.4f mins' %((time.time()-initial_start_time)/60)
