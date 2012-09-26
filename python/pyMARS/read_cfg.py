#!/usr/bin/env Python

import PythonMARS_funcs as pyMARS
import control_funcs as cont_funcs
import Batch_Launcher as batch_launch
import pickle, time, os,sys
import numpy as num
from ConfigParser import SafeConfigParser
import numpy as num

config_filename = sys.argv[1]
parser = SafeConfigParser()
parser.optionxform=str
parser.read(config_filename)

#process control
start_from_step = int(parser.get('process_control', 'start_from_step'))
end_at_step = int(parser.get('process_control', 'end_at_step'))
include_chease_PEST_run = int(parser.get('process_control', 'include_chease_PEST_run'))

#directory details
project_name = parser.get('directory_details', 'project_name')
base_directory = parser.get('directory_details', 'base_directory')
efit_file_location = parser.get('directory_details', 'efit_file_location')
template_directory = parser.get('directory_details', 'template_directory')
post_proc_script = parser.get('directory_details', 'post_proc_script')

#execution_scripts
MARS_execution_script = parser.get('execution_scripts', 'MARS_execution_script')
CHEASE_execution_script = parser.get('execution_scripts', 'CHEASE_execution_script')

#template_names
CORSICA_template_name = parser.get('template_names', 'CORSICA_template_name')
CHEASE_template_name = parser.get('template_names', 'CHEASE_template_name')
MARS_template_name = parser.get('template_names', 'MARS_template_name')

#Cluster related variables
cluster_job = int(parser.get('cluster_details', 'cluster_job'))
CHEASE_simultaneous_jobs = int(parser.get('cluster_details', 'CHEASE_simultaneous_jobs'))
MARS_simultaneous_jobs = int(parser.get('cluster_details', 'MARS_simultaneous_jobs'))
post_proc_simultaneous_jobs = int(parser.get('cluster_details', 'post_proc_simultaneous_jobs'))
CORSICA_workers = int(parser.get('cluster_details', 'CORSICA_workers'))

#RMZM stuff
RMZM_python = int(parser.get('RMZM_python_details', 'RMZM_python'))
if cluster_job==1 or os.getenv('HOST')=='venusa' or os.getenv('HOST')=='venusb':
    print 'running a cluster job or running on venus, must use Python RZplot funcs'
    print 'Matlab not available'
    RMZM_python = 1


#filters
filter_WTOTN1 = int(parser.get('filters', 'filter_WTOTN1'))
filter_WTOTN2 = int(parser.get('filters', 'filter_WTOTN2'))
filter_WTOTN3 = int(parser.get('filters', 'filter_WTOTN3'))
filter_WWTOTN1 = int(parser.get('filters', 'filter_WWTOTN1'))
q95_range = map(float, parser.get('filters', 'q95_range').split(','))
Bn_Div_Li_range = map(float, (parser.get('filters', 'Bn_Div_Li_range').split(',')))

#i_coil_details
coilN1 = map(float, parser.get('i_coil_details', 'coilN1').split(','))
coilN2 = map(float, parser.get('i_coil_details', 'coilN2').split(','))
coilN = num.array([coilN1, coilN2])
I_coil_frequency = float(parser.get('i_coil_details', 'I_coil_frequency'))
N_Icoils = int(parser.get('i_coil_details', 'N_Icoils'))
I_coil_current = num.array(map(float, parser.get('i_coil_details', 'I_coil_current').split(',')))


#pickup details
probe = parser.get('pickup_probe_details', 'probe').split(',')
probe_type = map(int, parser.get('pickup_probe_details', 'probe_type').split(','))
Rprobe = num.array(map(float, parser.get('pickup_probe_details', 'Rprobe').split(',')))
Zprobe = num.array(map(float, parser.get('pickup_probe_details', 'Zprobe').split(',')))
tprobe = num.array(map(float, parser.get('pickup_probe_details', 'tprobe').split(',')))*num.pi/180.
lprobe = num.array(map(float, parser.get('pickup_probe_details', 'lprobe').split(',')))


#read in corsica settings
single_runthrough = int(parser.get('corsica_settings', 'single_runthrough'))
p_mult_min = float(parser.get('corsica_settings', 'p_mult_min'))
p_mult_max = float(parser.get('corsica_settings', 'p_mult_max'))
p_mult_number = int(parser.get('corsica_settings', 'p_mult_number'))
p_mult_increment = (p_mult_max-p_mult_min)/(p_mult_number-1)
q_mult_min = float(parser.get('corsica_settings', 'q_mult_min'))
q_mult_max = float(parser.get('corsica_settings', 'q_mult_max'))
q_mult_number = int(parser.get('corsica_settings', 'q_mult_number'))

if q_mult_number==1:
    q_mult_increment = 0
else:
    q_mult_increment = (q_mult_max - q_mult_min)/(q_mult_number-1)

if single_runthrough==1:
    p_mult_min = 1
    p_mult_increment = 0
    q_mult_increment = 0
    q_mult_min = 1
    p_mult_number = 1
    q_mult_number = 1

corsica_settings = {'<<pmin>>':'%3f'%(p_mult_min),
                     '<<qmin>>':'%.3f'%(q_mult_min),
                     '<<pstep>>':'%.3f'%(p_mult_increment),
                     '<<qstep>>':'%.3f'%(q_mult_increment),
                     '<<npmult>>':'%d'%(p_mult_number),
                     '<<nqmult>>':'%d'%(q_mult_number)}

for i,j in parser.items('corsica_settings2'):
    corsica_settings[i] = j
corsica_settings = [corsica_settings]

CHEASE_settings = {}
for i,j in parser.items('CHEASE_settings'):
    CHEASE_settings[i] =j

#MARS
MARS_phasing = float(parser.get('MARS_settings', 'MARS_phasing'))
upper_and_lower = int(parser.get('MARS_settings', 'upper_and_lower'))
MARS_settings = {}
FEEDI_string =  pyMARS.construct_FEEDI(MARS_phasing)
print FEEDI_string
MARS_settings['<<FEEDI>>'] = FEEDI_string
for i,j in parser.items('MARS_settings2'):
    print i, j
    MARS_settings[i] = int(j)

print MARS_settings
#cleaning up to save space
MARS_rm_files = parser.get('clean_up_settings', 'MARS_rm_files')
MARS_rm_files2 = parser.get('clean_up_settings', 'MARS_rm_files2')
CHEASE_rm_files = parser.get('clean_up_settings', 'CHEASE_rm_files')
CHEASE_PEST_rm_files = parser.get('clean_up_settings', 'CHEASE_PEST_rm_files')




#=================End read in config file===============================
#========================================================================
initial_start_time = time.time()

project_dir = base_directory + project_name +'/'

#Create the project dictionary
project_dict={}
project_dict['details']={}
project_dict['details']['base_dir'] = base_directory + project_name +'/'

if not os.path.exists(project_dict['details']['base_dir']):
    print 'project directory doesnt exist - creating...',
    os.system('mkdir ' + project_dict['details']['base_dir'])
    print 'done'
else:
    print 'project directory already exists - continuing'

i = 1
curr_filename = sys.argv[0].lstrip('.').lstrip('/')
tmp_filename = '/%d_%s'%(i, curr_filename)
print tmp_filename
while os.path.exists(project_dict['details']['base_dir']+tmp_filename):
    i+=1
    tmp_filename = '%d_%s'%(i, curr_filename)
    print '%d already exists'%(i)
os.system('cp ' + os.path.abspath(sys.argv[0]) + ' ' + project_dict['details']['base_dir']+tmp_filename)
print 'script file copied across for record keeping'


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
            shot_number = int(dir_list[i][1:7])
            if dir_list[i].find('_')== -1:
                shot_time = int(dir_list[i][8:])
            else:
                shot_time = int(dir_list[i][8:dir_list[i].find('_')])
            print shot_number, shot_time

    project_dict['details']['shot'] = shot_number
    project_dict['details']['shot_time'] = shot_time

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

    project_dict = cont_funcs.check_chease_run(project_dict, RMZM_name)
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
    project_dict = cont_funcs.post_processing(project_dict, post_proc_simultaneous_jobs, post_proc_script, directory = 'post_proc_tmp/', upper_and_lower = upper_and_lower, cluster_job = cluster_job)

    #project_dict = cont_funcs.coil_outputs_B(project_dict,serial_list)

    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_processing.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)

print 'completion time : %.4f mins' %((time.time()-initial_start_time)/60)


print '####################################################################'
print '##**************** STEP 8 - MARS PEST Post Processing **********####'
if start_from_step <=8 and end_at_step>=8:
    if start_from_step == 8:
        print 'reading pickle_file'
        tmp_filename = project_dir + project_name + '_post_processing.pickle'
        print tmp_filename
        project_dict = pyMARS.read_data(project_dir + project_name + '_post_processing.pickle')

    #serial_list = project_dict['sims'].keys()
    post_proc_script = '/u/haskeysr/code/NAMP_analysis/python/pyMARS/post_proc_script_PEST.py'
    working_directory = 'post_proc_tmp2/'
    os.system('mkdir ' + project_dir + working_directory)
    project_dict = cont_funcs.post_processing(project_dict, post_proc_simultaneous_jobs, post_proc_script, directory = working_directory, upper_and_lower = upper_and_lower, cluster_job = cluster_job)

    #project_dict = cont_funcs.coil_outputs_B(project_dict,serial_list)
    pyMARS.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_processing_PEST.pickle')
    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
