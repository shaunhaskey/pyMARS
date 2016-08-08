#!/bin/env python2.7

import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.control_funcs as cont_funcs
import pyMARS.Batch_Launcher as batch_launch
import pyMARS as pyMARS_mod
import pickle, time, os, sys, copy
import subprocess as sub
import numpy as num
import ConfigParser
import numpy as num
import numpy as np
import shutil
config_filename = sys.argv[1]
parser = ConfigParser.SafeConfigParser()
parser.optionxform=str
parser.read(config_filename)

print 'pyMARS module version {}'.format(pyMARS_mod.__version__)
#process control
start_from_step = int(parser.get('process_control', 'start_from_step'))
end_at_step = int(parser.get('process_control', 'end_at_step'))
include_chease_PEST_run = int(parser.get('process_control', 'include_chease_PEST_run'))
try:
    multiple_efits = int(parser.get('process_control', 'multiple_efits'))
except ConfigParser.NoOptionError, e:
    print 'Couldnt find multiple_efits - setting it to 0', e
    multiple_efits = 0

try:
    rotation_scan = int(parser.get('process_control', 'rotation_scan'))
    rotation_start = float(parser.get('process_control', 'rotation_scan_start'))
    rotation_end = float(parser.get('process_control', 'rotation_scan_end'))
    rotation_num = float(parser.get('process_control', 'rotation_scan_number'))
except ConfigParser.NoOptionError, e:
    print 'Couldnt find rotation scan settings, setting them to defaults', e
    rotation_scan = 0; rotation_start  = 0; rotation_end = 1; rotation_num = 0

try:
    resistivity_scan = int(parser.get('process_control', 'resistivity_scan'))
    resistivity_start = float(parser.get('process_control', 'resistivity_scan_start'))
    resistivity_end = float(parser.get('process_control', 'resistivity_scan_end'))
    resistivity_num = float(parser.get('process_control', 'resistivity_scan_number'))
except ConfigParser.NoOptionError, e:
    print 'Couldnt find resistivity scan settings, setting them to defaults', e
    resistivity_scan = 0; resistivity_start = 0; resistivity_end = 0; resistivity_num = 0
try:
    rotation_spacing = str(parser.get('process_control', 'rotation_spacing'))
except ConfigParser.NoOptionError, e:
    print 'Couldnt find rotation spacing, setting to lin', e
    rotation_spacing = 'lin'

try:
    resistivity_spacing = str(parser.get('process_control', 'resistivity_spacing'))
except ConfigParser.NoOptionError, e:
    print 'Couldnt find restivity spacing, setting to lin', e
    resistivity_spacing = 'lin'
if resistivity_scan:
    if resistivity_spacing=='log':
        res_scan_list = 10**(np.linspace(resistivity_start, resistivity_end, resistivity_num, endpoint=True))
    else:
        res_scan_list = np.linspace(resistivity_start, resistivity_end, resistivity_num, endpoint=True)
else:
    res_scan_list = None
if rotation_scan:
    if rotation_spacing=='log':
        rot_scan_list = 10**(np.linspace(rotation_start, rotation_end, rotation_num, endpoint=True))
    else:
        rot_scan_list = np.linspace(rotation_start, rotation_end, rotation_num, endpoint=True)
else:
    rot_scan_list = None
print 'resistivity_scan {}; resistivity_start  {}; resistivity_end {}; resistivity_num {}'.format(rotation_scan, rotation_start, rotation_end, rotation_num )
print res_scan_list
print 'rotation_scan {}; rotation_start  {}; rotation_end {}; rotation_num {}'.format(rotation_scan, rotation_start, rotation_end, rotation_num )
print rot_scan_list

# if rotation_scan and resistivity_scan:
#     print "cant do rotation and resistivity scans at the same time... choose again...."
#     raise(ValueError)

#directory details
project_name = parser.get('directory_details', 'project_name')
base_directory = parser.get('directory_details', 'base_directory')
if base_directory[-1]!='/':base_directory+='/'
efit_file_location = parser.get('directory_details', 'efit_file_location')
if efit_file_location[-1]!='/':efit_file_location+='/'
template_directory = parser.get('directory_details', 'template_directory')
if template_directory[-1]!='/':template_directory+='/'
post_proc_script = parser.get('directory_details', 'post_proc_script')
post_proc_script_PEST = parser.get('directory_details', 'post_proc_script_PEST')

try:
    profile_file_location = parser.get('directory_details', 'profile_file_location')
    if profile_file_location[-1]!='/':profile_file_location+='/'
except ConfigParser.NoOptionError, e:
    print 'Couldnt find profile_file_location - setting it to same dir as efit', e
    profile_file_location = efit_file_location

#execution_scripts
MARS_execution_script = parser.get('execution_scripts', 'MARS_execution_script')
CHEASE_execution_script = parser.get('execution_scripts', 'CHEASE_execution_script')

#template_names
CORSICA_template_name = parser.get('template_names', 'CORSICA_template_name')
try:
    CORSICA_template_name2 = parser.get('template_names', 'CORSICA_template_name2')
except ConfigParser.NoOptionError, e:
    print 'Second CORSICA template not named, setting to same as first name'
    CORSICA_template_name2 = CORSICA_template_name

CHEASE_template_name = parser.get('template_names', 'CHEASE_template_name')
MARS_template_name = parser.get('template_names', 'MARS_template_name')

#Cluster related variables
cluster_job = int(parser.get('cluster_details', 'cluster_job'))
CHEASE_simultaneous_jobs = int(parser.get('cluster_details', 'CHEASE_simultaneous_jobs'))
MARS_simultaneous_jobs = int(parser.get('cluster_details', 'MARS_simultaneous_jobs'))
post_proc_simultaneous_jobs = int(parser.get('cluster_details', 'post_proc_simultaneous_jobs'))
CORSICA_workers = int(parser.get('cluster_details', 'CORSICA_workers'))
try:
    MARS_memory = int(parser.get('cluster_details', 'MARS_memory'))
    print('MARS memory requirement: {}GB'.format(MARS_memory))
except ConfigParser.NoOptionError, e:
    print 'MARS memory requirement not specified, going with 15GB'
    MARS_memory = 15

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
I_coil_raw_data = parser.get('i_coil_details', 'I_coil_current')


#pickup details
probe = [i.rstrip(' ').lstrip(' ') for i in parser.get('pickup_probe_details', 'probe').split(',')]
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
if p_mult_number==1:
    p_mult_increment = 0
else:
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
n_eqs = p_mult_number * q_mult_number
print 'p_mult_number:{}, q_mult_number:{}, n_eqs:{}, CORSICA_workers:{}'.format(p_mult_number, q_mult_number, n_eqs, CORSICA_workers)
if n_eqs < CORSICA_workers:
    print("Reducing number of CORSICA workers from {} to {} so they all have something to do".format(CORSICA_workers, n_eqs))
    CORSICA_workers = n_eqs


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
FEEDI_string =  pyMARS_funcs.construct_FEEDI(MARS_phasing)
print FEEDI_string
MARS_settings['<<FEEDI>>'] = FEEDI_string
new_mars_settings = {}
for i,j in parser.items('MARS_settings2'):
    print i, j
    try:
        MARS_settings[i] = int(j)
        new_mars_settings[i] = int(j)
    except ValueError:
        MARS_settings[i] = float(j)
        new_mars_settings[i] = float(j)

print MARS_settings

#cleaning up to save space
MARS_rm_files = parser.get('clean_up_settings', 'MARS_rm_files')
MARS_rm_files2 = parser.get('clean_up_settings', 'MARS_rm_files2')
CHEASE_rm_files = parser.get('clean_up_settings', 'CHEASE_rm_files')
CHEASE_PEST_rm_files = parser.get('clean_up_settings', 'CHEASE_PEST_rm_files')
try:
    CORSICA_rm_files = parser.get('clean_up_settings', 'CORSICA_rm_files')
except ConfigParser.NoOptionError, e:
    print 'CORSICA rm files not set'
    CORSICA_rm_files = ''


if I_coil_raw_data == '':
    tmp_n = np.abs(MARS_settings['<<RNTOR>>'])
    I_coil_current = num.cos(tmp_n * num.deg2rad(num.arange(0, 360, 60)))
    print "I_coil_current in input is '', setting to:{}".format(I_coil_current)
else:
    I_coil_current = num.array(map(float, I_coil_raw_data.split(',')))
    print "I_coil_current read from input: {}".format(I_coil_current)

#=================End read in config file===============================
#========================================================================
initial_start_time = time.time()

project_dir = base_directory + project_name +'/'

#Create the project dictionary
project_dict={}
project_dict['details']={}
proj_base_dir = base_directory + project_name +'/'
#project_dict['details']['base_dir'] = base_directory + project_name +'/'

if not os.path.exists(proj_base_dir):
    print 'project directory doesnt exist - creating...',
    os.system('mkdir ' + proj_base_dir)
    print 'done'
else:
    print 'project directory already exists - continuing'

if os.path.normpath(proj_base_dir) != os.path.normpath(os.getcwd()):
    raise ValueError("Current working directory is not the same as project base directory - Aborting!!")
i = 1
#curr_filename = sys.argv[0].lstrip('.').lstrip('/')
curr_filename = sys.argv[1].lstrip('.').lstrip('/')
tmp_filename = '/%d_%s'%(i, curr_filename)
print tmp_filename
while os.path.exists(proj_base_dir+tmp_filename):
    i+=1
    tmp_filename = '/%d_%s'%(i, curr_filename)
    print '%d already exists'%(i)
print sys.argv[1]
#os.system('cp ' + os.path.abspath(sys.argv[0]) + ' ' + project_dict['details']['base_dir']+tmp_filename)
os.system('cp ' + sys.argv[1] + ' ' + proj_base_dir+tmp_filename)
print 'script file copied across for record keeping'

host = sub.check_output('echo $HOSTNAME',shell=True) # add host name to master dict (DBW 8/3/2016)
host = host.splitlines()[0]

project_dict['details'] = {'template_dir':template_directory, 'efit_master':efit_file_location,
                           'profile_master':profile_file_location,'CHEASE_settings':CHEASE_settings,
                           'CHEASE_settings_PEST':CHEASE_settings,'MARS_settings':MARS_settings,
                           'corsica_settings':corsica_settings,'ICOIL_FREQ':I_coil_frequency,
                           'base_dir':proj_base_dir,'host':host}

pickup_coils_details =  {'probe':probe,'probe_type':probe_type,'Rprobe':Rprobe, 
                         'Zprobe':Zprobe, 'tprobe':tprobe,'lprobe':lprobe}
I_coil_details = {'N_Icoils': N_Icoils, 'I_coil_current': I_coil_current}

project_dict['details']['pickup_coils'] = copy.deepcopy(pickup_coils_details)
project_dict['details']['I-coils'] = copy.deepcopy(I_coil_details)

corsica_base_dir = project_dict['details']['base_dir']+ '/corsica/' #this is a temporary location for corsica files



print '#####################################################################'
print '##*****STEP 1 - Copy EFIT files + CORSICA(??) ********************###'
# Need a neater way to deal with CORSICA + give it its own step?    #
if start_from_step == 1:
    overall_start = time.time()
    project_dict['sims']={'host':host}
    project_dict['details'] = cont_funcs.generate_master_dir(project_dict['details'],project_dict)

    if multiple_efits == 1:
        time_list, gfile_list = cont_funcs.find_relevant_efit_files(project_dict['details']['efit_master'], project_dict['details']['profile_master'])
        print 'efit times :', time_list
        project_dict['details']['multiple_efit'] = []
        density_prefix = 'dne'; density_post = ''
        rotation_prefix = 'dtrot'; rotation_post = ''
        te_prefix = 'dte'; te_post = ''
        ti_prefix = 'dti'; ti_post = ''
        for i,time_tmp in enumerate(time_list):
            project_dict['details']['multiple_efit'].append(project_dict['details']['efit_dir'] + '/' + str(time_tmp) + '/')
            try:
                os.mkdir(project_dict['details']['multiple_efit'][-1])
            except OSError, e:
                print e
            #copy across the efit files
            for efit_file_type in ['a','g','k','m']:
                os.system('cp ' + project_dict['details']['efit_master'] + '/' + efit_file_type + gfile_list[i][1:] + ' ' + project_dict['details']['multiple_efit'][-1])
            #copy across the rotation files
            density_filename = 'dne'+gfile_list[i][1:]+'.dat'
            #rotation_filename = 'dpr' + gfile_list[i].split('.')[0][1:] + '.' + str(time_tmp) + '_Er_RBpol.dat'
            density_filename = density_prefix+gfile_list[i][1:]+'.dat' + density_post
            rotation_filename = rotation_prefix+gfile_list[i][1:]+'.dat' + rotation_post
            te_filename = te_prefix+gfile_list[i][1:]+'.dat' + te_post
            ti_filename = ti_prefix+gfile_list[i][1:]+'.dat' + ti_post
            
            #rotation_filename = 'dpr' + gfile_list[i].split('.')[0][1:] + '.' + str(time_tmp) + '_Er_RBpol.dat'

            for tmp_file, mars_name in zip([density_filename, rotation_filename, ti_filename, te_filename],['PROFDEN', 'PROFROT', 'PROFTI', 'PROFTE']):
                shutil.copy('{}/{}'.format(project_dict['details']['profile_master'], tmp_file), project_dict['details']['multiple_efit'][-1])
                shutil.copy('{}/{}'.format(project_dict['details']['profile_master'], tmp_file), '{}/{}'.format(project_dict['details']['multiple_efit'][-1], mars_name))
            shot_number = int(gfile_list[i][1:7])

        project_dict['details']['shot'] = shot_number
        project_dict['details']['shot_time'] = time_list
    else:
        os.system('cp ' + project_dict['details']['efit_master'] + '* ' + project_dict['details']['efit_dir'])
        dir_list = os.listdir(project_dict['details']['efit_dir'])

        #get shot number and time from the efit filename
        for i in range(0,len(dir_list)):
            if dir_list[i].find('.') == 7 and dir_list[i][0]=='g' and dir_list[i][-1]!='~':
                shot_number = int(dir_list[i][1:7])
                if dir_list[i].find('_')== -1:
                    shot_time = int(dir_list[i][8:])
                else:
                    shot_time = int(dir_list[i][8:dir_list[i].find('_')])
                print shot_number, shot_time

        project_dict['details']['shot'] = shot_number
        project_dict['details']['shot_time'] = shot_time

    #Output data structure for the next step
    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_initial_setup.pickle')

    #corsica_base_dir = '/scratch/haskeysr/corsica_test9/'
    print 'start corsica setup'

    #p_string,q_string,len_p_list = cont_funcs.corsica_qmult_pmult()

    #creating lots of directories for the various runs of CORSICA
    #CORSICA_run_on_venus=1
    #CORSICA_workers_tmp=10
    if multiple_efits == 1:
        for i in range(0, len(project_dict['details']['shot_time'])):
            cont_funcs.corsica_run_setup(corsica_base_dir, project_dict['details']['multiple_efit'][i], project_dict['details']['template_dir'] + CORSICA_template_name, [str(project_dict['details']['shot_time'][i])], corsica_settings[0], rm_files = CORSICA_rm_files)
        for i in range(0, len(project_dict['details']['shot_time'])):
            cont_funcs.corsica_multiple_efits(str(project_dict['details']['shot_time'][i]), project_dict, corsica_base_dir, rm_files = CORSICA_rm_files)
        #elif CORSICA_workers>1:
    elif CORSICA_template_name!=CORSICA_template_name2:
        #elif CORSICA_workers>1:
        #setup the prerun directory
        cont_funcs.corsica_run_setup(corsica_base_dir, project_dict['details']['efit_dir'],project_dict['details']['template_dir'] + CORSICA_template_name, ['prerun'], corsica_settings[0], rm_files = CORSICA_rm_files)
        #run prerun
        print "submitting scaling job"
        cont_funcs.corsica_qsub(corsica_base_dir + '/prerun/', 'corsica.job')
        #find out when prerun is finished
        print "start waiting for scaling job to finish..."
        cont_funcs.check_corsica_finished(corsica_base_dir + '/prerun/', "corsica_finished")
        print "scaling job finished"
        #read in all the different jobs to do
        #make all the seperate directories
        corsica_directory_list = []
        for i in range(0,CORSICA_workers): 
            corsica_directory_list.append("worker"+str(i))
            cont_funcs.corsica_run_setup(corsica_base_dir, project_dict['details']['efit_dir'],project_dict['details']['template_dir'] + CORSICA_template_name2, [corsica_directory_list[-1]], corsica_settings[0], rm_files = CORSICA_rm_files)
        print "runs have been setup pt1"
        cont_funcs.read_qmult_pmult_values(corsica_base_dir + '/prerun/', corsica_base_dir, corsica_directory_list)
        print "runs have been setup pt2"
        #run all the seperate directories
        for i in corsica_directory_list: cont_funcs.corsica_qsub(corsica_base_dir + '/'+i, 'corsica.job')
        print "run jobs submitted to venus"
        #check everything is finished
        print "waiting for jobs to finish..."
        for i in corsica_directory_list: cont_funcs.check_corsica_finished(corsica_base_dir + '/'+i, "corsica_finished")
        print "jobs finished, putting everything together in the efit dir"
        #combine everything
        cont_funcs.copy_files_combine_stab_setups(corsica_base_dir, corsica_directory_list, project_dict['details']['efit_dir'])
        print "finished corsica runs"
    else:
        corsica_list = [['ml10']] 
        #for i in range(0,len(corsica_list)):
        corsica_run_dir = 'run_dir'
        cont_funcs.corsica_run_setup(corsica_base_dir, project_dict['details']['efit_dir'],project_dict['details']['template_dir'] + CORSICA_template_name, [corsica_run_dir], corsica_settings[0], rm_files = CORSICA_rm_files)
        print 'finished corsica setup, starting corsica runs'

        #Cluster way of doing it
        cont_funcs.corsica_qsub(corsica_base_dir + '/{}/'.format(corsica_run_dir), 'corsica.job')
        cont_funcs.check_corsica_finished(corsica_base_dir + '/{}/'.format(corsica_run_dir), "corsica_finished")
        cont_funcs.copy_files_combine_stab_setups(corsica_base_dir, [corsica_run_dir], project_dict['details']['efit_dir'])

        #Old way of doing it
        #cont_funcs.corsica_batch_run(corsica_list, project_dict, corsica_base_dir, workers = 1, rm_files = CORSICA_rm_files)

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)

print '#####################################################################'
print '##***************************STEP 2 - Generate Directory Structure ############'
if start_from_step <=2 and end_at_step>=2:
    overall_start = time.time()
    if start_from_step ==2:
        project_dict = pyMARS_funcs.read_data(project_dir +project_name+'_initial_setup.pickle')

    #Read the stab_results file and create the serial numbers in the data structure for each equilibria
    if multiple_efits:
        master_dict = {}
        print project_dict['details']['shot_time']
        master_serial = 1
        for i,curr_dir_tmp in enumerate(project_dict['details']['multiple_efit']):
            file_location = curr_dir_tmp +'/stab_setup_results.dat'
            tmp_dict = pyMARS_funcs.read_stab_results(file_location)
            for j in tmp_dict.keys():
                master_dict[master_serial] = copy.deepcopy(tmp_dict[j])
                print i
                print project_dict['details']['shot_time'][i]
                #master_dict[master_serial]['time'] = 20000
                #print master_dict[master_serial]['time']
                master_dict[master_serial]['shot_time'] = project_dict['details']['shot_time'][i]
                master_serial += 1
        print master_dict
        project_dict['sims'] = copy.deepcopy(master_dict)
    else:
        file_location = project_dict['details']['efit_dir']+'/stab_setup_results.dat'
        project_dict['sims'] = pyMARS_funcs.read_stab_results(file_location)
  
    #Filter according to settings at top of file
    project_dict = cont_funcs.remove_certain_values(project_dict, q95_range, Bn_Div_Li_range, filter_WTOTN1, filter_WTOTN2, filter_WTOTN3, filter_WWTOTN1)


    #generate the directory structure for the project
    project_dict = cont_funcs.generate_directories_func(project_dict, project_dict['details']['base_dir'], multiple_efits = multiple_efits)

    #Dump the data structure so it can be read by the next step if required
    print 'STEP 2 : dumping data to pickle file'
    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir']+ project_name + '_setup_directories.pickle')

    print 'Total Time for step 2 (DIR_Setup) : %.2f'%((time.time()-overall_start)/60)


print '#####################################################################'
print '##***************************STEP 3 - CHEASE RUN********************#'
if start_from_step <=3 and end_at_step>=3:
    overall_start = time.time()
    if start_from_step ==3:
        project_dict = pyMARS_funcs.read_data(project_dict['details']['base_dir'] + project_name+'_setup_directories.pickle')

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
    #Set fxrun = 0 for MARSQ update -DBW
    project_dict = cont_funcs.chease_setup_run(project_dict,job_num_filename, CHEASE_execution_script,PEST=0, fxrun = 0, rm_files = CHEASE_rm_files, cluster_job = cluster_job)

    if include_chease_PEST_run ==1:
        project_dict = cont_funcs.chease_setup_run(project_dict,job_num_filename, CHEASE_execution_script, CHEASE_template = CHEASE_template_name, PEST=1, fxrun = 0, rm_files = CHEASE_PEST_rm_files, cluster_job = cluster_job)
        #Does FourierX need to be run ON THE PEST JOB??? ask Matt or Yueqiang

    #Dump the data structure so it can be read by the next step if required
    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_chease.pickle')
    print 'Total Time for step 3 (CHEASE) : %.2f'%((time.time()-overall_start)/60)


print '########################################################################'
print '##*************** STEP 4 - I-coil grid location *** *          #########'
## This step used to be performed by Matlab and RZplot from Liu ###
## Currently using my version of RZplot in Python - need to benchmark code more ##
## Need to add some switch to allow the Matlab option###################

if start_from_step <=4 and end_at_step>=4:
    overall_start = time.time()
    if start_from_step == 4:
        project_dict = pyMARS_funcs.read_data(project_dir + project_name+'_post_chease.pickle')
        #Need to apply new MARS settings from MARS_settings2 here:
    for i in project_dict['sims'].keys():
        print i
        for tmp_key in new_mars_settings.keys():
            print project_dict['sims'][i]['MARS_settings'][tmp_key],
            project_dict['sims'][i]['MARS_settings'][tmp_key] = new_mars_settings[tmp_key]
            print project_dict['sims'][i]['MARS_settings'][tmp_key]
    for tmp_key in new_mars_settings.keys():
        project_dict['details']['MARS_settings'][tmp_key] = new_mars_settings[tmp_key]

    RMZM_name = 'RMZM_F' #This is here so that you can choose between pest and this?

    project_dict = cont_funcs.check_chease_run(project_dict, RMZM_name)
    if RMZM_python == 1:
        project_dict = cont_funcs.RMZM_func(project_dict, coilN, RMZM_name)
    else:
        project_dict = cont_funcs.RMZM_func_matlab(project_dict, '/u/haskeysr/matlab/RZplot3/','/u/haskeysr/matlab/RZplot3/MacMainD3D_Master.m')
    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_RMZM.pickle')

    print 'Total Time for step 4 : %.2f'%((time.time()-overall_start)/60)


print '#########################################################################'
print '##*************************** Step 5 - MARS setup**************############'
if start_from_step <=5 and end_at_step>=5:
    overall_start = time.time()
    if start_from_step == 5:
        project_dict = pyMARS_funcs.read_data(project_dir + project_name+'_post_RMZM.pickle')

    project_dict = cont_funcs.setup_mars_func(project_dict, upper_and_lower = upper_and_lower, MARS_template_name = MARS_template_name, multiple_efits = multiple_efits, rot_scan_list=rot_scan_list,res_scan_list=res_scan_list)

    #Save the data structure so that it can be read by the next step
    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_setup.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


print '####################################################################'
print '##*************************** Step 6 - run MARS ********************########'
if start_from_step <=6 and end_at_step>=6:
    overall_start = time.time()
    if start_from_step == 6:
        project_dict = pyMARS_funcs.read_data(project_dir + project_name + '_post_setup.pickle')

    job_num_filename = project_dict['details']['base_dir']+'MARS_simul_jobs.txt'
    with file(job_num_filename,'w') as file_handle: file_handle.write('%d\n'%(MARS_simultaneous_jobs))

    project_dict = cont_funcs.run_mars_function(project_dict, job_num_filename, MARS_execution_script,rm_files = MARS_rm_files, rm_files2 = MARS_rm_files2, cluster_job = cluster_job, upper_and_lower = upper_and_lower, mem_requirement = MARS_memory)

    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir']+ project_name+'_post_mars_run.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)



print '####################################################################'
print '##***************************STEP 7 - Post Processing **********####'
## **Set this up to have the pickup coil details passed to it** ##
if start_from_step <=7 and end_at_step>=7:
    overall_start = time.time()
    if start_from_step == 7:
        print 'reading pickle_file'
        project_dict = pyMARS_funcs.read_data(project_dir + project_name + '_post_mars_run.pickle')
        project_dict['details']['pickup_coils'] = copy.deepcopy(pickup_coils_details)
        project_dict['details']['I-coils'] = copy.deepcopy(I_coil_details)
    serial_list = project_dict['sims'].keys()
    project_dict = cont_funcs.post_processing(project_dict, post_proc_simultaneous_jobs, post_proc_script, directory = 'post_proc_tmp/', upper_and_lower = upper_and_lower, cluster_job = cluster_job)

    #project_dict = cont_funcs.coil_outputs_B(project_dict,serial_list)

    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_processing.pickle')

    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)

print 'completion time : %.4f mins' %((time.time()-initial_start_time)/60)


print '####################################################################'
print '##**************** STEP 8 - MARS PEST Post Processing **********####'
if start_from_step <=8 and end_at_step>=8:
    overall_start = time.time()
    if start_from_step == 8:
        print 'reading pickle_file'
        tmp_filename = project_dir + project_name + '_post_processing.pickle'
        print tmp_filename
        project_dict = pyMARS_funcs.read_data(project_dir + project_name + '_post_processing.pickle')
        project_dict['details']['pickup_coils'] = copy.deepcopy(pickup_coils_details)
        project_dict['details']['I-coils'] = copy.deepcopy(I_coil_details)

    #serial_list = project_dict['sims'].keys()
    #post_proc_script = '/u/haskeysr/code/NAMP_analysis/python/pyMARS/post_proc_script_PEST.py'
    working_directory = 'post_proc_tmp2/'
    os.system('mkdir ' + project_dir + working_directory)
    project_dict = cont_funcs.post_processing(project_dict, post_proc_simultaneous_jobs, post_proc_script_PEST, directory = working_directory, upper_and_lower = upper_and_lower, cluster_job = cluster_job)

    #project_dict = cont_funcs.coil_outputs_B(project_dict,serial_list)
    pyMARS_funcs.dump_data(project_dict, project_dict['details']['base_dir'] + project_name+'_post_processing_PEST.pickle')
    print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)


