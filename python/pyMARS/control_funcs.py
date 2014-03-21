#!/usr/bin/env Python
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import time, pickle, copy, os
import pyMARS.Batch_Launcher as batch_launcher
import pyMARS.RZfuncs as RZfuncs
import pyMARS.results_class
import multiprocessing
import numpy as num

######################################################
#step 1
def generate_master_dir(master, project_dict):
    #master['shot_dir'] =  project_dict['details']['base_dir'] + 'shot' + str(master['shot'])+'/'
    master['shot_dir'] =  project_dict['details']['base_dir']
    os.system('mkdir ' + master['shot_dir'])
    master['thetac_dir'] = master['shot_dir']# + 'tc_%03d/'%(master['thetac']*1000)
    os.system('mkdir ' + master['thetac_dir'])
    master['efit_dir'] =  master['thetac_dir']  + 'efit/'
    os.system('mkdir ' + master['efit_dir'])
    return master



def find_relevant_efit_files(efit_directory, profile_directory):
    directory_listing = os.listdir(efit_directory)
    print directory_listing
    time_list = []
    gfile_list = []
    for i in directory_listing:
        if i[0]=='g' and i.find('.') == 7:
            time_list.append(int(i.split('.')[1]))
            gfile_list.append(i)
    for i in gfile_list:
        pass
    #Make sure all the files we want exist!

    return time_list, gfile_list



######################################################
############## CORSICA FUNCTIONS ###############
def copy_files_combine_stab_setups(base_directory, new_directories, final_directory):
    '''Copy the files from each individual CORSICA run and combine
    them into a single final directory and create a single stab_setup_results.dat
    that contains information on each of the scalings
    SH: Feb 18 2013
    '''
    overal_list = []
    first_time = 1
    for i in new_directories:
        #copy EXPEQ files across
        os.system('cp '+base_directory+'/'+i+'/EXPEQ* ' + final_directory)
        #read in each individual stab_setup_results file
        tmp_file = file(base_directory+'/'+i+'/stab_setup_results.dat','r')
        #first time to include the header
        if first_time:
            tmp_data = tmp_file.readlines()
            first_time = 0
        else:
            tmp_data = tmp_file.readlines()[3:]
        tmp_file.close()
        for j in tmp_data:
            overal_list.append(j)

    #output the final stab_setup_results file
    tmp_file = file(final_directory+'/stab_setup_results.dat','w')
    tmp_file.writelines(overal_list)
    tmp_file.close()
    

def corsica_qsub(qsub_directory, qsub_file):
    '''qsub a corsica job
    SH: Feb 18 2013
    '''
    os.chdir(qsub_directory)
    os.system('qsub '+ qsub_file)

def read_qmult_pmult_values(prerun_dir, base_directory, new_directories):
    '''read in the list of pmult and qmult values then distribute them
    evenly between the new_directories which are going to be the working
    directories of each seperate job on venus
    prerun_dir : the location of the CORSICA directory when all the scalings were calculated
    base_dir : base corsica working directory
    new_directories : listing of final directory name for all the worker directories
    SH: Feb 18 2013
    '''

    pmult_file = file(prerun_dir + '/pmult_values','r')
    qmult_file = file(prerun_dir + '/qmult_values','r')
    pmult_values = pmult_file.readlines()
    qmult_values = qmult_file.readlines()
    pmult_file.close()
    qmult_file.close()
    loc = 0
    while len(pmult_values)>0:
        cur_pmult_file = file(base_directory + '/' + new_directories[loc] + '/pmult_values', 'a')
        cur_qmult_file = file(base_directory + '/' + new_directories[loc] + '/qmult_values', 'a')
        cur_pmult_file.write(pmult_values.pop(0))
        cur_qmult_file.write(qmult_values.pop(0))
        cur_pmult_file.close()
        cur_qmult_file.close()
        loc+=1
        if loc==len(new_directories):
            loc=0
    
def check_corsica_finished(corsica_directory, filename):
    '''check to see if the CORSICA run has finished 
    SH: Feb 18 2013
    '''
    while not os.path.exists(corsica_directory + '/' + filename):
        time.sleep(5)
    

def corsica_run_setup(base_dir, efit_dir, template_file, input_data, settings):
    running_dir = base_dir + input_data[0]
    if not os.path.exists(base_dir):
        print 'project directory doesnt exist - creating...',
        os.system('mkdir ' + base_dir)
        print 'done'
    os.system('mkdir ' + running_dir)
    os.chdir(running_dir)

    os.system('cp ' + efit_dir + '* .')

    command_file = open('commands_test.txt','w')
    command_file.write('read "sspqi_sh_this_dir.bas"\nquit\n')
    command_file.close()

    template_file = open(template_file,'r')
    template_text = template_file.read()
    template_file.close()

    #make all the changes
    for i in settings.keys():
        #print i, settings[i]
        template_text = template_text.replace(i,str(settings[i]))

    template_file = open('sspqi_sh_this_dir.bas','w')
    template_file.write(template_text)
    template_file.close()
    try:
        os.remove('corsica_finished')
    except OSError:
        pass
    #os.system('rm corsica_finished')
    corsica_job_string = "#!/bin/bash\n#$ -N corsica_SH\n#$ -q all.q\n#$ -o sge_output.dat\n#$ -e sge_error.dat\n#$ -cwd\n"
    corsica_job_string += "rm corsica_finished\n"
    corsica_job_string += "/d/caltrans/vcaltrans/bin/caltrans -probname eq_vary_p_q < commands_test.txt >caltrans_out.log\n"
    corsica_job_string += "touch corsica_finished\n"
    with file('corsica.job','w') as corsica_job_file:
        #corsica_job_file = open('corsica.job','w')
        corsica_job_file.write(corsica_job_string)
        #corsica_job_file.close()
        
def run_corsica_file(base_dir, input_list, script_name, sleep_time = 1):
    script = 'sleep %d\n'%(sleep_time)
    for input_data in input_list:#len(list)):
        running_dir = base_dir + input_data[0]
        script += 'cd ' + running_dir + '\n'
        script += 'pwd\n'
        command = '/d/caltrans/vcaltrans/bin/caltrans -probname eq_vary_p_q < commands_test.txt > caltrans_out.log'
        script += 'sleep %d\n'%(sleep_time)
        script += command + '\n'
        script += 'sleep %d\n'%(sleep_time)
    script += 'sleep %d\n'%(sleep_time)
    script += 'exit\nexit\n'
    file_name = open(base_dir + script_name,'w')
    file_name.write(script)
    file_name.close()

def corsica_batch_qrsh_func(command):
    os.system(command)
            
def corsica_batch_run_qrsh(corsica_list, project_dict, corsica_base_dir, workers = 1):
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
        run_corsica_file(corsica_base_dir, send_list, 'corsica_script'+str(i)+'.sh', sleep_time = 30)
        script_file_list.append('corsica_script' + str(i)+'.sh')
    process_list = []
    for i in script_file_list:
        command = 'qrsh < ' + corsica_base_dir + i + '>'+corsica_base_dir+'log'+i
        print 'running ' + i + ' script, command is', command
        process_curr = multiprocessing.Process(target=corsica_batch_qrsh_func,args=([command]))
        process_curr.start()
        process_list.append(process_curr)
        time.sleep(5)

    print 'processes should have started'
    for i in range(0,len(process_list)):
        while process_list[i].is_alive():
            time.sleep(10)
            print 'waiting for ', i, ' to finish'

    #need to stich everything together here!!
    #for i in corsica_list:
    #    print 'copying files across'
    #    os.system('cp ' + corsica_base_dir + i[0] + '/EXPEQ* stab_setup* ' + project_dict['details']['efit_dir'])
    #os.system('rm -r ' +corsica_base_dir)



def corsica_batch_run(corsica_list, project_dict, corsica_base_dir, workers = 1):
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
        run_corsica_file(corsica_base_dir, send_list, 'corsica_script'+str(i)+'.sh', sleep_time = 1)
        script_file_list.append('corsica_script' + str(i)+'.sh')
    for i in script_file_list:
        print 'running ' + i + ' script'
        os.system('source ' + corsica_base_dir + i)
    for i in corsica_list:
        print 'copying files across'
        os.system('cp ' + corsica_base_dir + i[0] + '/EXPEQ* stab_setup* ' + project_dict['details']['efit_dir'])
    #os.system('rm -r ' +corsica_base_dir)


def corsica_multiple_efits(efit_time, project_dict, corsica_base_dir):
    script_file = 'corsica_script'+str(efit_time)+'.sh'
    run_corsica_file(corsica_base_dir, [[efit_time]], script_file, sleep_time = 1)
    os.system('source ' + corsica_base_dir + script_file)
    os.system('cp ' + corsica_base_dir + '/'+str(efit_time) + '/EXPEQ* ' + project_dict['details']['efit_dir']+'/'+efit_time+'/')
    os.system('cp ' + corsica_base_dir + '/'+str(efit_time) + '/stab_setup_results.dat ' + project_dict['details']['efit_dir']+'/'+efit_time+'/')


###################################################################
#step 2

def remove_certain_values(project_dict, q95_range, Bn_Div_Li_range, filter_WTOTN1, filter_WTOTN2, filter_WTOTN3, filter_WWTOTN1):
    removed_q95 = 0;removed_Bn = 0;removed_DCON = 0;removed_read_error = 0
    for i in project_dict['sims'].keys():
        current_q95 = project_dict['sims'][i]['Q95']
        current_Bn_Div_Li = project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI']

        #This is so that the script doesn't die on some strange error, must disable these if you are trying
        #to debug what is happening***!!!
        try:
            WTOTN1 = project_dict['sims'][i]['WTOTN1']
            WTOTN2 = project_dict['sims'][i]['WTOTN2']
            WTOTN3 = project_dict['sims'][i]['WTOTN3']
            WWTOTN1 = project_dict['sims'][i]['WWTOTN1']

            #Filter for relevant q95
            if (current_q95<q95_range[0]) or (current_q95>q95_range[1]):
                del project_dict['sims'][i]
                print 'removed item q95 out of range'
                removed_q95 += 1

            #Filter for relevant Beta_n/Li
            elif (current_Bn_Div_Li<Bn_Div_Li_range[0]) or (current_Bn_Div_Li>Bn_Div_Li_range[1]):
                del project_dict['sims'][i]
                print 'removed item Bn_Div_Li out of range'
                removed_Bn += 1

            #Filter for DCON failure
            elif (WTOTN1<=0 and filter_WTOTN1==1) or  (WTOTN2<=0 and filter_WTOTN2==1)  or (WTOTN3<=0 and filter_WTOTN3==1) or (WWTOTN1<=0 and filter_WWTOTN1==1):
                del project_dict['sims'][i]
                print 'removed item due to stability'
                removed_DCON += 1
            else:
                pass
        except:
            del project_dict['sims'][i]
            print 'removed due to error reading'
            removed_read_error += 1

    print 'Removed %d for q95, %d for Bn_Div_Li, %d for stability, and %d due to read error'%(removed_q95, removed_Bn, removed_DCON, removed_read_error)
    print 'Remaining eq : %d'%(len(project_dict['sims'].keys()))
    return project_dict

def generate_directories_func(project_dict, base_dir, multiple_efits = 0):
    print multiple_efits
    for i in project_dict['sims'].keys():
        print 'generate_directories serial : ', i
        project_dict['sims'][i]['shot']=project_dict['details']['shot']
        if multiple_efits!=0:
            pass
        else:
            project_dict['sims'][i]['shot_time']=project_dict['details']['shot_time']
        project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%d_q%d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*1000)),int(round(project_dict['sims'][i]['QMULT']*1000)))

        project_dict['sims'][i]['ICOIL_FREQ'] = project_dict['details']['ICOIL_FREQ']

        #possible here is an opportunity to vary settings between runs??
        project_dict['sims'][i]['CHEASE_settings'] = copy.deepcopy(project_dict['details']['CHEASE_settings'])
        project_dict['sims'][i]['CHEASE_settings_PEST'] = copy.deepcopy(project_dict['details']['CHEASE_settings_PEST'])
        project_dict['sims'][i]['MARS_settings'] = copy.deepcopy(project_dict['details']['MARS_settings'])
        project_dict['sims'][i]['corsica_settings'] = copy.deepcopy(project_dict['details']['corsica_settings'])
        project_dict['sims'][i]['I-coils'] = project_dict['details']['I-coils']
        project_dict['sims'][i] = pyMARS_funcs.generate_directories(project_dict['sims'][i],base_dir, multiple_efits = multiple_efits)

    return project_dict


def generate_directories_func_serial(project_dict, base_dir):
    for i in project_dict['sims'].keys():
        print 'generate_directories serial : ', i
        project_dict['sims'][i]['shot']=project_dict['details']['shot']
        project_dict['sims'][i]['shot_time']=project_dict['details']['shot_time']
        project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_%d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],i)

        project_dict['sims'][i]['ICOIL_FREQ'] = project_dict['details']['ICOIL_FREQ']

        #possible here is an opportunity to vary settings between runs??
        project_dict['sims'][i]['CHEASE_settings'] = copy.deepcopy(project_dict['details']['CHEASE_settings'])
        project_dict['sims'][i]['CHEASE_settings_PEST'] = copy.deepcopy(project_dict['details']['CHEASE_settings_PEST'])
        project_dict['sims'][i]['MARS_settings'] = copy.deepcopy(project_dict['details']['MARS_settings'])
        project_dict['sims'][i]['corsica_settings'] = copy.deepcopy(project_dict['details']['corsica_settings'])
        project_dict['sims'][i]['I-coils'] = project_dict['details']['I-coils']
        project_dict['sims'][i] = pyMARS_funcs.generate_directories(project_dict['sims'][i],base_dir)

    return project_dict

###################################################################
#step 3
def chease_setup_run(project_dict, job_num_filename, CHEASE_execution_script,CHEASE_template = 'datain_template',PEST=0,fxrun = 0, rm_files = '', cluster_job =1):
    identifier_string = 'CH' + str(int(num.random.rand(1)*100)) #to identify which jobs have been submitted to venus
    for i in project_dict['sims'].keys():
        print i
        #copy the Chease template
        pyMARS_funcs.copy_chease_files(project_dict['sims'][i], PEST = PEST)
        #modify the datain so it is relevant for this project
        project_dict['sims'][i] = pyMARS_funcs.modify_datain(project_dict['sims'][i],project_dict['details']['template_dir'], project_dict['sims'][i]['CHEASE_settings'], CHEASE_template = CHEASE_template, PEST = PEST)
        #modify_datain(project_dict['sims'][i],project_dict['details']['template_dir'], PEST = PEST)
        #generate a job file that can be submitted to the cluster

        tmp = project_dict['sims'][i]
        if PEST==0:
            pyMARS_funcs.fxin_create(tmp['dir_dict']['chease_dir'], tmp['MARS_settings']['<<M1>>'],tmp['MARS_settings']['<<M2>>'],project_dict['sims'][i]['R0EXP'], project_dict['sims'][i]['B0EXP'])
        else:
            pyMARS_funcs.fxin_create(tmp['dir_dict']['chease_dir_PEST'], tmp['MARS_settings']['<<M1>>'],tmp['MARS_settings']['<<M2>>'],project_dict['sims'][i]['R0EXP'], project_dict['sims'][i]['B0EXP'])
            
        pyMARS_funcs.generate_chease_job_file(project_dict['sims'][i], CHEASE_execution_script, PEST = PEST, fxrun = fxrun, id_string = identifier_string,rm_files = rm_files)


    #This is the step that launches the batch job
    if cluster_job == 0:
        project_dict['sims'] = batch_launcher.single_launch_chease(project_dict['sims'],PEST=0)
    else:
        project_dict['sims'] = batch_launcher.batch_launch_chease(project_dict['sims'], job_num_filename, PEST = PEST, id_string = identifier_string)
    return project_dict


###################################################################
#step 4
#could implement this as part of the chease run and get multiproc benefits from it
def check_chease_run(project_dict, RMZM_name):
    for i in project_dict['sims'].keys():
        if not os.path.exists(project_dict['sims'][i]['dir_dict']['chease_dir'] + RMZM_name):
            print '%d chease run unsuccessful, dir: %s, removing from dict...' %(i, project_dict['sims'][i]['dir_dict']['chease_dir'])
            del project_dict['sims'][i]
    return project_dict

def RMZM_func(project_dict, coilN, RMZM_name):
    start_time = time.time()
    total_jobs = len(project_dict['sims'].keys())
    total_finished = 0
    failed = 0
    for i in project_dict['sims'].keys():
        RMZMFILE = project_dict['sims'][i]['dir_dict']['chease_dir'] + RMZM_name
        Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
        #try:
        print coilN, RMZMFILE, Nchi
        FCCHI, FWCHI, IFEED = RZfuncs.Icoil_MARS_grid_details(coilN,RMZMFILE,Nchi)
        print i, ' FCCHI=',FCCHI, ' FWCHI=',FWCHI,' IFEED=',IFEED,' time:',time.time()-start_time,'s, finished ',total_finished,' of ',total_jobs
        project_dict['sims'][i]['FCCHI'] = FCCHI
        project_dict['sims'][i]['FWCHI'] = FWCHI
        project_dict['sims'][i]['IFEED'] = IFEED
        #except:
        #print ' RMZM ERROR on serial ', i, ' deleting from dictionary'
        #print ' directory is ', project_dict['sims'][i]['dir_dict']['chease_dir']
        #del project_dict['sims'][i]
        #failed+=1
        total_finished+=1
    print 'finished RMZM calculations, errors : ', failed
    return project_dict


def RMZM_func_matlab(project_dict, matlab_RZplot_files, main_template):
    mat_commands = "addpath('"+ matlab_RZplot_files + "')\n"
    start_time = time.time()
    total_jobs = len(project_dict['sims'].keys())
    total_finished = 0

    for i in project_dict['sims'].keys():
        print i

        #copy a MacMain.m script to the chease dir and modify to perform the job we require
        project_dict['sims'][i] = pyMARS_funcs.modify_RMZM_F2(project_dict['sims'][i], main_template)

        mat_commands += 'close all;clear all;\n'
        #Go to correct dir
        mat_commands += 'cd ' + project_dict['sims'][i]['dir_dict']['chease_dir']+'\n'
        #Record output to obtain results later
        mat_commands += "diary('testing_output')\n"
        mat_commands += "diary on\n"
        #Track what has been done
        mat_commands += "disp('Finished %d of %d')\n"%(total_finished, total_jobs)
        #Run the script that does the calculations
        mat_commands += "MacMainD3D_current\n"
        #Save the picture it creates so it can be viewed if required
        mat_commands += "print(gcf,'-dpng','p" +str(project_dict['sims'][i]['PMULT'])+'_q'+str(project_dict['sims'][i]['QMULT']) +".png')\n"
        mat_commands += "diary off\n"

        total_finished += 1
        print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)

    mat_commands += "quit\n"

    #Write the list of Matlab commands to a file
    os.chdir(project_dict['details']['base_dir'])
    file = open('matlab_commands.txt','w')
    file.write(mat_commands)
    file.close()

    #Run Matlab
    os.chdir(project_dict['details']['base_dir'])
    #Not sure if this is the best way to run Matlab - may be better to use -r
    os.system('matlab -nodesktop -nodisplay < matlab_commands.txt')

    #Retrieve the results from the Matlab run
    for i in project_dict['sims'].keys():
        print i
        project_dict['sims'][i] = pyMARS_funcs.RMZM_post_matlab(project_dict['sims'][i])

    return project_dict

###################################################################
#step 5
def setup_mars_func(project_dict, upper_and_lower = 0, MARS_template_name = 'RUN_template', multiple_efits = 0, rot_scan_list = None, res_scan_list = None):
    print '!!!!!!', os.getcwd()
    if res_scan_list!=None or rot_scan_list!=None:
        project_dict_new = copy.deepcopy(project_dict)
        project_dict_new['sims'] = {}
        new_key = 1
        for i in project_dict['sims'].keys():
            if res_scan_list == None: res_scan_list = [+project_dict['sims'][i]['MARS_settings']['<<ETA>>']]
            if rot_scan_list == None: rot_scan_list = [+project_dict['sims'][i]['MARS_settings']['<<ROTE>>']]
            for eta in res_scan_list:
                for ROTE in rot_scan_list:
                    project_dict_new['sims'][new_key] = copy.deepcopy(project_dict['sims'][i])
                    project_dict_new['sims'][new_key]['MARS_settings']['<<ETA>>'] = +eta
                    project_dict_new['sims'][new_key]['MARS_settings']['<<ROTE>>'] = +ROTE
                    new_key+=1
        project_dict = project_dict_new
#     if res_scan_list!=None:
#         project_dict_new = copy.deepcopy(project_dict)
#         project_dict_new['sims'] = {}
#         new_key = 1
#         for i in project_dict['sims'].keys():
#             for eta in res_scan_list:
#                 project_dict_new['sims'][new_key] = copy.deepcopy(project_dict['sims'][i])
#                 project_dict_new['sims'][new_key]['MARS_settings']['<<ETA>>'] = +eta
#                 new_key+=1
#         project_dict = project_dict_new
#     elif rot_scan_list!=None:
#         project_dict_new = copy.deepcopy(project_dict)
#         project_dict_new['sims'] = {}
#         new_key = 1
#         for i in project_dict['sims'].keys():
#             for ROTE in rot_scan_list:
#                 project_dict_new['sims'][new_key] = copy.deepcopy(project_dict['sims'][i])
#                 project_dict_new['sims'][new_key]['MARS_settings']['<<ROTE>>'] = +ROTE
#                 new_key+=1
#         project_dict = project_dict_new

    for i in project_dict['sims'].keys():
        print i
        print '!!! before NW', os.getcwd()
        #Extract required values from CHEASE log file
        project_dict['sims'][i] = pyMARS_funcs.extract_NW(project_dict['sims'][i])
        print '!!! after NW', os.getcwd()

        #Link required files
        if multiple_efits:
            special_dir = str(project_dict['sims'][i]['shot_time'])
        else:
            special_dir = ''
        pyMARS_funcs.mars_setup_files(project_dict['sims'][i], upper_and_lower = upper_and_lower, special_dir = special_dir)
        print '!!! mars_setup_files', os.getcwd()
        #pyMARS_funcs.mars_setup_files(project_dict['sims'][i], vac = 0, upper_and_lower = upper_and_lower)

        #Calculate the values that need to be normalised to something to do with Alfven speed/frequency
        project_dict['sims'][i] = pyMARS_funcs.mars_setup_alfven(project_dict['sims'][i], project_dict['sims'][i]['ICOIL_FREQ'], upper_and_lower = upper_and_lower)
        print '!!! after mars_setup_alfven', os.getcwd()

        #Create the MARS RUN files
        project_dict['sims'][i] = pyMARS_funcs.mars_setup_run_file_new(project_dict['sims'][i], project_dict['details']['template_dir'] + MARS_template_name, upper_and_lower = upper_and_lower)
        #project_dict['sims'][i] = pyMARS_funcs.mars_setup_run_file_new(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 0, upper_and_lower = upper_and_lower)

        #generate_job_file(project_dict['sims'][i],1) #create Venus cluster job file
        #mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 0)
        #generate_job_file(project_dict['sims'][i],0) #create Venus cluster job file
    return project_dict




###################################################################
#step 6
def run_mars_function(project_dict,job_num_filename, MARS_execution_script,rm_files = '', rm_files2 = '', cluster_job =1, upper_and_lower = 0):
    id_string = 'MAR' + str(int(num.random.rand(1)*100)) #to identify which jobs have been submitted to venus
    for i in project_dict['sims'].keys():
        pyMARS_funcs.generate_job_file(project_dict['sims'][i], MARS_execution_script,id_string = id_string, rm_files = rm_files, rm_files2 = rm_files2, upper_and_lower=upper_and_lower)

        #pyMARS_funcs.generate_job_file(project_dict['sims'][i],1, id_string = id_string, rm_files = rm_files) #create Venus cluster job file
        #pyMARS_funcs.generate_job_file(project_dict['sims'][i],0, id_string = id_string, rm_files = rm_files) #create Venus cluster job file

    if cluster_job == 0:
        project_dict['sims'] = batch_launcher.single_launch_mars(project_dict['sims'])
    else:
        project_dict['sims'] = batch_launcher.batch_launch_mars(project_dict['sims'], job_num_filename, id_string = id_string)
    return project_dict




###################################################################
#step8
def post_processing(master_pickle, post_proc_workers, python_file, directory = 'post_proc_tmp/', upper_and_lower = 0, cluster_job = 0):
    start_time = time.time()
    pickle_file_list = []; worker_serial_list = []
    serial_list = master_pickle['sims'].keys()

    if cluster_job == 0:
        post_proc_workers =  1
    else:
        if len(serial_list)<post_proc_workers:
            print 'changing number of post_proc workers'
            post_proc_workers = len(serial_list)
            
    for i in range(0,post_proc_workers):
        worker_serial_list.append([])

    #split up the jobs
    for i in range(0, len(serial_list)):
        worker_serial_list[i%post_proc_workers].append(serial_list[i])

    os.system('rm -r '+master_pickle['details']['base_dir']+directory) #remove temp directory contents
    os.system('mkdir '+master_pickle['details']['base_dir']+directory) #remove temp directory contents

    #setup the serial numbers for each worker
    for i in range(0,post_proc_workers):
        tmp_pickle = {}; tmp_pickle['details']=copy.deepcopy(master_pickle['details']); tmp_pickle['sims']={}
        for jjj in worker_serial_list[i]:
            tmp_pickle['sims'][jjj]=copy.deepcopy(master_pickle['sims'][jjj])

        pickle_file_name = master_pickle['details']['base_dir']+directory+'/tmp_'+str(i)+'.pickle'
        pickle.dump(tmp_pickle, open(pickle_file_name,'w'))

        job_string = '#!/bin/bash\n#$ -N '+ 'PostProc'+str(i)+'\n#$ -q all.q\n#$ -o sge_output.dat\n#$ -e sge_error.dat\n#$ -cwd\nexport PATH=$PATH:/f/python/linux64/bin\nsource ~/.bashrc\n'
        log_file_name = master_pickle['details']['base_dir']+directory + '/log_test' + str(i) + '.log'
        #execute_command = python_file + ' ' + pickle_file_name + ' '  ' > ' + log_file_name + '\n'
        execute_command = '%s %s %d > %s\n'%(python_file,pickle_file_name, upper_and_lower,log_file_name)
        job_string += execute_command

        job_name = master_pickle['details']['base_dir'] + directory+'/step9_'+str(i)+'.job'
        job_file = open(job_name,'w')
        job_file.write(job_string)
        job_file.close()
        pickle_file_list.append(pickle_file_name)

        if cluster_job ==0:
            os.system('source ' + job_name)
        else:
            os.system('qsub ' + job_name)

    for i in pickle_file_list:
        while os.path.exists(i+'output') != True:
            print 'waiting for ',i
            time.sleep(5)
        print 'finished ', i

    combined_answer = {}
    combined_answer['details']= copy.deepcopy(master_pickle['details'])
    combined_answer['sims']={}
    for i in pickle_file_list:
        current_tmp = pickle.load(open(i+'output'))
        current_serials = current_tmp['sims'].keys()
        for jjj in current_serials:
            combined_answer['sims'][jjj]=copy.deepcopy(current_tmp['sims'][jjj])
    return combined_answer
