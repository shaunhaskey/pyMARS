#!/usr/bin/env Python
from PythonMARS_funcs import *
import time, pickle, copy
import Chease_Batch_Launcher as ch_launch
from RZfuncs import *
import results_class

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

#step 2
def generate_directories_func(project_dict, base_dir):
    for i in project_dict['sims'].keys():
        print i
        project_dict['sims'][i]['shot']=project_dict['details']['shot']
        project_dict['sims'][i]['shot_time']=project_dict['details']['shot_time']
        project_dict['sims'][i]['thetac']=project_dict['details']['thetac']
        #project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%.3d_q%.3d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*100)),int(round(project_dict['sims'][i]['QMULT']*100)))
        project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%d_q%d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*1000)),int(round(project_dict['sims'][i]['QMULT']*1000)))
        #project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%03d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*100))) #For Benchmark

        project_dict['sims'][i]['M1'] = project_dict['details']['M1']
        project_dict['sims'][i]['M2'] = project_dict['details']['M2']
        project_dict['sims'][i]['NTOR'] = project_dict['details']['NTOR']
        project_dict['sims'][i]['ICOIL_FREQ'] = project_dict['details']['ICOIL_FREQ']
        project_dict['sims'][i]['FEEDI'] = project_dict['details']['FEEDI']
        project_dict['sims'][i] = generate_directories(project_dict['sims'][i],base_dir)
    return project_dict

def remove_certain_values(project_dict, q95_range, Bn_Div_Li_range, filter_WTOTN1, filter_WTOTN2, filter_WTOTN3, filter_WWTOTN1):
    removed_q95 = 0;removed_Bn = 0;removed_DCON = 0;removed_read_error = 0
    for i in project_dict['sims'].keys():
        current_q95 = project_dict['sims'][i]['Q95']
        current_Bn_Div_Li = project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI']

        #This is so that the script doesn't die on some strange error, must disable these if you are trying
        #to debug what is happening
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

#step 3
def chease_setup_run(project_dict,simultaneous_jobs):
    for i in project_dict['sims'].keys():
        print i
        #copy the Chease template
        copy_chease_files(project_dict['sims'][i])
        #modify the datain so it is relevant for this project
        modify_datain(project_dict['sims'][i],project_dict['details']['template_dir'])
        #generate a job file that can be submitted to the cluster
        generate_chease_job_file(project_dict['sims'][i])

    #This is the step that launches the batch job
    project_dict['sims'] = ch_launch.batch_launch_chease(project_dict['sims'], simultaneous_jobs)
    return project_dict

#step 4
def setup_run_fxrun_func(project_dict):
    total_finished = 0
    start_time = time.time()
    total_jobs = len(project_dict['sims'].keys())
    for i in project_dict['sims'].keys():

        #Create the fxin file
        fxin_create(project_dict['sims'][i])

        #Run fxrun
        fxrun(project_dict['sims'][i])
        project_dict['sims'][i]['fxrun']=1
        total_finished += 1

        print 'Finished %d of %d, %.2fmins'%(total_finished, total_jobs, (time.time()-start_time)/60)
    return project_dict


#step 5
def RMZM_func(project_dict, coilN, RMZM_name, Nchi):
    start_time = time.time()
    total_jobs = len(project_dict['sims'].keys())
    total_finished = 0
    for i in project_dict['sims'].keys():
        RMZMFILE = project_dict['sims'][i]['dir_dict']['chease_dir'] + RMZM_name
        FCCHI, FWCHI, IFEED = Icoil_MARS_grid_details(coilN,RMZMFILE,Nchi)
        total_finished+=1
        #left over code for testing purposes
        #FCCHI_old =  project_dict['sims'][i]['FCCHI']
        #FWCHI_old = project_dict['sims'][i]['FWCHI']
        #IFEED_old = project_dict['sims'][i]['IFEED']
        #print i, ' FCCHI=',FCCHI_old, ' FWCHI=',FWCHI_old,' IFEED=',IFEED_old,' time:',time.time()-overall_start,'s, finished ',total_finished,' of ',total_jobs
        print i, ' FCCHI=',FCCHI, ' FWCHI=',FWCHI,' IFEED=',IFEED,' time:',time.time()-start_time,'s, finished ',total_finished,' of ',total_jobs
        project_dict['sims'][i]['FCCHI'] = FCCHI
        project_dict['sims'][i]['FWCHI'] = FWCHI
        project_dict['sims'][i]['IFEED'] = IFEED
    return project_dict


#step 6
def setup_mars_func(project_dict):
    for i in project_dict['sims'].keys():
        print i
        #Extract required values from CHEASE log file
        project_dict['sims'][i] = extract_NW(project_dict['sims'][i])

        #Setup MARS vacuum run
        mars_setup_files(project_dict['sims'][i], vac = 1)

        #Calculate the values that need to be normalised to something to do with Alfven speed/frequency
        project_dict['sims'][i] = mars_setup_alfven(project_dict['sims'][i], project_dict['sims'][i]['ICOIL_FREQ'], vac = 1)

        mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 1)
        generate_job_file(project_dict['sims'][i],1) #create Venus cluster job file

        #Setup MARS plasma run
        mars_setup_files(project_dict['sims'][i], vac = 0)
        mars_setup_run_file(project_dict['sims'][i], project_dict['details']['template_dir'], vac = 0)
        generate_job_file(project_dict['sims'][i],0) #create Venus cluster job file

    return project_dict

#step 7
def run_mars_function(project_dict,simultaneous_jobs):
    project_dict['sims'] = ch_launch.batch_launch_mars(project_dict['sims'],simultaneous_jobs)
    return project_dict

#step8

def coil_outputs_B(project_dict,serial_list):
    fails = 0
    passes = 0
    total_finished = 0
    total_jobs = len(serial_list)*2
    for i in serial_list:
        for type in ['plasma', 'vac']:
            if type == 'plasma':
                dir = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
            else:
                dir = project_dict['sims'][i]['dir_dict']['mars_vac_dir']
            print i, type
            new_data = results_class.data(dir,Nchi=240,link_RMZM=0)
            new_data_R = new_data.R*new_data.R0EXP
            new_data_Z = new_data.Z*new_data.R0EXP
            new_answer = num.array(coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi))
            
            #comp_answer = new_answer[0:-2]
            comp_answer = new_answer *1
            if type == 'plasma':
                project_dict['sims'][i]['plasma_response4'] = new_answer
            else:
                project_dict['sims'][i]['vacuum_response4'] = new_answer
            del new_data
    return project_dict

def cluster_coil_outputs_B(project_dict,serial_list, output_file):
    project_dict = coil_outputs_B(project_dict,serial_list)
    pickle.dump(project_dict, open(output_file,'w'))
    



def post_processing(master_pickle, post_proc_workers, python_file):
    start_time = time.time()
    pickle_file_list = []; worker_serial_list = []

    for i in range(0,post_proc_workers):
        worker_serial_list.append([])

    #split up the jobs
    serial_list = master_pickle['sims'].keys()
    for i in range(0, len(serial_list)):
        worker_serial_list[i%post_proc_workers].append(serial_list[i])

    os.system('rm -r '+master_pickle['details']['base_dir']+'post_proc_tmp/') #remove temp directory contents
    os.system('mkdir '+master_pickle['details']['base_dir']+'post_proc_tmp/') #remove temp directory contents

    #setup the job for each worker
    for i in range(0,post_proc_workers):
        tmp_pickle = {}; tmp_pickle['details']=copy.deepcopy(master_pickle['details']); tmp_pickle['sims']={}
        for jjj in worker_serial_list[i]:
            tmp_pickle['sims'][jjj]=copy.deepcopy(master_pickle['sims'][jjj])

        pickle_file_name = master_pickle['details']['base_dir']+'post_proc_tmp/tmp_'+str(i)+'.pickle'
        pickle.dump(tmp_pickle, open(pickle_file_name,'w'))

        job_string = '#!/bin/bash\n#$ -N blah\n#$ -q all.q\n#$ -o sge_output.dat\n#$ -e sge_error.dat\n#$ -cwd\nexport PATH=$PATH:/f/python/linux64/bin\n'
        log_file_name = master_pickle['details']['base_dir']+'post_proc_tmp/' + 'log_test' + str(i) + '.log'
        execute_command = python_file + ' ' + pickle_file_name + ' > ' + log_file_name + '\n'
        job_string += execute_command

        job_name = master_pickle['details']['base_dir'] + 'post_proc_tmp/step9_'+str(i)+'.job'
        job_file = open(job_name,'w')
        job_file.write(job_string)
        job_file.close()
        pickle_file_list.append(pickle_file_name)
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
        current_tmp = pickle.load(open(i))
        #print current_tmp.keys()
        #print current_tmp['sims'].keys()
        current_serials = current_tmp['sims'].keys()
        for jjj in current_serials:
            combined_answer['sims'][jjj]=copy.deepcopy(current_tmp['sims'][jjj])
    return combined_answer



#Chease Setup
def chease_PEST_setup(project_dict):
    for i in project_dict['sims'].keys():
        print i
        project_dict['sims'][i]['dir_dict']['chease_dir_PEST'] = project_dict['sims'][i]['dir_dict']['chease_dir'].rstrip('/') + '_PEST/'
        os.system('mkdir ' + project_dict['sims'][i]['dir_dict']['chease_dir_PEST'])

        #copy the Chease template
        copy_chease_files(project_dict['sims'][i], PEST=1)
        #modify the datain so it is relevant for this project
        modify_datain(project_dict['sims'][i],project_dict['details']['template_dir'], PEST=1)
        #generate a job file that can be submitted to the cluster
        generate_chease_job_file(project_dict['sims'][i], PEST=1)

