#Contains functions that are used to launch batch jobs onto Venus for MARS and CHEASE runs

import os, time
import subprocess as sub


def launch_job(job_name):
    os.system('qsub ' + job_name)

def launch_job_mars(job_name):
    os.system('qsub -l mem_free=15G,h_vmem=15G ' + job_name)

def check_job_number_file(file_name):
    file_name = open(file_name,'r')
    simul_jobs_content = file_name.read()
    file_name.close()
    stab_lines = simul_jobs_content.split('\n')
    simul_jobs = int(stab_lines[0].rstrip('\n'))
    return simul_jobs

def running_jobs(id_string):
    #p = sub.Popen(['qstat', '-u', username],stdout=sub.PIPE,stderr=sub.PIPE)
    p = sub.Popen(['qstat'],stdout=sub.PIPE,stderr=sub.PIPE)
    output, errors = p.communicate()
    number_of_jobs = output.count(id_string)
    #number_of_jobs = output.count(count_string) - 2
    #if number_of_jobs<0:number_of_jobs = 0
    print 'Running Jobs : ' + str(number_of_jobs)
    return number_of_jobs

def batch_launch_chease(master_dict, job_num_filename, PEST=0, id_string = 'Chease_'):
    start_time = time.time()
    submitted_jobs = 0; startup = 1; startup_runs = 0

    total_jobs = len(master_dict.keys())

    for i in master_dict.keys():
        setpoint = check_job_number_file(job_num_filename)
        print i
        if PEST==1:
            os.chdir(master_dict[i]['dir_dict']['chease_dir_PEST'])
        else:
            os.chdir(master_dict[i]['dir_dict']['chease_dir'])
        if startup==1:
            launch_job('chease.job')
            startup_runs += 1
            if startup_runs == setpoint:
                startup = 0
                time.sleep(30)
        else:
            while running_jobs(id_string)>setpoint:
                time.sleep(10)
            launch_job('chease.job')
        master_dict[i]['chease_run'] = 1
        submitted_jobs += 1
        print 'Submitted %d jobs of %d, time %.2fmins'%(submitted_jobs,total_jobs,(time.time()-start_time)/60)
    while running_jobs(id_string)>0:
        print 'Submitted %d of %d, Waiting for last %d jobs to finish, time so far : %.2fmins'%(submitted_jobs, total_jobs, running_jobs(id_string), (time.time()-start_time)/60)
        time.sleep(15)
    return master_dict
            
def batch_launch_mars(master_dict, job_num_filename, id_string = 'MARS'):
    start_time = time.time()

    submitted_jobs = 0;startup = 1;startup_runs = 0
    total_jobs = len(master_dict.keys())*2

    for i in master_dict.keys():
        setpoint = check_job_number_file(job_num_filename)
        print 'setpoint :', setpoint
        print i
        if startup==1:
            os.chdir(master_dict[i]['dir_dict']['mars_vac_dir'])
            launch_job_mars('mars_venus.job')
            time.sleep(10)
            os.chdir(master_dict[i]['dir_dict']['mars_plasma_dir'])
            launch_job_mars('mars_venus.job')
            time.sleep(10)
            master_dict[i]['mars_vac_run'] = 1
            master_dict[i]['mars_p_run'] = 1
            startup_runs += 2
            submitted_jobs += 2
            print 'Submitted %d jobs of %d, time %.2fmins'%(submitted_jobs,total_jobs,(time.time()-start_time)/60)
            time.sleep(5)
            if startup_runs >= setpoint:
                startup = 0
                time.sleep(30)
        else:
            while running_jobs(id_string)>setpoint:
                time.sleep(10)
            os.chdir(master_dict[i]['dir_dict']['mars_vac_dir'])
            launch_job_mars('mars_venus.job')
            time.sleep(10)
            os.chdir(master_dict[i]['dir_dict']['mars_plasma_dir'])
            launch_job_mars('mars_venus.job')
            time.sleep(10)
            submitted_jobs += 2
            print 'Submitted %d jobs of %d, time %.2fmins'%(submitted_jobs,total_jobs,(time.time()-start_time)/60)
            master_dict[i]['mars_vac_run'] = 1
            master_dict[i]['mars_p_run'] = 1
    while running_jobs(id_string)>0:
        print 'Submitted %d of %d, Waiting for last %d jobs to finish, time so far : %.2fmins'%(submitted_jobs, total_jobs, running_jobs(id_string), (time.time()-start_time)/60)
        time.sleep(15)
    return master_dict
