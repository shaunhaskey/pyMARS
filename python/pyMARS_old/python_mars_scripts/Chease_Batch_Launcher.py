#!/usr/bin/python
import os
import time
import subprocess as sub


# Run Corsica over a range of pressure and q values - Edit Matts Corsica batch to achieve this, any benefit to splitting it up into batch jobs?
# Need to create the directory structure for all of the above
# Run Chease over the same range of pressure and q values - need to make a note of what they were from above
# Run Mars over the same range of pressure and q values
# Run fxrun over all of the above (can't do this from venus)

def launch_job(job_name):
    os.system('qsub ' + job_name)

def launch_job_mars(job_name):
    os.system('qsub -l mem_free=15G,h_vmem=15G ' + job_name)

def running_jobs():
    p = sub.Popen(['qstat', '-u', 'haskeysr'],stdout=sub.PIPE,stderr=sub.PIPE)
    output, errors = p.communicate()
    number_of_jobs = output.count('\n') - 2
    if number_of_jobs<0:number_of_jobs = 0
    print 'Running Jobs : ' + str(number_of_jobs)
    return number_of_jobs

def batch_launch_chease(master_dict, setpoint):
    start_time = time.time()
    submitted_jobs = 0
    total_jobs = len(master_dict.keys())
    startup = 1
    startup_runs = 0
    for i in master_dict.keys():
        print i
        os.chdir(master_dict[i]['dir_dict']['chease_dir'])
        if startup==1:
            launch_job('chease.job')
            startup_runs += 1
            if startup_runs == setpoint:
                startup = 0
                time.sleep(30)
        else:
            while running_jobs()>setpoint:
                time.sleep(10)
            launch_job('chease.job')
        master_dict[i]['chease_run'] = 1
        submitted_jobs += 1
        print 'Submitted %d jobs of %d, time %.2fmins'%(submitted_jobs,total_jobs,(time.time()-start_time)/60)
    while running_jobs()>0:
        print 'Submitted %d of %d, Waiting for last %d jobs to finish, time so far : %.2fmins'%(submitted_jobs, total_jobs, running_jobs(), (time.time()-start_time)/60)
        time.sleep(15)
    return master_dict
            
def batch_launch_mars(master_dict,setpoint):
    start_time = time.time()
    submitted_jobs = 0
    total_jobs = len(master_dict.keys())*2
    startup = 1
    startup_runs = 0
    for i in master_dict.keys():
        print i
        if startup==1:
            os.chdir(master_dict[i]['dir_dict']['mars_vac_dir'])
            launch_job_mars('mars_venus.job')
            os.chdir(master_dict[i]['dir_dict']['mars_plasma_dir'])
            launch_job_mars('mars_venus.job')
            master_dict[i]['mars_vac_run'] = 1
            master_dict[i]['mars_p_run'] = 1
            startup_runs += 2
            submitted_jobs += 2
            print 'Submitted %d jobs of %d, time %.2fmins'%(submitted_jobs,total_jobs,(time.time()-start_time)/60)
            if startup_runs >= setpoint:
                startup = 0
                time.sleep(30)
        else:
            while running_jobs()>setpoint:
                time.sleep(10)
            os.chdir(master_dict[i]['dir_dict']['mars_vac_dir'])
            launch_job_mars('mars_venus.job')
            os.chdir(master_dict[i]['dir_dict']['mars_plasma_dir'])
            launch_job_mars('mars_venus.job')
            submitted_jobs += 2
            print 'Submitted %d jobs of %d, time %.2fmins'%(submitted_jobs,total_jobs,(time.time()-start_time)/60)
            master_dict[i]['mars_vac_run'] = 1
            master_dict[i]['mars_p_run'] = 1
    while running_jobs()>0:
        print 'Submitted %d of %d, Waiting for last %d jobs to finish, time so far : %.2fmins'%(submitted_jobs, total_jobs, running_jobs(), (time.time()-start_time)/60)
        time.sleep(15)
    return master_dict
