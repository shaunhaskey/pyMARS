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
    os.system('qsub -l mem_free=30G,h_vmem=31G ' + job_name)


job_list = ['test.job','test1.job','test2.job','test3.job','test4.job']#,'test5.job','test6.job']

set_point = 3
start_up = 1
number_of_jobs = 1
start_up_started = 0
while (len(job_list)>=1) or (start_up==1):
    if start_up ==1:
        print 'Launching : ' + job_list[0]
        launch_job(job_list[0])
        job_list.remove(job_list[0])
        start_up_started = start_up_started + 1
        if start_up_started == set_point:
            print 'startup finished - reached setpoint'
            start_up = 0
            time.sleep(60)
    else:
        time.sleep(10)
        p = sub.Popen(['qstat', '-u', 'haskeysr'],stdout=sub.PIPE,stderr=sub.PIPE)
        output, errors = p.communicate()
        print output
        number_of_jobs = output.count('\n') - 2
        print 'Number of jobs : ' + str(number_of_jobs)
        if (number_of_jobs<set_point) and (len(job_list)>=1):
            print 'Launching : ' + job_list[0]
            launch_job(job_list[0])
            job_list.remove(job_list[0])

print 'finished launching all jobs - waiting for them to finish'

while number_of_jobs>1:
    p = sub.Popen(['qstat', '-u', 'haskeysr'],stdout=sub.PIPE,stderr=sub.PIPE)
    output, errors = p.communicate()
    print output
    number_of_jobs = output.count('\n') - 2
    print 'Number of jobs : ' + str(number_of_jobs)
    time.sleep(20)

print 'finished'
