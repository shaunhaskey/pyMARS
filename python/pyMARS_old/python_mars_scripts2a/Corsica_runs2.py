import os
from PythonMARS_funcs import modify_input_file

def corsica_run_setup(input_data):
    os.system('mkdir /scratch/haskeysr/corsica_test6')
    base_dir = '/scratch/haskeysr/corsica_test6/' + input_data[0]
    os.system('mkdir ' + base_dir)
    os.chdir(base_dir)

    efit_base_dir = '/u/haskeysr/mars/eq_from_matt/efit_files/'
    os.system('cp ~/caltrans/sspqi_sh.bas sspqi_sh_this_dir.bas')
    os.system('cp ' + efit_base_dir + '* .')

    command_file = open('commands_test.txt','w')
    command_string = 'read "sspqi_sh_this_dir.bas"\nquit\n'
    command_file.write(command_string)
    command_file.close()

    pmin = input_data[1]
    qmin = input_data[2]
    pstep = input_data[4]
    qstep = input_data[5]
    npmult = input_data[3]
    factor = 3
    print 'min value of p : %.4f'%(pmin+npmult*pstep)
    replacement_pmin = '  real pmin = %.2f'%(pmin)
    replacement_qmin = '  real qmin = %.2f'%(qmin)
    replacement_pstep = '  real pstep = %.4f'%(pstep * factor)
    replacement_qstep = '  real qstep = %.4f'%(qstep * factor)
    replacement_npmult = '  integer npmult = %d'%(int(npmult/factor))
    replacement_dcon = '  integer calldcon = %d'%(1)

    modify_input_file('sspqi_sh_this_dir.bas', ' pmin =',replacement_pmin)
    modify_input_file('sspqi_sh_this_dir.bas', ' qmin =',replacement_qmin)
    modify_input_file('sspqi_sh_this_dir.bas', ' pstep =',replacement_pstep)
    modify_input_file('sspqi_sh_this_dir.bas', ' qstep =',replacement_qstep)
    modify_input_file('sspqi_sh_this_dir.bas', ' npmult =',replacement_npmult)
    modify_input_file('sspqi_sh_this_dir.bas', ' calldcon =',replacement_dcon)

def run_corsica(input_list, script_name):
    script = '#!/bin/bash\n'
    for input_data in input_list:#len(list)):
        print i
        base_dir = '/scratch/haskeysr/corsica_test6/' + input_data[0]
        script+='cd ' + base_dir + '\n'
        command = '/d/caltrans/vcaltrans/bin/caltrans -probname eq_vary_p_q < commands_test.txt > caltrans_out.log'
        script+= command + '\n'
    file = open('/scratch/haskeysr/corsica_test6/' +script_name,'w')
    file.write(script)
    file.close()

    job_string = '#!/bin/bash\n'
    job_string = job_string + '#$ -N Caltrans\n'
    job_string = job_string + '#$ -q all.q\n'
    job_string = job_string + '#$ -o %s\n'%(script_name + 'sge_output.dat')
    job_string = job_string + '#$ -e %s\n'%(script_name + 'sge_error.dat')
    job_string = job_string + '#$ -cwd\n'
#    job_string = job_string + '#$ -M shaunhaskey@gmail.com\n'
#    job_string = job_string + '#$ -m e\n'
    job_string = job_string + 'echo $PATH\n'
    job_string = job_string + './'+ script_name+'\n'
    file = open('/scratch/haskeysr/corsica_test6/' + script_name+'.job','w')
    file.write(job_string)
    file.close()

    os.system('chmod +x /scratch/haskeysr/corsica_test6/'+script_name)



if __name__ == '__main__':
    list = [['ml_new_10', 0.08, 0.00, 12, -0.005, 0.03],
    ['ml_new_11', 0.11, 0.00, 18, -0.005, 0.03],
    ['ml_new_12', 0.14, 0.00, 24, -0.005, 0.03],
    ['ml_new_13', 0.17, 0.00, 30, -0.005, 0.03],
    ['ml_new_14', 0.20, 0.00, 36, -0.005, 0.03],
    ['ml_new_15', 0.23, 0.00, 42, -0.005, 0.03],
    ['ml_new_16', 0.26, 0.00, 48, -0.005, 0.03],
    ['ml_new_17', 0.29, 0.00, 54, -0.005, 0.03],
    ['ml_new_18', 0.32, 0.00, 60, -0.005, 0.03],
    ['ml_new_19', 0.35, 0.00, 66, -0.005, 0.03],
    ['ml_new_20', 0.38, 0.00, 72, -0.005, 0.03],
    ['ml_new_21', 0.41, 0.00, 78, -0.005, 0.03],
    ['ml_new_22', 0.44, 0.00, 84, -0.005, 0.03],
    ['ml_new_23', 0.47, 0.00, 90, -0.005, 0.03],
    ['ml_new_24', 0.50, 0.00, 90, -0.005, 0.03],
    ['ml_new_25', 0.53, 0.00, 90, -0.005, 0.03],
    ['ml_new_26', 0.56, 0.00, 90, -0.005, 0.03],
    ['ml_new_27', 0.60, 0.00, 90, -0.005, 0.03],
    ['ml_new_28', 0.63, 0.00, 90, -0.005, 0.03],
    ['ml_new_29', 0.66, 0.00, 90, -0.005, 0.03],
    ['ml_new_30', 0.70, 0.00, 90, -0.005, 0.03],
    ['ml_new_31', 0.73, 0.00, 90, -0.005, 0.03]]



    print 'start setup'
    for i in list:
        corsica_run_setup(i)
    print 'finished setup'
    workers = 4
    worker_list = []
    proportion = int(round(len(list)/workers))
    print proportion
    for i in range(0,workers):
        if i == (workers-1):
            send_list = list[proportion*i:len(list)]
        else:
            send_list = list[proportion*i:proportion*(i+1)]
        print i
        print send_list
        run_corsica(send_list, 'corsica_script'+str(i)+'.sh')
