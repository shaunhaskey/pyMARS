import os
from PythonMARS_funcs import modify_input_file
from multiprocessing import Process
from multiprocessing import Pool

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
    list = [['ml10', 0.50, 0.00, 30, -0.005, 0.03],
    ['ml11', 0.53, 0.00, 30, -0.005, 0.03],
    ['ml12', 0.56, 0.00, 30, -0.005, 0.03],
    ['ml13', 0.60, 0.00, 74, -0.005, 0.03],
    ['ml14', 0.63, 0.00, 77, -0.005, 0.03],
    ['ml15', 0.66, 0.00, 80, -0.005, 0.03],
    ['ml16', 0.70, 0.00, 86, -0.005, 0.03],
    ['ml17', 0.73, 0.00, 90, -0.005, 0.03],
    ['ml18', 0.76, 0.00, 90, -0.005, 0.03],
    ['ml19', 0.80, 0.00, 90, -0.005, 0.03],
    ['ml20', 0.83, 0.00, 90, -0.005, 0.03],
    ['ml21', 0.86, 0.00, 90, -0.005, 0.03],
    ['ml22', 0.90, 0.00, 90, -0.005, 0.03],
    ['ml23', 0.93, 0.00, 90, -0.005, 0.03],
    ['ml24', 0.96, 0.00, 90, -0.005, 0.03],
    ['ml25', 1.00, 0.00, 90, -0.005, 0.03],
    ['ml26', 1.03, 0.00, 90, -0.005, 0.03],
    ['ml27', 1.06, 0.00, 90, -0.005, 0.03],
    ['ml28', 1.10, 0.00, 90, -0.005, 0.03],
    ['ml29', 1.13, 0.00, 90, -0.005, 0.03],
    ['ml30', 1.16, 0.00, 90, -0.005, 0.03],
    ['ml31', 1.20, 0.00, 90, -0.005, 0.03],
    ['ml32', 1.23, 0.00, 90, -0.005, 0.03],
    ['ml33', 1.26, 0.00, 90, -0.005, 0.03],
    ['ml34', 1.30, 0.00, 90, -0.005, 0.03],
    ['ml35', 1.33, 0.00, 90, -0.005, 0.03],
    ['ml36', 1.36, 0.00, 90, -0.005, 0.03],
    ['ml37', 1.40, 0.00, 90, -0.005, 0.03],
    ['ml38', 1.43, 0.00, 90, -0.005, 0.03],
    ['ml39', 1.46, 0.00, 90, -0.005, 0.03],
    ['ml40', 1.50, 0.00, 90, -0.005, 0.03],
    ['ml41', 1.53, 0.00, 90, -0.005, 0.03],
    ['ml42', 1.56, 0.00, 90, -0.005, 0.03],
    ['ml43', 1.60, 0.00, 45, -0.005, 0.03],
    ['ml44', 1.63, 0.00, 45, -0.005, 0.03],
    ['ml45', 1.66, 0.00, 45, -0.005, 0.03],
    ['ml46', 1.70, 0.00, 45, -0.005, 0.03],
    ['ml47', 1.73, 0.00, 45, -0.005, 0.03],
    ['ml48', 1.76, 0.00, 30, -0.005, 0.03],
    ['ml49', 1.80, 0.00, 30, -0.005, 0.03],
    ['ml50', 1.83, 0.00, 30, -0.005, 0.03],
    ['ml51', 1.86, 0.00, 30, -0.005, 0.03]]

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
