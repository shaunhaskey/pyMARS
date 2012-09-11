import os
from PythonMARS_funcs import modify_input_file

def corsica_run_setup(base_dir, input_data):
    running_dir = base_dir + input_data[0]
    os.system('mkdir ' + base_dir)
    os.system('mkdir ' + running_dir)
    os.chdir(running_dir)

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
    replacement_dcon = '  integer calldcon = %d'%(0)

    modify_input_file('sspqi_sh_this_dir.bas', ' pmin =',replacement_pmin)
    modify_input_file('sspqi_sh_this_dir.bas', ' qmin =',replacement_qmin)
    modify_input_file('sspqi_sh_this_dir.bas', ' pstep =',replacement_pstep)
    modify_input_file('sspqi_sh_this_dir.bas', ' qstep =',replacement_qstep)
    modify_input_file('sspqi_sh_this_dir.bas', ' npmult =',replacement_npmult)
    modify_input_file('sspqi_sh_this_dir.bas', ' calldcon =',replacement_dcon)

def corsica_run_setup2(base_dir, input_data, settings):
    running_dir = base_dir + input_data[0]
    os.system('mkdir ' + base_dir)
    os.system('mkdir ' + running_dir)
    os.chdir(running_dir)

    efit_base_dir = '/u/haskeysr/mars/eq_from_matt/efit_files/'
    os.system('cp ~/caltrans/sspqi_sh2.bas sspqi_sh_this_dir.bas')
    os.system('cp ' + efit_base_dir + '* .')

    command_file = open('commands_test.txt','w')
    command_string = 'read "sspqi_sh_this_dir.bas"\nquit\n'
    command_file.write(command_string)
    command_file.close()

    template_file = open('sspqi_sh_this_dir.bas','r')
    template_text = template_file.read()
    template_file.close()

    #make all the changes
    for i in settings.keys():
        print i, settings[i]
        template_text = template_text.replace(i,settings[i])

    template_file = open('sspqi_sh_this_dir.bas','w')
    template_file.write(template_text)
    template_file.close()



def run_corsica(base_dir, input_list, script_name):
    script = '#!/bin/bash\n'
    for input_data in input_list:#len(list)):
        running_dir = base_dir + input_data[0]
        script+='cd ' + running_dir + '\n'
        command = '/d/caltrans/vcaltrans/bin/caltrans -probname eq_vary_p_q < commands_test.txt > caltrans_out.log'
        script+= command + '\n'
    file_name = open(base_dir + script_name,'w')
    file_name.write(script)
    file_name.close()

    job_string = '#!/bin/bash\n'
    job_string = job_string + '#$ -N Caltrans\n'
    job_string = job_string + '#$ -q all.q\n'
    job_string = job_string + '#$ -o %s\n'%(script_name + 'sge_output.dat')
    job_string = job_string + '#$ -e %s\n'%(script_name + 'sge_error.dat')
    job_string = job_string + '#$ -cwd\n'
    job_string = job_string + 'echo $PATH\n'
    job_string = job_string + './'+ script_name+'\n'
    file_name = open(base_dir + script_name +'.job','w')
    file_name.write(job_string)
    file_name.close()
    os.system('chmod +x ' + base_dir +script_name)


def execute_scripts(base_dir, script_name):
    os.system('source ' + base_dir + script_name)


def corsica_batch_run(corsica_list, project_dict, corsica_base_dir):
    workers = 1
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
        run_corsica(corsica_base_dir, send_list, 'corsica_script'+str(i)+'.sh')
        script_file_list.append('corsica_script' + str(i)+'.sh')
    for i in script_file_list:
        print 'running ' + i + ' script'
        os.system('source ' + corsica_base_dir + i)
    for i in corsica_list:
        print 'copying files across'
        os.system('cp ' + corsica_base_dir + i[0] + '/EXPEQ* stab_setup* ' + project_dict['details']['efit_dir'])
    #os.system('rm -r ' +corsica_base_dir)
