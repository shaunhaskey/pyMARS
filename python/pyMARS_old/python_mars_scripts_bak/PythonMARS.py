#!/usr/bin/python
import sys
import os
import time
import string

shot_input = sys.argv[1]
print shot_input
#--- List of user defined variables ---------
base_dir = '/u/haskeysr/mars/tutorial/'
efit_master = '/u/haskeysr/mars/tutorial/shot135762/efit'
#shot = 15
shot = int(shot_input)
pressure = 1.0
shot_time = 1805
thetac = 0.005


corsica_run = 0
chease_run = 0
fxrun_run = 0
mars_vac_run = 0
mars_plasma_run = 0

def issue_command(command, direct):
    os.system(command + direct)


def replace_value(filename,variable_name,end_limiter,new_value):
    file = open(filename,'r')
    input_string = file.read()
    file.close()

    success = 0
    for i in [' ', '\n', end_limiter]:
       if string.find(input_string, i+variable_name) != -1:
           name_start_location = string.find(i+input_string, variable_name)
           success = success + 1
    if success == 0:
        print 'ERROR : variable not found'
    if success >= 2 :
        print 'ERROR : multiple instances?'
        print success
    limiter_location= string.find(input_string[name_start_location:],end_limiter)+name_start_location
    new_string = variable_name + ' = ' + str(new_value)
    total_new_string = input_string[:name_start_location-1] + new_string + input_string[limiter_location:]

    file = open(filename,'w')
    file.write(total_new_string)
    file.close()

def extract_value(filename,variable_name,end_limiter):
    file = open(filename,'r')
    input_string = file.read()
    file.close()
    success = 0
    if end_limiter == ' ':
        prefix_list = ['\n', end_limiter]
    else:
        prefix_list = [' ','\n', end_limiter]

    for i in prefix_list:
       if string.find(input_string, i+variable_name) != -1:
           name_start_location = string.find(i+input_string, variable_name)
           success = success + 1
    if success == 0:
        print 'ERROR : variable not found'
    if success >= 2 :
        print 'ERROR : multiple instances?'
        print success

    temp = 1
    equal_location = string.find(input_string[name_start_location:],'=')+name_start_location
    print 'equal character ' + input_string[equal_location]
    while input_string[equal_location + temp] == end_limiter:
        temp = temp + 1
    limiter_location= string.find(input_string[equal_location+temp:],end_limiter) + equal_location + temp
    old_string = input_string[name_start_location:limiter_location+1]
    formated_old_string=old_string.replace(' ','').replace(end_limiter, '').replace('\n','')
    old_value = formated_old_string[formated_old_string.find('=')+1:]
    return old_value # note this is returned as a string      


#--- Simple function dify an parameter in an input file ---
def modify_input_file(filename, search_text, replacement):
    file = open(filename,'r')
    text_orig = file.read()
    file.close()
    text_list = text_orig.splitlines()
    mod_txt = ''
    for i in range(0,text_orig.count('\n')):
        check = text_list[i].find(search_text)
        if check != -1:
            print filename + ' OLD : ' + text_list[i]
            text_list[i]= replacement
            print filename + ' NEW : ' + replacement
            mod_txt=mod_txt + replacement + '\n'
        else:
            mod_txt=mod_txt + text_list[i] + '\n'

    file = open(filename,'w')
    file.write(mod_txt)
    file.close()


#Create relevant directories
#Create shot
start = time.time()
print 'Creating all required Directories'
'''
def generate_dir_names(base_dir, shot, thetac, pressure):
    shot_dir =  base_dir + 'shot' + str(shot)
    thetac_dir = shot_dir + '/'+ 'tc_%03d'%(thetac*1000)
    efit_dir =  thetac_dir + '/' + 'efit'
    exp_dir = thetac_dir + '/exp%.1f'%pressure
    mars_dir =  exp_dir + '/marsrun'
    chease_dir = exp_dir + '/cheaserun'
    mars_vac_dir =  mars_dir + '/RUNrfa.vac'
    mars_plasma_dir =  mars_dir + '/RUNrfa.p'
'''
shot_dir =  base_dir + 'shot' + str(shot)
issue_command('mkdir ', shot_dir)

thetac_dir = shot_dir + '/'+ 'tc_%03d'%(thetac*1000)
issue_command('mkdir ', thetac_dir)

efit_dir =  thetac_dir + '/' + 'efit'
issue_command('mkdir ', efit_dir)

exp_dir = thetac_dir + '/exp%.1f'%pressure
issue_command('mkdir ', exp_dir)

mars_dir =  exp_dir + '/marsrun'
issue_command('mkdir ', mars_dir)

chease_dir = exp_dir + '/cheaserun'
issue_command('mkdir ', chease_dir)

mars_vac_dir =  mars_dir + '/RUNrfa.vac'
issue_command('mkdir ', mars_vac_dir)

mars_plasma_dir =  mars_dir + '/RUNrfa.p'
issue_command('mkdir ', mars_plasma_dir)

#----------- Copy EFIT files
issue_command('cp ', efit_master + '/* ' + efit_dir)

print 'Finished creating Directories'
directory_time = time.time() - start
#----------------- Corsica -------------------------#
#----------------- Corsica : Modify stab file -------
print 'Modifying corsica stab file for thetac value'
os.chdir(efit_dir)
os.system('cp '+base_dir+'stab_setup.bas .')
newline_thetac = '  real mythetac = %.3f         #thetac '%(thetac)


modify_input_file('stab_setup.bas', 'real mythetac', newline_thetac)

#------------ Corsica : Run --------------------

os.chdir(efit_dir)
corsica_run_command_file = open('corsica_run_command.txt','w')
corsica_run_command_file.write('read "' + efit_dir + '/stab_setup.bas"\nquit\n')
corsica_run_command_file.close()

if corsica_run ==1:
    issue_command('/d/caltrans/vcaltrans/bin/caltrans -probname eqdsk ', '< corsica_run_command.txt')
    print 'Running Corsica'
else:
    print 'Corsica run skipped'
    
#--------- Extract R0 and B0 from the stab_setup_results --------------
stab_results = open('stab_setup_results.dat')
stab_results.readline()
stab_results.readline()
variables = stab_results.readline()
variable_values = stab_results.readline()
R0_location = variables.rfind('R0')
R0 = float(variable_values[R0_location:R0_location+10])
B0_location = variables.rfind('B0')
B0 = float(variable_values[B0_location:B0_location+10])
stab_results.close()
print 'Finished Corsica section'
corsica_time = time.time() - directory_time - start

#---------- End Corsica ---------------
#---------- Chease : Copy files required--------------
print 'Start Chease section'
os.chdir(chease_dir)
issue_command('cp ', efit_dir + '/EXPEQ* ' + chease_dir)
issue_command('ln -s ', 'EXPEQ* EXPEQ')
issue_command('cp ', base_dir + '/datain ' + chease_dir)


#--------- Chease : Modify datain file (B0,R0) ------------

print 'Modifying corsica stab file for thetac value'
os.chdir(chease_dir)
os.system('cp '+base_dir+'datain .')
newline_B0 = '   B0EXP=%.8f, R0EXP=%.8f,'%(B0,R0)
newline_header = '***    D3D%d @ %d   For n=1 RFA'%(shot,shot_time)

modify_input_file('datain', 'B0EXP', newline_B0)
modify_input_file('datain', 'D3D', newline_header)


#---------Chease : Run ------------------------------

if chease_run ==1:
    os.system('/u/haskeysr/bin/runchease')
    print 'Run Chease'
else:
    print 'Chease run skipped'
print 'Finished Chease'


#---------Chease : Extract NW from log_chease--------------
os.chdir(chease_dir)
NW_value = extract_value('log_chease','NW',' ')
print 'NW : ' + NW_value
chease_time = time.time() - corsica_time - start

#-------- fxrun section ---------------
print 'creating fxin and runing fxrun'
os.chdir(chease_dir)
fxin = open('fxin','w')
fxin.write('59\n%.8f\n%.8f\n'%(R0,B0))
fxin.close()

if fxrun_run ==1:
    os.system("bash -i -c 'fxrun'") #need bash -i -c to force a read of .bashrc to understand alias
    print 'Finished fxrun section'
else:
    print 'Fxrun skipped'

fxrun_time = time.time() - chease_time - start

#--------- Mars Vacuum : setup -------------------
print 'Start Mars Vacuum Section'
issue_command('cp ', chease_dir + '/OUT*MAR ' + mars_dir)
issue_command('cp ', chease_dir + '/RMZM_F ' + mars_dir)
issue_command('cp ', chease_dir + '/log_chease ' + mars_dir)

os.chdir(mars_vac_dir)
os.system("for file in $(ls '../../../efit/' | grep PROF ) ; do ln -s '../../../efit/'$file ; done")
os.system("for file in $(ls '../../cheaserun/' | grep OUT|grep MAR ) ; do ln -s '../../cheaserun/'$file ; done")


print 'Modify Mars run file'
os.chdir(mars_vac_dir)
os.system('cp '+base_dir+'RUN .')
modify_input_file('RUN', 'INCFEED', ' INCFEED= 4,')


#****Need to include how the external coils are mapped from R,Z coordinates into MARS coordinates****
#os.system('cp ' + base_dir + 'MacMainD3D135762-1805.m .')
#os.system('matlab -nodesktop < >matlab_output_pipe.txt')



#---------- Mars Vacuum : Run --------------------
print 'Run Mars vacuum'
if mars_vac_run == 1:
    os.system('nice -10 /u/haskeysr/bin/runmarsf > log_runmars')
    print 'Finished Mars Vacuum Run'
else:
    print 'Mars Vacuum Run Skipped'

mars_vac_time = time.time() - fxrun_time - start

#--------- Mars Plasma : setup -------------------
print 'Mars Plasma section'
os.chdir(mars_plasma_dir)
os.system("for file in $(ls '../../../efit/' | grep PROF ) ; do ln -s '../../../efit/'$file ; done")
os.system("for file in $(ls '../../cheaserun/' | grep OUT|grep MAR ) ; do ln -s '../../cheaserun/'$file ; done")

print 'Modify Mars run file'
os.chdir(mars_plasma_dir)
os.system('cp '+base_dir+'RUN .')
newline_B0 = '   B0EXP=%.8f, R0EXP=%.8f,'%(B0,R0)
newline_INCFEED = ' INCFEED= 8,'
modify_input_file('RUN', 'INCFEED', newline_INCFEED)

#---------- Mars Plasma : Run --------------------
print 'Run Mars plasma'
if mars_plasma_run == 1:
    os.system('nice -10 /u/haskeysr/bin/runmarsf > log_runmars')
    print 'Finished Mars Plasma Run'
else:
    print 'Mars Plasma run skipped'
    
mars_plasma_time = time.time() - mars_vac_time - start

print 'Directory Time :%.2fs'%(directory_time)
print 'Corsica Time :%.2fs'%(corsica_time)
print 'Chease Time :%.2fs'%(chease_time)
print 'fxrun Time :%.2fs'%(fxrun_time)
print 'Mars Vac Time :%.2fmins'%(mars_vac_time/60)
print 'Mars Plasma Time :%.2fmins'%(mars_plasma_time/60)
print 'Total Time : %.2fmins'%((time.time()-start)/60)
