import numpy as num
import time, os, sys, string, re, csv, pickle
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata
#function to read in a stab_results file and generate a dictionary containing
#information about the equilibria
def read_stab_results(file_location):
    stab_setup_contents = open(file_location).read()
    stab_lines = stab_setup_contents.split('\n')
    line = 0

    dictionary_list = {}
    
    #skip lines at the start with ;
    while stab_lines[line][0] == ';':
        line += 1

    var_names = []

    #extract variable names
    stab_lines[line] = stab_lines[line].lstrip(' ').rstrip('\n').rstrip(' ')
    stab_lines[line] = stab_lines[line]+ ' '

    while len(stab_lines[line]) >= 1:
        end = stab_lines[line].find(' ')
        var_names.append(stab_lines[line][0:end])
        stab_lines[line] = stab_lines[line].lstrip(var_names[-1]).lstrip(' ')

    #extract equilibrium run values
    line += 1
    values = []
    #item = 1
    success = 0
    fails = 0
    item = 1
    while (line< len(stab_lines)) and (len(stab_lines[line])>1):
        try:
            current_value = []
            stab_lines[line] = stab_lines[line].lstrip(' ').rstrip('\n').rstrip(' ')
            stab_lines[line] = stab_lines[line] + ' ' #pad so while loop works
            while len(stab_lines[line]) >= 1:
                end = stab_lines[line].find(' ')
                current_value.append(stab_lines[line][0:end])
                stab_lines[line] = stab_lines[line].lstrip(current_value[-1]).lstrip(' ')
            #index each eq by a tuple (pmult,qmult)
            #item = (current_value[0],current_value[1])

            dictionary_list[item]={}
            for i in range(0,len(current_value)):
                dictionary_list[item][var_names[i]]=float(current_value[i])

            values.append(current_value)
            item += 1
            success += 1
        except:
            fails += 1
            print '!!!!! error reading values on line %d likely due to large -ve DCON value' %(line)
        line += 1

    print 'Successful read in : %d, failed read in : %d'%(success, fails)
    time.sleep(5)
    return dictionary_list

#Issue an os command - probably useless
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

def extract_value(filename, variable_name, end_limiter, strip_spaces = 1):
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
    if strip_spaces ==1:
        formated_old_string=old_string.replace(' ','').replace(end_limiter, '').replace('\n','')
    else:
        formated_old_string=old_string
        
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

def generate_directories(master, base_dir):
    dir_dict = {}    
    dir_dict['shot_dir'] =  base_dir + 'shot' + str(master['shot']) +'/'
    os.system('mkdir ' + dir_dict['shot_dir'])
    dir_dict['thetac_dir'] = dir_dict['shot_dir'] + 'tc_%03d/'%(master['thetac']*1000)
    os.system('mkdir ' + dir_dict['thetac_dir'])
    dir_dict['efit_dir'] =  dir_dict['thetac_dir'] + 'efit/'
    os.system('mkdir ' + dir_dict['efit_dir'])
    dir_dict['q_dir'] = dir_dict['thetac_dir'] + 'qmult%.3f/'%master['QMULT']
    os.system('mkdir ' + dir_dict['q_dir'])
    dir_dict['exp_dir'] = dir_dict['q_dir'] + 'exp%.3f/'%master['PMULT']
    os.system('mkdir ' + dir_dict['exp_dir'])
    dir_dict['mars_dir'] =  dir_dict['exp_dir'] + 'marsrun/'
    os.system('mkdir ' + dir_dict['mars_dir'])
    dir_dict['chease_dir'] = dir_dict['exp_dir'] + 'cheaserun/'
    os.system('mkdir ' + dir_dict['chease_dir'])
    dir_dict['mars_vac_dir'] =  dir_dict['mars_dir'] + 'RUNrfa.vac/'
    os.system('mkdir ' + dir_dict['mars_vac_dir'])
    dir_dict['mars_plasma_dir'] =  dir_dict['mars_dir'] + 'RUNrfa.p/'
    os.system('mkdir ' + dir_dict['mars_plasma_dir'])
    master['dir_dict']=dir_dict
    return master



def mars_directories(master, vac_name, plasma_name):
    dir_dict = {}
    master['dir_dict']['mars_vac_dir'] =  master['dir_dict']['mars_dir'] + vac_name + '/'
    os.system('mkdir ' + master['dir_dict']['mars_vac_dir'])
    master['dir_dict']['mars_plasma_dir'] =  master['dir_dict']['mars_dir'] + plasma_name + '/'
    os.system('mkdir ' + master['dir_dict']['mars_plasma_dir'])
    return master

#----------------- Corsica -------------------------#
#----------------- Corsica : Modify stab file -------
def modify_bas_file(master, template_dir):
    print 'Modifying corsica stab file for thetac value'
    os.chdir(master['dir_dict']['efit_dir'])
    os.system('cp '+template_dir+'sspqi_master.bas sspqi_current.bas')
    newline_thetac = '  real mythetac = %.3f         #thetac '%(master['thetac'])
    modify_input_file('stab_setup.bas', 'real mythetac', newline_thetac)
    #nqmult, npmult, pstemp, pmi, qmin, qstep, calldcon, myfuzzy, mysmjpar, q0gt1


#------------ Corsica : Run --------------------
def run_corsica(master):
    os.chdir(master['dir_dict']['efit_dir'])
    corsica_run_command_file = open('corsica_run_command.txt','w')
    corsica_run_command_file.write('read "' + master['dir_dict']['efit_dir'] + 'stab_setup.bas"\nquit\n')
    corsica_run_command_file.close()
    print 'Start Corsica running'
    issue_command('/d/caltrans/vcaltrans/bin/caltrans -probname eqdsk ', '< corsica_run_command.txt')
    print 'Finished Corsica'

#---------- End Corsica ---------------
#---------- Chease : Copy files required--------------
def copy_chease_files(master):
    os.chdir(master['dir_dict']['chease_dir'])
    issue_command('cp ', master['dir_dict']['efit_dir'] + master['EXPEQ_name'] + ' ' + master['dir_dict']['chease_dir'])
    issue_command('ln -s ', master['EXPEQ_name'] + ' EXPEQ')

#--------- Chease : Modify datain file (B0,R0) ------------
def modify_datain(master, template_dir):
    print 'Modifying corsica stab file for thetac value'
    os.chdir(master['dir_dict']['chease_dir'])
    os.system('cp '+ template_dir+'datain .')
    #newline_B0 = '   B0EXP=%.8f, R0EXP=%.8f,'%(master['B0EXP'],master['R0EXP'])
    newline_header = '***    D3D%d @ %d   For n=1 RFA'%(master['shot'],master['shot_time'])
    #modify_input_file('datain', 'B0EXP', newline_B0)
    modify_input_file('datain', 'D3D', newline_header)
    replace_value('datain','QSPEC', ',',str(master['QMAX']))
    replace_value('datain','B0EXP', ',',master['B0EXP'])
    replace_value('datain','R0EXP', ',',master['R0EXP'])
    replace_value('datain','NTOR', ',', master['NTOR'])
    
#---------Chease : Run ------------------------------
def execute_chease(master):
    os.chdir(master['dir_dict']['chease_dir'])
    print 'Start Chease executable'
    os.system('/u/haskeysr/bin/runchease')
    #os.system('bash -i -c /u/lanctot/chease/CheaseMerge20080822/bin/LinuxPortland64/chease.x < datain > log_chease')
    print 'Finished Chease Executable'

#---------Chease : Extract NW from log_chease--------------
def extract_NW(master):
    os.chdir(master['dir_dict']['chease_dir'])
    master['NW'] = int(round(float(extract_value('log_chease','NW',' '))))
    print 'NW : ' + str(master['NW'])
    return master

#-------- fxrun section ---------------
def fxin_create(master):
    print 'creating fxin and runing fxrun'
    os.chdir(master['dir_dict']['chease_dir'])
    fxin = open('fxin','w')
    fxin.write('%d\n%.8f\n%.8f\n'%(abs(master['M1'])+abs(master['M2'])+1,master['R0EXP'],master['B0EXP']))
    fxin.close()

def fxrun(master):
    os.chdir(master['dir_dict']['chease_dir'])
    os.system("bash -i -c 'fxrun'") #need bash -i -c to force a read of .bashrc to understand alias
    print 'Finished fxrun section'

#------------RMZM Matlab section - setup and run ---------------------
def modify_RMZM_F2(master):
    os.chdir(master['dir_dict']['chease_dir'])
    os.system('cp ~/matlab/RZplot3/MacMainD3D_Master.m MacMainD3D_current.m')
    SDIR_newline = "SDIR='"+ master['dir_dict']['chease_dir']+"';"

    #modify Matlab file
    modify_input_file('MacMainD3D_current.m', 'SDIR=', SDIR_newline)
    replace_value('MacMainD3D_current.m','Mac.resetCoil', ';', str(1))
    replace_value('MacMainD3D_current.m','Mac.Nm2', ';', str(1+int(abs(master['M1'])+abs(master['M2']))))
    return master





def RMZM_post_matlab(master):
    os.chdir(master['dir_dict']['chease_dir'])
    def extract_RMZM_output(file_name,variable):
        value = extract_value(file_name, variable, '\n', strip_spaces = 0)
        value = value.lstrip(' ').rstrip('\n')
        value1 = value[0:value.find(' ')]
        value2 = value[value.find(' '):].lstrip(' ').rstrip(' ')
        return [value1,value2]
    output = extract_RMZM_output('testing_output','FCCHI')
    FCCHI1 = float(output[0]);FCCHI2=float(output[1])
    print 'FCCHI :' + '|' +str(FCCHI1) + ',' +str(FCCHI2)

    output = extract_RMZM_output('testing_output','FWCHI')
    FWCHI1 = float(output[0]);FWCHI2=float(output[1])
    print 'FWCHI :' +'|' +str(FWCHI1) + ',' +str(FWCHI2)

    output = extract_RMZM_output('testing_output','IFEED')
    IFEED1 = int(output[0]); IFEED2 = int(output[1])
    print 'IFEED :' + str(IFEED1) + ',' +str(IFEED2)
    
    master['FCCHI'] = [FCCHI1,FCCHI2]
    master['FWCHI'] = [FWCHI1,FWCHI2]
    master['IFEED'] = [IFEED1,IFEED2]
    return master

#--------- Mars Vacuum : setup -------------------
def mars_setup_files(master, vac):
    if vac==1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
    os.system('ln -s ../../../../efit/PROFDEN PROFDEN')
    os.system('ln -s ../../../../efit/PROFROT PROFROT')
    os.system('ln -s ../../cheaserun/OUTRMAR OUTRMAR')
    os.system('ln -s ../../cheaserun/OUTVMAR OUTVMAR')
    os.system('ln -s ../../cheaserun/RMZM_F RMZM_F')
    os.system('ln -s ../../cheaserun/log_chease log_chease')

    #os.system('ln -s ' + master['dir_dict']['efit_dir'] + 'PROFDEN PROFDEN')
    #os.system('ln -s ' + master['dir_dict']['efit_dir'] + 'PROFROT PROFROT')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'OUTRMAR OUTRMAR')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'OUTVMAR OUTVMAR')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'RMZM_F RMZM_F')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'log_chease log_chease')

def mars_setup_files2(dir):
    os.chdir(dir)
    os.system('ln -s ../../../../efit/PROFDEN PROFDEN')
    os.system('ln -s ../../../../efit/PROFROT PROFROT')
    os.system('ln -s ../../cheaserun/OUTRMAR OUTRMAR')
    os.system('ln -s ../../cheaserun/OUTVMAR OUTVMAR')
    os.system('ln -s ../../cheaserun/RMZM_F RMZM_F')
    os.system('ln -s ../../cheaserun/log_chease log_chease')

    #os.system('ln -s ' + master['dir_dict']['efit_dir'] + 'PROFDEN PROFDEN')
    #os.system('ln -s ' + master['dir_dict']['efit_dir'] + 'PROFROT PROFROT')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'OUTRMAR OUTRMAR')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'OUTVMAR OUTVMAR')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'RMZM_F RMZM_F')
    #os.system('ln -s ' + master['dir_dict']['chease_dir'] + 'log_chease log_chease')

def mars_setup_alfven(master, input_frequency, vac):
    mu0 = 4e-7 * num.pi
    mi = 1.6726e-27 + 1.6749e-27
    e = 1.60217e-19
    if vac ==1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
    PROFDEN_file = open('PROFDEN','r')
    PROFDEN_data = PROFDEN_file.readlines()
    print PROFDEN_data[1]

    pattern = ''
    PROFDEN_data[1]
    re.search(pattern, PROFDEN_data[1])
    pattern = '\d+.\d+e*\+*\-*\d*'
    string1 = re.search(pattern, PROFDEN_data[1])
    string2 = re.search(pattern, PROFDEN_data[1][string1.end()+1:])
    ne0_r = float(PROFDEN_data[1][string1.start():string1.end()])
    ne0 = float(PROFDEN_data[1][string2.start()+string1.end()+1:string2.end()+string1.end()+1])


    #rotation data
    print 'Rotation section ------------' 
    PROFROT_file = open('PROFROT','r')
    PROFROT_data = PROFROT_file.readlines()
    print PROFROT_data[1]

    pattern = ''
    re.search(pattern, PROFROT_data[1])
    pattern = '\d+.\d+e*\+*\-*\d*'
    string1 = re.search(pattern, PROFROT_data[1])
    string2 = re.search(pattern, PROFROT_data[1][string1.end()+1:])
    vtor0_r = float(PROFROT_data[1][string1.start():string1.end()])
    vtor0 = float(PROFROT_data[1][string2.start()+string1.end()+1:string2.end()+string1.end()+1])
    print 'vtor0_r', vtor0_r, 'vtor0', vtor0


    
    B0EXP = master['B0EXP']
    R0EXP = master['R0EXP']

    v0a     = B0EXP/num.sqrt(mu0*mi*ne0)
    taua    = R0EXP/v0a
    tauwp   = 0.01405       # mu0*h*d/eta from /u/reimerde/mars/shot127838/README
    tauwm   = tauwp /taua

    f_v0a = v0a/R0EXP       # alfven frequency (1/s)
    fcio  = (e*R0EXP*num.sqrt(mu0*ne0))/num.sqrt(mi)

    ichz = num.array([1,2,5,10,20,40,60,100,120,160,200,500,1000,5000,10000.],dtype=float)
    ichz = num.array([input_frequency],dtype = float)

    vtorn=vtor0/v0a

#    nichz = N_ELEMENTS(ichz)
    omega = ichz/f_v0a
                        
    print 'From /u/lanctot/mars/utils/MARSplot/write_mars_params.pro'
    print '======================================='
    print 'Input values'
    print '======================================='
    print 'B0EXP (T)                  : ',B0EXP
    print 'R0EXP (m)                  : ',R0EXP
    print 'Central density (m^-3)    : ',ne0
    print 'Tau wall physics (s)      : ',tauwp
    print '======================================='
    print 'Normalized values'
    print '======================================='
    print 'Central Alfven speed (m/s)    : ',v0a
    print 'Central Alfven time (s)       :',taua
    print 'Central Alfven Frequency (1/s):',1.0/taua
    print 'TAUW              : ',tauwm
    print 'ROTE              : ',vtorn
    print 'Thermal OMEGACIO : ',fcio
#    IF nprofda GT 0 THEN  print 'Fast OMEGACIO    : ',fcio_fast
    print '======================================='
    print 'Coil Frequencies'
    print '======================================='
    print 'I-coil (Hz)      OMEGA'
    for k in range(0, len(ichz)): print '%.4e       %.4e'%(ichz[k], omega[k])
    print '======================================='
                                                                                                   
    print 'B0EXP: %.8f'%(B0EXP)
    print 'R0EXP: %.8f'%(R0EXP)
    print 'tauwp: %.8f'%(tauwp)
    print 'tauwm: %.8f'%(tauwm)
    print 'f_v0a: %.8f'%(f_v0a)
    print 'fcio: %.8f'%(fcio)
    master['ROTE'] = vtorn
    master['OMEGA_NORM'] = omega[0]
    master['TAUWM'] = tauwm
    master['v0a'] = v0a
    return master


def mars_setup_alfven2(dir, B0EXP, R0EXP, master, input_frequency):
    mu0 = 4e-7 * num.pi
    mi = 1.6726e-27 + 1.6749e-27
    e = 1.60217e-19
    os.chdir(dir)

    PROFDEN_file = open('PROFDEN','r')
    PROFDEN_data = PROFDEN_file.readlines()
    print PROFDEN_data[1]

    pattern = ''
    PROFDEN_data[1]
    re.search(pattern, PROFDEN_data[1])
    pattern = '\d+.\d+e*\+*\-*\d*'
    string1 = re.search(pattern, PROFDEN_data[1])
    string2 = re.search(pattern, PROFDEN_data[1][string1.end()+1:])
    ne0_r = float(PROFDEN_data[1][string1.start():string1.end()])
    ne0 = float(PROFDEN_data[1][string2.start()+string1.end()+1:string2.end()+string1.end()+1])

    v0a     = B0EXP/num.sqrt(mu0*mi*ne0)
    taua    = R0EXP/v0a
    tauwp   = 0.01405       # mu0*h*d/eta from /u/reimerde/mars/shot127838/README
    tauwm   = tauwp /taua

    f_v0a = v0a/R0EXP       # alfven frequency (1/s)
    fcio  = (e*R0EXP*num.sqrt(mu0*ne0))/num.sqrt(mi)

    ichz = num.array([1,2,5,10,20,40,60,100,120,160,200,500,1000,5000,10000.],dtype=float)
    ichz = num.array([input_frequency],dtype = float)

#    nichz = N_ELEMENTS(ichz)
    omega = ichz/f_v0a
                        
    print 'From /u4/lanctot/mars/utils/MARSplot/write_mars_params.pro'
    print '======================================='
    print 'Input values'
    print '======================================='
    print 'B0EXP (T)                  : ',B0EXP
    print 'R0EXP (m)                  : ',R0EXP
    print 'Central density (m^-3)    : ',ne0
    print 'Tau wall physics (s)      : ',tauwp
    print '======================================='
    print 'Normalized values'
    print '======================================='
    print 'Central Alfven speed (m/s)    : ',v0a
    print 'Central Alfven time (s)       :',taua
    print 'Central Alfven Frequency (1/s):',1.0/taua
    print 'TAUW              : ',tauwm
#    print 'ROTE              : ',vtorn[0]
    print 'Thermal OMEGACIO : ',fcio
#    IF nprofda GT 0 THEN  print 'Fast OMEGACIO    : ',fcio_fast
    print '======================================='
    print 'Coil Frequencies'
    print '======================================='
    print 'I-coil (Hz)      OMEGA'
    for k in range(0, len(ichz)): print '%.4e       %.4e'%(ichz[k], omega[k])
    print '======================================='
                                                                                                   
    print 'B0EXP: %.8f'%(B0EXP)
    print 'R0EXP: %.8f'%(R0EXP)
    print 'tauwp: %.8f'%(tauwp)
    print 'tauwm: %.8f'%(tauwm)
    print 'f_v0a: %.8f'%(f_v0a)
    print 'fcio: %.8f'%(fcio)

    master['OMEGA_NORM'] = omega[0]
    master['TAUWM'] = tauwm
    master['v0a'] = v0a
    return omega[0], tauwm, v0a


#--------------------
def mars_setup_run_file(master, template_dir, vac):
    print 'Modify Mars run file'
    if vac == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
        INCFEED = 4
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
        INCFEED = 8
    os.system('cp '+template_dir+'RUN .')
    replace_value('RUN','INCFEED', ',', str(INCFEED))
    replace_value('RUN','RNTOR', ',', '-'+str(master['NTOR']))
    FCCHI_string = str(master['FCCHI'][0]) + ', ' + str(master['FCCHI'][1]) + ','
    FWCHI_string = str(master['FWCHI'][0]) + ', ' + str(master['FWCHI'][1]) + ','
    AL0_string = '( 0, ' + str(master['OMEGA_NORM']) + '),'
                     
    replace_value('RUN','FCCHI','\n',FCCHI_string)
    replace_value('RUN','FWCHI','\n',FWCHI_string)
    replace_value('RUN','AL0','\n',AL0_string)
    replace_value('RUN','FEEDI','\n',master['FEEDI'])

    replace_value('RUN','M1',',',str(master['M1']))
    replace_value('RUN','M2',',',str(master['M2']))
    replace_value('RUN','TAUW',',',str(master['TAUWM']))
    replace_value('RUN','IWALL',',',str(master['NW']))

    if master['IFEED'][0]>= master['NW']:
        print "Possible error IFEED value is greater than or equal to NW :" + str(master['NW']) + ', IFEED' + str(master['IFEED'][0])
        replace_value('RUN','IFEED', ',', str(master['NW']-1))
        replace_value('RUN','ISENS', ',', str(master['NW']-1)) #because MARS cares about this!!! for some reason
    else:
        replace_value('RUN','IFEED', ',', str(master['IFEED'][0]))
        replace_value('RUN','ISENS', ',', str(master['IFEED'][0]))


def mars_setup_run_file_special(master, name, val, vac):
    print 'Modify Mars run file'
    if vac == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
        INCFEED = 4
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
        INCFEED = 8
                     
    replace_value('RUN',name,',',val)



#----------- Generate Job file for MARS batch run -----------
def generate_job_file(master,vac_true):
    if vac_true == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
        type_text = 'vac'
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
        type_text = 'p'
    job_string = '#!/bin/bash\n'
    job_string = job_string + '#$ -N Mars_p%d_q%d\n'%(int(round(master['PMULT']*100)),int(round(master['QMULT']*100)))
    job_string = job_string + '#$ -q all.q\n'
    job_string = job_string + '#$ -o %s\n'%('sge_output.dat')
    job_string = job_string + '#$ -e %s\n'%('sge_error.dat')
    job_string = job_string + '#$ -cwd\n'
#    job_string = job_string + '#$ -M shaunhaskey@gmail.com\n'
#    job_string = job_string + '#$ -m e\n'
    job_string = job_string + 'echo $PATH\n'
    job_string = job_string + '/u/haskeysr/bin/runmarsf > log_runmars\n'
    file = open('mars_venus.job','w')
    file.write(job_string)
    file.close()

#------------ Generate job file for Chease batch run ----------
def generate_chease_job_file(master):
    os.chdir(master['dir_dict']['chease_dir'])
    job_string = '#!/bin/bash\n'
    job_string = job_string + '#$ -N Chease_p%.3d_q%.3d\n'%(int(round(master['PMULT']*100)),int(round(master['QMULT']*100)))
    job_string = job_string + '#$ -q all.q\n'
    job_string = job_string + '#$ -o %s\n'%('sge_output.dat')
    job_string = job_string + '#$ -e %s\n'%('sge_error.dat')
    job_string = job_string + '#$ -cwd\n'
#    job_string = job_string + '#$ -M shaunhaskey@gmail.com\n'
#    job_string = job_string + '#$ -m e\n'
    job_string = job_string + 'echo $PATH\n'
    job_string = job_string + '/u/haskeysr/bin/runchease\n'
    file = open('chease.job','w')
    file.write(job_string)
    file.close()


def post_mars_matlab(file_name):
    variable_name = ['B1', 'B2', 'B3']
    end_limiter = '\n'
    out_list = []
    for i in variable_name:
        answer_real = float(extract_value(file_name, i + '_real', end_limiter))
        answer_imag = float(extract_value(file_name, i + '_imag', end_limiter))
        out_list.append(complex(answer_real, answer_imag))
    return out_list

def post_mars_r_z(directory):

    reader_r = csv.reader(open(directory+'R.out','rb'))
    r_list = []

    for row in reader_r: r_list.append(row)
    r_array = num.array(r_list,dtype=float)

    reader_z = csv.reader(open(directory+'Z.out','rb'))
    z_list = []

    for row in reader_z: z_list.append(row)
    z_array = num.array(z_list,dtype=float)
    return r_array,z_array

def find_r_z(r_array, z_array, R, Z, B1, B2, B3):
    min_arg = num.argmin((r_array-R)**2 + (z_array-Z)**2)
    return B1.ravel()[min_arg], B2.ravel()[min_arg], B3.ravel()[min_arg], r_array.ravel()[min_arg], z_array.ravel()[min_arg] 


def extractB(directory, variable_name):
    reader_real = csv.reader(open(directory+ variable_name +'_real.out','rb'))
    real_list = []

    reader_imag = csv.reader(open(directory+ variable_name + '_imag.out','rb'))
    imag_list = []

    for row in reader_real: real_list.append(row)
    for row in reader_imag: imag_list.append(row)

    output_array = num.array(real_list,dtype=float) + num.array(imag_list,dtype=float) * 1j

    return output_array



def coil_responses(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680])  # Length of probe
    Nprobe = len(probe)

    Navg = 41    # points along probe to interpolate
    Bprobem = [] # final output

    for k in range(0, Nprobe):
        print 'k', k
        #depending on poloidal/radial
        if type[k] == 1:
            print Rprobe[k],lprobe[k],tprobe[k], num.cos(tprobe[k])
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)

        r_filt = [];z_filt = [];Br_filt = [];Bz_filt = []
        Rprobek_min = num.min(Rprobek)
        Rprobek_max = num.max(Rprobek)
        Zprobek_min = num.min(Zprobek)
        Zprobek_max = num.max(Zprobek)

        #search box for interpolation to minimise computation

        #whittle down to poindts near the coil
        print Rprobek_min, Rprobek_max, Zprobek_min, Zprobek_max

        for iii in range(0,r_array.shape[0]):
            for jjj in range(0,r_array.shape[1]):
                if (((Rprobek_min <= r_array[iii,jjj]) and (r_array[iii,jjj] <= Rprobek_max)) and ((Zprobek_min <= z_array[iii,jjj]) and (z_array[iii,jjj] <= Zprobek_max))):
                    r_filt.append(r_array[iii,jjj])
                    z_filt.append(z_array[iii,jjj])
                    Br_filt.append(Br[iii,jjj])
                    Bz_filt.append(Bz[iii,jjj])

        if len(r_filt) < 30:
            print 'r_filt is low'
            r_filt = [];z_filt = [];Br_filt = [];Bz_filt = []
            Rprobek_min = Rprobek_min - 0.01
            Rprobek_max = Rprobek_max + 0.01
            Zprobek_min = Zprobek_min - 0.01
            Zprobek_max = Rprobek_max + 0.01
            for iii in range(0,r_array.shape[0]):
                for jjj in range(0,r_array.shape[1]):
                    if (((Rprobek_min <= r_array[iii,jjj]) and (r_array[iii,jjj] <= Rprobek_max)) and ((Zprobek_min <= z_array[iii,jjj]) and (z_array[iii,jjj] <= Zprobek_max))):
                        r_filt.append(r_array[iii,jjj])
                        z_filt.append(z_array[iii,jjj])
                        Br_filt.append(Br[iii,jjj])
                        Bz_filt.append(Bz[iii,jjj])
        else:
            print 'r_filt is fine'

        r_filt_array = num.array(r_filt)
        z_filt_array = num.array(z_filt)
        Br_filt_array = num.array(Br_filt)
        Bz_filt_array = num.array(Bz_filt)

        print r_filt_array.shape
        print z_filt_array.shape
        print Br_filt_array.shape

        #Create interpolation functions
        newfuncBrr = interpolate.Rbf(r_filt_array, z_filt_array, num.real(Br_filt_array), function='linear') 
        newfuncBri = interpolate.Rbf(r_filt_array, z_filt_array, num.imag(Br_filt_array),function='linear')

        newfuncBzr = interpolate.Rbf(r_filt_array, z_filt_array, num.real(Bz_filt_array), function='linear')
        newfuncBzi = interpolate.Rbf(r_filt_array, z_filt_array, num.imag(Bz_filt_array), function='linear')

        #Create interpolated values
        Brprobek = newfuncBrr(Rprobek,Zprobek) + newfuncBri(Rprobek,Zprobek)*1j
        Bzprobek = newfuncBzr(Rprobek,Zprobek) + newfuncBzi(Rprobek,Zprobek)*1j


        #Find perpendicular components
        Bprobek  =  (num.sin(tprobe[k])*num.real(Bzprobek) + num.cos(tprobe[k])*num.real(Brprobek)) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek) +num.cos(tprobe[k])*num.imag(Brprobek))

        Bprobem.append(num.average(Bprobek))
    return Bprobem


def coil_responses2(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = [] # final output

    for k in range(0, Nprobe):

        #depending on poloidal/radial
        if type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)

        Br_list=[];Bz_list=[];Bphi_list=[]
        for iii in range(0,len(Rprobek)):
            R = Rprobek[iii]
            Z = Zprobek[iii]
            Br_val, Bz_val, Bphi_val, R_val, Z_val = find_r_z(r_array, z_array, R, Z, Br, Bz, Bphi)
            #print (R-R_val)/R*100, (Z-Z_val)/Z*100
            Br_list.append(Br_val*1.0)
            Bz_list.append(Bz_val*1.0)
            Bphi_list.append(Bphi_val*1.0)
        Br_coil_array = num.array(Br_list)
        Bz_coil_array = num.array(Bz_list)
        Bphi_coil_array = num.array(Bphi_list)
        
        Bprobek  =  (num.sin(tprobe[k])*num.real(Bz_coil_array) + num.cos(tprobe[k])*num.real(Br_coil_array)) + 1j * (num.sin(tprobe[k])*num.imag(Bz_coil_array) +num.cos(tprobe[k])*num.imag(Br_coil_array))
        Bprobem.append(num.average(Bprobek))
    return Bprobem


def coil_responses3(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = [] # final output

    for k in range(0, Nprobe):

        #depending on poloidal/radial
        if type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)

        Br_list=[];Bz_list=[];Bphi_list=[]

        #Create interpolated values
        Brprobek = griddata(r_array.flatten(), z_array.flatten(), num.real(Br.flatten()), Rprobek, Zprobek) + 1j*griddata(r_array.flatten(), z_array.flatten(), num.imag(Br.flatten()), Rprobek, Zprobek)
        Bzprobek = griddata(r_array.flatten(), z_array.flatten(), num.real(Bz.flatten()), Rprobek, Zprobek) + 1j*griddata(r_array.flatten(), z_array.flatten(), num.imag(Bz.flatten()), Rprobek, Zprobek)
        

        #Find perpendicular components
        Bprobek  =  (num.sin(tprobe[k])*num.real(Bzprobek) + num.cos(tprobe[k])*num.real(Brprobek)) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek) +num.cos(tprobe[k])*num.imag(Brprobek))

        Bprobem.append(num.average(Bprobek))

    return Bprobem



def coil_responses4(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = [] # final output

    for k in range(0, Nprobe):

        #depending on poloidal/radial
        if type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)

        Br_list=[];Bz_list=[];Bphi_list=[]

        r_filt = [];z_filt = [];Br_filt = [];Bz_filt = []
        Rprobek_min = num.min(Rprobek) - 0.1
        Rprobek_max = num.max(Rprobek) + 0.1
        Zprobek_min = num.min(Zprobek) - 0.1
        Zprobek_max = num.max(Zprobek) + 0.1

        #search box for interpolation to minimise computation

        #whittle down to poindts near the coil


        for iii in range(0,r_array.shape[0]):
            for jjj in range(0,r_array.shape[1]):
                if (((Rprobek_min <= r_array[iii,jjj]) and (r_array[iii,jjj] <= Rprobek_max)) and ((Zprobek_min <= z_array[iii,jjj]) and (z_array[iii,jjj] <= Zprobek_max))):
                    r_filt.append(r_array[iii,jjj])
                    z_filt.append(z_array[iii,jjj])
                    Br_filt.append(Br[iii,jjj])
                    Bz_filt.append(Bz[iii,jjj])


        r_filt_array = num.array(r_filt)
        z_filt_array = num.array(z_filt)
        Br_filt_array = num.array(Br_filt)
        Bz_filt_array = num.array(Bz_filt)

#        print r_filt_array.shape
#        print z_filt_array.shape
#        print Br_filt_array.shape


        #Create interpolated values

        Brprobek = griddata(r_filt_array, z_filt_array, num.real(Br_filt_array), Rprobek, Zprobek) + 1j*griddata(r_filt_array, z_filt_array, num.imag(Br_filt_array), Rprobek, Zprobek)
        Bzprobek = griddata(r_filt_array, z_filt_array, num.real(Bz_filt_array), Rprobek, Zprobek) + 1j*griddata(r_filt_array, z_filt_array, num.imag(Bz_filt_array), Rprobek, Zprobek)


        #Find perpendicular components
        Bprobek  =  (num.sin(tprobe[k])*num.real(Bzprobek) + num.cos(tprobe[k])*num.real(Brprobek)) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek) +num.cos(tprobe[k])*num.imag(Brprobek))

        Bprobem.append(num.average(Bprobek))

    return Bprobem




def coil_responses_single(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6])*2*num.pi/360  # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = [] # final output

    for k in range(0, Nprobe):

        #depending on poloidal/radial
        Br_list=[];Bz_list=[];Bphi_list=[]

        R = Rprobe[k]
        Z = Zprobe[k]
        Br_val, Bz_val, Bphi_val, R_val, Z_val = find_r_z(r_array, z_array, R, Z, Br, Bz, Bphi)
        #print (R-R_val)/R*100, (Z-Z_val)/Z*100
        Br_list.append(Br_val*1.0)
        Bz_list.append(Bz_val*1.0)
        Bphi_list.append(Bphi_val*1.0)
        Br_coil_array = num.array(Br_list)
        Bz_coil_array = num.array(Bz_list)
        Bphi_coil_array = num.array(Bphi_list)
        
        Bprobek  =  (num.sin(tprobe[k])*num.real(Bz_coil_array) + num.cos(tprobe[k])*num.real(Br_coil_array)) + 1j * (num.sin(tprobe[k])*num.imag(Bz_coil_array) +num.cos(tprobe[k])*num.imag(Br_coil_array))
        Bprobem.append(num.average(Bprobek))
    return Bprobem


def modify_pickle(old_base_dir, new_base_dir, old_pickle_name, new_pickle_name):
    #old_base_dir = '/u/haskeysr/mars/project1/'
    #new_base_dir = '/u/haskeysr/mars/'
    #old_pickle_name = '5_post_RMZM.pickle'
    #new_pickle_name = '5_project_vary_ROTEblah.pickle'

    dir_dict = pickle.load(open(old_base_dir + old_pickle_name))

    def edit_string(string, old_base_dir, new_base_dir):
        string
        new_string = new_base_dir + string[len(old_base_dir):]
        return new_string

    dir_dict['details']['base_dir']=edit_string(dir_dict['details']['base_dir'],old_base_dir, new_base_dir)
    dir_dict['details']['efit_dir']=edit_string(dir_dict['details']['base_dir'],old_base_dir, new_base_dir)
    dir_dict['details']['shot_dir']=edit_string(dir_dict['details']['base_dir'],old_base_dir, new_base_dir)
    dir_dict['details']['thetac_dir']=edit_string(dir_dict['details']['base_dir'],old_base_dir, new_base_dir)

    for i in dir_dict['sims'].keys():
        for current_dir in dir_dict['sims'][i]['dir_dict'].keys():
            dir_dict['sims'][i]['dir_dict'][current_dir] = edit_string(dir_dict['sims'][i]['dir_dict'][current_dir], old_base_dir, new_base_dir)

    pickle_file = open(new_base_dir + new_pickle_name,'w')
    pickle.dump(dir_dict, pickle_file)
    pickle_file.close()
