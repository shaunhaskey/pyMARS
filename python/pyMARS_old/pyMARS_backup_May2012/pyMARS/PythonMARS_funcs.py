import numpy as num
import time, os, sys, string, re, csv, pickle
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata
from scipy.interpolate import griddata as scipy_griddata


#function to read in a stab_results file and generate a dictionary containing
#information about the equilibria
def read_stab_results(file_location):
    stab_setup_contents = open(file_location).read()
    stab_lines = stab_setup_contents.split('\n')
    line = 0

    dictionary_list = {}
    
    #skip lines at the start with ;
    #can get shot number and time out of the first line - need to update this
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
    #Don't want to pick up instances of the variable name that is part of another word:
    #This means the variable will be found multiple times if the end limiter is ' ' or '\n'
    #Need to add a catch for this so that the error isn't output
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
    #print 'equal character ' + input_string[equal_location]
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


#--------- Create working directories ---------------
def generate_directories(master, base_dir):
    dir_dict = {}    
    #dir_dict['shot_dir'] =  base_dir + 'shot' + str(master['shot']) +'/'
    dir_dict['shot_dir'] =  base_dir
    #dir_dict['thetac_dir'] = dir_dict['shot_dir'] + 'tc_%03d/'%(master['thetac']*1000)
    dir_dict['thetac_dir'] = dir_dict['shot_dir']
    dir_dict['efit_dir'] =  dir_dict['thetac_dir'] + 'efit/'
    dir_dict['q_dir'] = dir_dict['thetac_dir'] + 'qmult%.3f/'%master['QMULT']
    dir_dict['exp_dir'] = dir_dict['q_dir'] + 'exp%.3f/'%master['PMULT']
    dir_dict['mars_dir'] =  dir_dict['exp_dir'] + 'marsrun/'
    dir_dict['chease_dir'] = dir_dict['exp_dir'] + 'cheaserun/'
    dir_dict['chease_dir_PEST'] = dir_dict['exp_dir'] + 'cheaserun_PEST'
    dir_dict['mars_vac_dir'] =  dir_dict['mars_dir'] + 'RUNrfa.vac/'
    dir_dict['mars_plasma_dir'] =  dir_dict['mars_dir'] + 'RUNrfa.p/'
    #os.system('mkdir ' + dir_dict['shot_dir'])
    #os.system('mkdir ' + dir_dict['thetac_dir'])
    #os.system('mkdir ' + dir_dict['efit_dir'])
    #os.system('mkdir ' + dir_dict['q_dir'])
    #os.system('mkdir ' + dir_dict['exp_dir'])
    #os.system('mkdir ' + dir_dict['mars_dir'])
    os.system('mkdir -p ' + dir_dict['chease_dir'])
    os.system('mkdir -p ' + dir_dict['chease_dir_PEST'])
    os.system('mkdir -p ' + dir_dict['mars_vac_dir'])
    os.system('mkdir -p ' + dir_dict['mars_plasma_dir'])
    master['dir_dict']=dir_dict
    return master

#----------Create MARS directories --------------------
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
def copy_chease_files(master, PEST=0):
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
    else:
        os.chdir(master['dir_dict']['chease_dir'])
    issue_command('cp ', master['dir_dict']['efit_dir'] + master['EXPEQ_name'] + ' .')
    issue_command('ln -sf ', master['EXPEQ_name'] + ' EXPEQ')

#--------- Chease : Modify datain file (B0,R0) ------------
def modify_datain(master, template_dir, PEST=0):
    print 'Modifying CHEASE datain file'
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
    else:
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
    if PEST==1:
        replace_value('datain','NEGP', ',',str(0))
        replace_value('datain','NER', ',',str(2))



def modify_datain_replace(master,template_dir, replace_values,PEST=0):
    print 'Modifying CHEASE datain file'
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
        dict_key = 'CHEASE_settings_PEST'
        master[dict_key]['<<NEGP>>'] = 0
        master[dict_key]['<<NER>>'] = 2
    else:
        os.chdir(master['dir_dict']['chease_dir'])
        dict_key = 'CHEASE_settings'
        master[dict_key]['<<NEGP>>'] = -1
        master[dict_key]['<<NER>>'] = 1
        
    master[dict_key]['<<QSPEC>>'] = master['QMAX']
    master[dict_key]['<<B0EXP>>'] = master['B0EXP']
    master[dict_key]['<<R0EXP>>'] = master['R0EXP']
    master[dict_key]['<<NTOR>>'] = num.abs(master['MARS_settings']['<<RNTOR>>'])
    master[dict_key]['<<SHOT>>'] = master['shot']
    master[dict_key]['<<TIME>>'] = master['shot_time']

    datain_file = open(template_dir + 'datain_template','r')
    datain_text = datain_file.read()
    datain_file.close()

    for current in master[dict_key].keys():
        #print current, master[dict_key][current]
        datain_text = datain_text.replace(current,str(master[dict_key][current]))

    datain_file = open('datain','w')
    datain_file.write(datain_text)
    datain_file.close()
    return master

    
#---------Chease : Run ------------------------------
def execute_chease(master, PEST=0):
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
    else:
        os.chdir(master['dir_dict']['chease_dir'])
    print 'Start Chease executable'
    os.system('/u/haskeysr/bin/runchease')
    #os.system('bash -i -c /u/lanctot/chease/CheaseMerge20080822/bin/LinuxPortland64/chease.x < datain > log_chease')
    print 'Finished Chease Executable'

#---------Chease : Extract NW from log_chease--------------
def extract_NW(master):
    os.chdir(master['dir_dict']['chease_dir'])
    master['NW'] = int(round(float(extract_value('log_chease','NW',' '))))
    #print 'NW : ' + str(master['NW'])
    return master

#-------- fxrun section ---------------
#def fxin_create(master,M1,M2,R0EXP,B0EXP, PEST=0):
def fxin_create(dir, M1, M2,R0EXP,B0EXP):
    print 'creating fxin and runing fxrun'
    os.chdir(dir)
    fxin = open('fxin','w')
    fxin.write('%d\n%.8f\n%.8f\n'%(abs(M1)+abs(M2)+1,R0EXP,B0EXP))
    fxin.close()

def fxrun(master, PEST=0):
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
    else:
        os.chdir(master['dir_dict']['chease_dir'])
    os.system("bash -i -c 'fxrun'") #need bash -i -c to force a read of .bashrc to understand alias??
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
    def extract_RMZM_output(file,variable):
        value = extract_value(file, variable, '\n', strip_spaces = 0)
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
#Link the PROFDEN, PROFROT, and chease outputs into the mars directory
def mars_setup_files(master, vac):
    if vac==1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
    os.system('ln -sf ../../../../efit/PROFDEN PROFDEN')
    os.system('ln -sf ../../../../efit/PROFROT PROFROT')
    os.system('ln -sf ../../cheaserun/OUTRMAR OUTRMAR')
    os.system('ln -sf ../../cheaserun/OUTVMAR OUTVMAR')
    os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F')
    os.system('ln -sf ../../cheaserun/log_chease log_chease')


#--------- Calculate Alvfen velociy  -------------------
# and related alfven velocity scaled values :
# for mars input file : ROTE, OMEGA_NORM, TAUWM, v0a
def mars_setup_alfven(master, input_frequency, vac):
    mu0 = 4e-7 * num.pi
    mi = 1.6726e-27 + 1.6749e-27 #rest mass of deuterium kg
    e = 1.60217e-19 #coulombs

    if vac ==1:
        os.chdir(master['dir_dict']['mars_vac_dir'])

    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])

    PROFDEN_file = open('PROFDEN','r')
    PROFDEN_data = PROFDEN_file.readlines()
    #print PROFDEN_data[1]

    pattern = ''
    PROFDEN_data[1]
    re.search(pattern, PROFDEN_data[1])
    pattern = '\d+.\d+e*\+*\-*\d*'
    string1 = re.search(pattern, PROFDEN_data[1])
    string2 = re.search(pattern, PROFDEN_data[1][string1.end()+1:])
    ne0_r = float(PROFDEN_data[1][string1.start():string1.end()])
    ne0 = float(PROFDEN_data[1][string2.start()+string1.end()+1:string2.end()+string1.end()+1])


    #rotation data
    #print 'Rotation section ------------' 
    PROFROT_file = open('PROFROT','r')
    PROFROT_data = PROFROT_file.readlines()
    #print PROFROT_data[1]

    pattern = ''
    re.search(pattern, PROFROT_data[1])
    pattern = '\d+.\d+e*\+*\-*\d*'
    string1 = re.search(pattern, PROFROT_data[1])
    string2 = re.search(pattern, PROFROT_data[1][string1.end()+1:])
    vtor0_r = float(PROFROT_data[1][string1.start():string1.end()])
    vtor0 = float(PROFROT_data[1][string2.start()+string1.end()+1:string2.end()+string1.end()+1])
    #print 'vtor0_r', vtor0_r, 'vtor0', vtor0
    
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
    output_string = ['From /u/lanctot/mars/utils/MARSplot/write_mars_params.pro']
    output_string.append('=======================================')
    output_string.append('Input values')
    output_string.append('=======================================')
    output_string.append('B0EXP (T)                  : %.4f'%(B0EXP))
    output_string.append('R0EXP (m)                  : %.4f'%(R0EXP))
    output_string.append('Central density (m^-3)    : %.4f'%(ne0))
    output_string.append('Tau wall physics (s)      : %.4f'%(tauwp))
    output_string.append('=======================================')
    output_string.append('Normalized values')
    output_string.append('=======================================')
    output_string.append('Central Alfven speed (m/s)    : %.4f'%(v0a))
    output_string.append('Central Alfven time (s)       :%.4f'%(taua))
    output_string.append('Central Alfven Frequency (1/s): %.4f'%(1.0/taua))
    output_string.append('TAUW              : %.4f'%(tauwm))
    output_string.append('ROTE              : %.4f'%(vtorn))
    output_string.append('Thermal OMEGACIO : %.4f'%(fcio))
#    IF nprofda GT 0 THEN  print 'Fast OMEGACIO    : ',fcio_fast
    output_string.append('=======================================')
    output_string.append('Coil Frequencies')
    output_string.append('=======================================')
    output_string.append('I-coil (Hz)      OMEGA')
    for k in range(0, len(ichz)): output_string.append('%.4e       %.4e'%(ichz[k], omega[k]))
    output_string.append('=======================================')
                                                                                                   
    output_string.append('B0EXP: %.8f'%(B0EXP))
    output_string.append('R0EXP: %.8f'%(R0EXP))
    output_string.append('tauwp: %.8f'%(tauwp))
    output_string.append('tauwm: %.8f'%(tauwm))
    output_string.append('f_v0a: %.8f'%(f_v0a))
    output_string.append('fcio: %.8f'%(fcio))

    for i in range(0,len(output_string)):
        output_string[i]+='\n'
    alf_calc_file = open(master['dir_dict']['mars_dir']+'Alf_calcs.txt','w')
        
    alf_calc_file.writelines(output_string)
    alf_calc_file.close()
    
    master['ROTE'] = vtorn
    master['OMEGA_NORM'] = omega[0]
    master['TAUWM'] = tauwm
    master['v0a'] = v0a
    print 'ROTE:', vtorn, ' OMEGA_NORM:', omega[0],' TAUWM:', tauwm, ' v0a:', v0a
    return master


#------Setup the MARS RUN file
'''
def mars_setup_run_file(master, template_dir, vac):
    print 'Modify Mars run file'
    if vac == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
        INCFEED = 4
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
        INCFEED = 8
        
    #Copy the template across
    os.system('cp '+template_dir+'RUN .')

    #Replace with relevant values for this particular run:
    replace_value('RUN','INCFEED', ',', str(INCFEED))
    replace_value('RUN','RNTOR', ',', str(master['NTOR']))
    FCCHI_string = str(master['FCCHI'][0]) + ', ' + str(master['FCCHI'][1]) + ','
    FWCHI_string = str(master['FWCHI'][0]) + ', ' + str(master['FWCHI'][1]) + ','
    AL0_string = '( 0, ' + str(master['OMEGA_NORM']) + '),'
                     
    replace_value('RUN','FCCHI','\n',FCCHI_string)
    replace_value('RUN','FWCHI','\n',FWCHI_string)

    print AL0_string
    replace_value('RUN','AL0','\n',AL0_string)
    replace_value('RUN','FEEDI','\n',master['FEEDI'])

    replace_value('RUN','M1',',',str(master['M1']))
    replace_value('RUN','M2',',',str(master['M2']))
    replace_value('RUN','TAUW',',',str(master['TAUWM']))
    replace_value('RUN','IWALL',',',str(master['NW']))

    #Set IFEED value at the NW calculated value, or just inside it so its not on the wall
    if master['IFEED'][0]>= master['NW']:
        print "Possible error IFEED value is greater than or equal to NW :" + str(master['NW']) + ', IFEED' + str(master['IFEED'][0])
        replace_value('RUN','IFEED', ',', str(master['NW']-1))
        replace_value('RUN','ISENS', ',', str(master['NW']-1)) #because MARS cares about this!!! for some reason
    else:
        replace_value('RUN','IFEED', ',', str(master['IFEED'][0]))
        replace_value('RUN','ISENS', ',', str(master['IFEED'][0]))
'''

def mars_setup_run_file_new(master, template_dir, vac=0):
    #print 'Modify Mars run file'
    dict_key = 'MARS_settings'
    if vac == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
        master[dict_key]['<<INCFEED>>'] = 4
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
        master[dict_key]['<<INCFEED>>'] = 8

    master[dict_key]['<<FCCHI>>'] = str(master['FCCHI'][0]) + ', ' + str(master['FCCHI'][1])
    master[dict_key]['<<FWCHI>>'] = str(master['FWCHI'][0]) + ', ' + str(master['FWCHI'][1])
    master[dict_key]['<<ISENS>>'] = master['IFEED']
    master[dict_key]['<<IWALL>>'] = master['NW']
    master[dict_key]['<<TAUW>>'] = master['TAUWM']
    #master[dict_key]['<<FEEDI>>'] = master['FEEDI']
    master[dict_key]['<<AL0>>'] = '( 0, ' + str(master['OMEGA_NORM']) + ')'
     
    if master['IFEED'][0]>= master['NW']:
        print "Possible error IFEED value is greater than or equal to NW :" + str(master['NW']) + ', IFEED' + str(master['IFEED'][0])
        master[dict_key]['<<IFEED>>'] = master['NW']-1
        master[dict_key]['<<ISENS>>'] = master['NW']-1
    else:
        master[dict_key]['<<IFEED>>'] = master['IFEED'][0]
        master[dict_key]['<<ISENS>>'] = master['IFEED'][0]

    RUN_file = open(template_dir + 'RUN_template','r')
    RUN_text = RUN_file.read()
    RUN_file.close()

    for current in master[dict_key].keys():
        #print current, master[dict_key][current]
        RUN_text = RUN_text.replace(current,str(master[dict_key][current]))

    RUN_file = open('RUN','w')
    RUN_file.write(RUN_text)
    RUN_file.close()
    return master

'''
def mars_setup_run_file_single(master, template_dir, coil, vac):
    print 'Modify Mars run file'
    if vac == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
        INCFEED = 4
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
        INCFEED = 8
        
    #Copy the template across
    os.system('cp '+template_dir+'RUN .')

    #Replace with relevant values for this particular run:
    replace_value('RUN','INCFEED', ',', str(INCFEED))
    replace_value('RUN','RNTOR', ',', '-'+str(master['NTOR']))
    if coil == 'upper':
        loc = 0
    else:
        loc = 1
    FCCHI_string = str(master['FCCHI'][loc]) + ','
    FWCHI_string = str(master['FWCHI'][loc]) + ','
    AL0_string = '( 0, ' + str(master['OMEGA_NORM']) + '),'
                     
    replace_value('RUN','FCCHI','\n',FCCHI_string)
    replace_value('RUN','FWCHI','\n',FWCHI_string)

    print AL0_string
    replace_value('RUN','AL0','\n',AL0_string)
    replace_value('RUN','FEEDI','\n',master['FEEDI'])

    replace_value('RUN','M1',',',str(master['M1']))
    replace_value('RUN','M2',',',str(master['M2']))
    replace_value('RUN','TAUW',',',str(master['TAUWM']))
    replace_value('RUN','IWALL',',',str(master['NW']))
    replace_value('RUN','NCOIL',',',str(1))

    #Set IFEED value at the NW calculated value, or just inside it so its not on the wall
    if master['IFEED'][0]>= master['NW']:
        print "Possible error IFEED value is greater than or equal to NW :" + str(master['NW']) + ', IFEED' + str(master['IFEED'][0])
        replace_value('RUN','IFEED', ',', str(master['NW']-1))
        replace_value('RUN','ISENS', ',', str(master['NW']-1)) #because MARS cares about this!!! for some reason
    else:
        replace_value('RUN','IFEED', ',', str(master['IFEED'][0]))
        replace_value('RUN','ISENS', ',', str(master['IFEED'][0]))

'''

#-------Special file to modify a special variable -eg ROTE------
def mars_setup_run_file_special(master, name, val, vac):
    print 'Modify Mars run file'
    if vac == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])                     
    replace_value('RUN',name,',',val)



#----------- Generate Job file for MARS batch run -----------
def generate_job_file(master,vac_true, id_string = 'MARS', rm_files = 'OUTDATA  JPLASMA VPLASMA PLASMA JACOBIAN'):
    if vac_true == 1:
        os.chdir(master['dir_dict']['mars_vac_dir'])
        type_text = 'vac'
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])
        type_text = 'p'
    job_string = '#!/bin/bash\n'
    job_string = job_string + '#$ -N ' + id_string + '_p%d_q%d\n'%(int(round(master['PMULT']*100)),int(round(master['QMULT']*100)))
    job_string = job_string + '#$ -q all.q\n'
    job_string = job_string + '#$ -o %s\n'%('sge_output.dat')
    job_string = job_string + '#$ -e %s\n'%('sge_error.dat')
    job_string = job_string + '#$ -cwd\n'
#    job_string = job_string + '#$ -M shaunhaskey@gmail.com\n'
#    job_string = job_string + '#$ -m e\n'
    job_string = job_string + 'echo $PATH\n'
    job_string = job_string + '/u/haskeysr/bin/runmarsf > log_runmars\n'
    job_string = job_string + 'rm ' + rm_files
    file = open('mars_venus.job','w')
    file.write(job_string)
    file.close()

#------------ Generate job file for Chease batch run ----------
def generate_chease_job_file(master, PEST=0, fxrun = 0, id_string = 'Chease_', rm_files = ''):
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
    else:
        os.chdir(master['dir_dict']['chease_dir'])
        
    job_string = '#!/bin/bash\n'
    job_string = job_string + '#$ -N ' + id_string + 'p%.3d_q%.3d\n'%(int(round(master['PMULT']*100)),int(round(master['QMULT']*100)))
    job_string = job_string + '#$ -q all.q\n'
    job_string = job_string + '#$ -o %s\n'%('sge_output.dat')
    job_string = job_string + '#$ -e %s\n'%('sge_error.dat')
    job_string = job_string + '#$ -cwd\n'
#    job_string = job_string + '#$ -M shaunhaskey@gmail.com\n'
#    job_string = job_string + '#$ -m e\n'
    job_string = job_string + 'echo $PATH\n'
    job_string = job_string + '/u/haskeysr/bin/runchease\n'
    if fxrun==1:
        job_string += 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/c/source/PGI/pgi/linux86/8.0-2/lib\n'

        job_string += '/u/reimerde/mars/MarsF20060714/FourierRF/FourierRF.x < fxin > fxrun_log.log\n'
        #job_string += 'fxrun > fxrun_log.log\n'

    job_string += 'rm '+ rm_files
    file_name = open('chease.job','w')
    print 'opening file'
    file_name.write(job_string)
    file_name.close()

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
    print directory
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
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe
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
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

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

    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe
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
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe
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


def coil_responses5(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = [] # final output
    Rprobek_total = []
    Zprobek_total = []
    start_time = time.time()
    for k in range(0, Nprobe):

        #depending on poloidal/radial
        if probe_type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
        Rprobek_total.append(Rprobek)
        Zprobek_total.append(Zprobek)
        Br_list=[];Bz_list=[];Bphi_list=[]

        r_filt = [];z_filt = [];Br_filt = [];Bz_filt = []
        overshoot = 0.1
        Rprobek_min = num.min(Rprobek) - overshoot
        Rprobek_max = num.max(Rprobek) + overshoot
        Zprobek_min = num.min(Zprobek) - overshoot
        Zprobek_max = num.max(Zprobek) + overshoot

        #search box for interpolation to minimise computation

        #whittle down to points near the coil


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
        #print 'unfilt'
        #r_filt_array = r_array.flatten()
        #z_filt_array = z_array.flatten()
        #Br_filt_array = Br.flatten()
        #Bz_filt_array = Bz.flatten()
        print r_filt_array.shape
        coords_array = num.ones((len(r_filt_array),2),dtype=float)
        coords_array[:,0] = r_filt_array
        coords_array[:,1] = z_filt_array

        interp_coords_array = num.ones((len(Rprobek),2),dtype=float)
        interp_coords_array[:,0] = Rprobek
        interp_coords_array[:,1] = Zprobek
        
        Brprobek = scipy_griddata(coords_array, num.real(Br_filt_array), interp_coords_array,method='linear') + 1j*scipy_griddata(coords_array, num.imag(Br_filt_array), interp_coords_array,method ='linear')
        Bzprobek = scipy_griddata(coords_array, num.real(Bz_filt_array), interp_coords_array,method='linear') + 1j*scipy_griddata(coords_array, num.imag(Bz_filt_array), interp_coords_array,method ='linear')
        #Brprobek2 = scipy_griddata((r_array.flatten(),z_array.flatten()), Br.flatten(), (Rprobek,Zprobek),method='cubic')
        #Bzprobek2 = scipy_griddata((r_array.flatten(),z_array.flatten()), Bz.flatten(), (Rprobek,Zprobek),method='cubic')
        #Brprobek2 = scipy_griddata((r_filt_array,z_filt_array), Br_filt_array, (Rprobek,Zprobek),method='cubic')
        #Bzprobek2 = scipy_griddata((r_filt_array,z_filt_array), Bz_filt_array, (Rprobek,Zprobek),method='cubic')
        #print 'Br difference'
        #print num.sum(num.abs(Brprobek-Brprobek2))
        #print 'Bz_difference'
        #print num.sum(num.abs(Bzprobek-Bzprobek2))

        #Create interpolated values

        #Brprobek = griddata(r_filt_array, z_filt_array, num.real(Br_filt_array), Rprobek, Zprobek) + 1j*griddata(r_filt_array, z_filt_array, num.imag(Br_filt_array), Rprobek, Zprobek)
        #Bzprobek = griddata(r_filt_array, z_filt_array, num.real(Bz_filt_array), Rprobek, Zprobek) + 1j*griddata(r_filt_array, z_filt_array, num.imag(Bz_filt_array), Rprobek, Zprobek)

        #Find perpendicular components
        Bprobek  =  (num.sin(tprobe[k])*num.real(Bzprobek) + num.cos(tprobe[k])*num.real(Brprobek)) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek) +num.cos(tprobe[k])*num.imag(Brprobek))

        Bprobem.append(num.average(Bprobek))
    print 'sep total time :', time.time() - start_time    
    R_tot_array = num.array(Rprobek_total)
    Z_tot_array = num.array(Zprobek_total)
    start_time = time.time()

    #seems as though it must be linear interpolation, otherwise there are sometimes problems
    Brprobek2 = scipy_griddata((r_array.flatten(),z_array.flatten()), Br.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear')
    Bzprobek2 = scipy_griddata((r_array.flatten(),z_array.flatten()), Bz.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear')
    Brprobek2 = num.resize(Brprobek2,R_tot_array.shape) #reshape back into individual coils 
    Bzprobek2 = num.resize(Brprobek2,R_tot_array.shape) #reshape back into individual coils
    Bprobem2 = []
    for k in range(0, Nprobe):
        #print k,tprobe[k],Brprobek2[k,:].shape
        #print (num.sin(tprobe[k])*num.real(Bzprobek2[k,:]))# + num.cos(tprobe[k])*num.real(Brprobek2[k,:]))
        #print 1j * (num.sin(tprobe[k])*num.imag(Bzprobek2[k,:]))# +num.cos(tprobe[k])*num.imag(Brprobek2[k,:]))
        #num.cos(tprobe[k])*num.real(Brprobek2[k,:])
        Bprobek2 = (num.sin(tprobe[k])*num.real(Bzprobek2[k,:]) + num.cos(tprobe[k])*num.real(Brprobek2[k,:])) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek2[k,:]) +num.cos(tprobe[k])*num.imag(Brprobek2[k,:]))
        Bprobem2.append(num.average(Bprobek2))


    print 'tog total time :', time.time() - start_time
    print Bprobem
    print Bprobem2

    return Bprobem


def coil_responses6(r_array,z_array,Br,Bz,Bphi, probe, probe_type, Rprobe,Zprobe,tprobe,lprobe):
    #probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    #probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    #Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    #Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    #tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    #lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = []; Rprobek_total = []; Zprobek_total = []

    start_time = time.time()
    for k in range(0, Nprobe):
        #depending on poloidal/radial - what is really going on here? why is there a difference between the two cases?
        if probe_type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
        Rprobek_total.append(Rprobek)
        Zprobek_total.append(Zprobek)

    R_tot_array = num.array(Rprobek_total)
    Z_tot_array = num.array(Zprobek_total)

    #must be linear interpolation, otherwise there are sometimes problems
    Brprobek = num.resize(scipy_griddata((r_array.flatten(),z_array.flatten()), Br.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear'),R_tot_array.shape)
    Bzprobek = num.resize(scipy_griddata((r_array.flatten(),z_array.flatten()), Bz.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear'),Z_tot_array.shape)
    Bprobem = []

    #calculate normal to coil and average over data points
    for k in range(0, Nprobe):
        Bprobem.append(num.average((num.sin(tprobe[k])*num.real(Bzprobek[k,:]) + num.cos(tprobe[k])*num.real(Brprobek[k,:])) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek[k,:]) +num.cos(tprobe[k])*num.imag(Brprobek[k,:]))))
    print 'total time :', time.time() - start_time 
    return Bprobem




def coil_responses_single(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe
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
def dump_data(data, file_name):
    opened_file = open(file_name,'w')
    pickle.dump(data, opened_file)
    opened_file.close()

def read_data(file_name):
    opened_file = open(file_name,'r')
    data = pickle.load(opened_file)
    opened_file.close()
    return data
