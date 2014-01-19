import numpy as num
import numpy as np
import time, os, sys, string, re, csv, pickle
import scipy.interpolate as interpolate
from scipy.interpolate import griddata as scipy_griddata


def extract_surfmn_data(filename, n):
    a = file(filename,'r').readlines()
    tmp = a[0].rstrip('\n').split(" ")
    unfinished=1
    while unfinished:
        try:
            tmp.remove("")
            #print 'removed something'
        except ValueError:
            #print 'finished?'
            unfinished = 0
    print tmp
    nst = int(tmp[0])
    nfpts = int(tmp[1])
    irpt = int(tmp[2])
    iradvar = int(tmp[3])
    khand = int(tmp[4])
    gfile = tmp[5]
    imax = nst -1
    jmax = 2*nfpts
    kmax = nfpts

    ms = np.arange(-nfpts, nfpts+1,1)
    ns = np.arange(0,nst+1,1)

    rvals = np.fromstring(a[1],dtype=float,sep=" ")
    qvals = np.fromstring(a[2],dtype=float,sep=" ")


    adat = np.zeros((nst, 2*nfpts+1, nfpts+1),dtype=float)
    line_num = 3
    for i in range(0,nst):
        for j in range(0,2*nfpts+1):
            tmp = np.fromstring(a[line_num],dtype=float,sep=" ")
            adat[i,j,:]=tmp[:]
            line_num += 1
    qlvals = np.fromstring(a[line_num],dtype=float,sep=" ")
    zdat = adat[:,:,n]
    xdat = np.tile(ms,(zdat.shape[0],1)).transpose()
    ydat = np.tile(rvals, (zdat.shape[1],1))
    zdat = zdat.transpose()

    return qlvals, xdat, ydat, zdat

#create the string version of FEEDI required for phasing
def construct_FEEDI(phase):
    real_part = num.cos(phase/360.*2*num.pi)
    imag_part = num.sin(phase/360.*2*num.pi)

    FEEDI_string =  '(1.0,0.0),(%.5f, %.5f)'%(real_part, imag_part)
    return FEEDI_string

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


def read_stab_results_serial(file_location):
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

            dictionary_list[int(current_value[-1])]={}
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
def generate_directories(master, base_dir, multiple_efits = 0):
    print multiple_efits
    dir_dict = {}    
    #dir_dict['shot_dir'] =  base_dir + 'shot' + str(master['shot']) +'/'
    dir_dict['shot_dir'] =  base_dir
    #dir_dict['thetac_dir'] = dir_dict['shot_dir'] + 'tc_%03d/'%(master['thetac']*1000)
    dir_dict['thetac_dir'] = dir_dict['shot_dir']
    if multiple_efits:
        dir_dict['efit_dir'] =  base_dir + 'efit/'+str(master['shot_time']) +'/'
    else:
        dir_dict['efit_dir'] =  base_dir + 'efit/'
    if multiple_efits:
        dir_dict['q_dir'] = dir_dict['thetac_dir'] + '/times/' + str(master['shot_time']) + '/'
        dir_dict['exp_dir'] = dir_dict['q_dir']
    else:
        dir_dict['q_dir'] = dir_dict['thetac_dir'] + 'qmult%.3f/'%master['QMULT']
        dir_dict['exp_dir'] = dir_dict['q_dir'] + 'exp%.3f/'%master['PMULT']
    dir_dict['mars_dir'] =  dir_dict['exp_dir'] + 'marsrun/'
    dir_dict['chease_dir'] = dir_dict['exp_dir'] + 'cheaserun/'
    dir_dict['chease_dir_PEST'] = dir_dict['exp_dir'] + 'cheaserun_PEST'
    dir_dict['mars_vac_dir'] =  dir_dict['mars_dir'] + 'RUNrfa.vac/'
    dir_dict['mars_plasma_dir'] =  dir_dict['mars_dir'] + 'RUNrfa.p/'
    print dir_dict['chease_dir']
    os.chdir(base_dir)
    os.system('mkdir -p ' + dir_dict['chease_dir'])
    print dir_dict['chease_dir_PEST']
    os.system('mkdir -p ' + dir_dict['chease_dir_PEST'])
    print dir_dict['mars_vac_dir']
    os.system('mkdir -p ' + dir_dict['mars_vac_dir'])
    print dir_dict['mars_plasma_dir']
    os.system('mkdir -p ' + dir_dict['mars_plasma_dir'])
    master['dir_dict']=dir_dict
    return master

#---------- End Corsica ---------------
#---------- Chease : Copy files required--------------
def copy_chease_files(master, PEST=0):
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
    else:
        os.chdir(master['dir_dict']['chease_dir'])
    issue_command('cp ', master['dir_dict']['efit_dir'] + master['EXPEQ_name'] + ' .')
    issue_command('ln -sf ', master['EXPEQ_name'] + ' EXPEQ')


def modify_datain(master,template_dir, replace_values,CHEASE_template = 'datain_template',PEST=0):
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

    datain_file = open(template_dir + CHEASE_template,'r')
    datain_text = datain_file.read()
    datain_file.close()

    for current in master[dict_key].keys():
        #print current, master[dict_key][current]
        datain_text = datain_text.replace(current,str(master[dict_key][current]))

    datain_file = open('datain','w')
    datain_file.write(datain_text)
    datain_file.close()
    return master

    
#---------Chease : Extract NW from log_chease--------------
def extract_NW(master):
    os.chdir(master['dir_dict']['chease_dir'])
    master['NW'] = int(round(float(extract_value('log_chease','NW',' '))))
    #print 'NW : ' + str(master['NW'])
    return master


#-------- fxrun section ---------------
#def fxin_create(master,M1,M2,R0EXP,B0EXP, PEST=0):
def fxin_create(dir, M1, M2,R0EXP,B0EXP):
    print 'creating fxin'
    os.chdir(dir)
    fxin = open('fxin','w')
    fxin.write('%d\n%.8f\n%.8f\n'%(abs(M1)+abs(M2)+1,R0EXP,B0EXP))
    fxin.close()

#not used anymore
def fxrun(master, PEST=0):
    if PEST==1:
        os.chdir(master['dir_dict']['chease_dir_PEST'])
    else:
        os.chdir(master['dir_dict']['chease_dir'])
    os.system("bash -i -c 'fxrun'") #need bash -i -c to force a read of .bashrc to understand alias??
    print 'Finished fxrun section'

##### I-coil location using RZplot #####
def modify_RMZM_F2(master, main_template):
    os.chdir(master['dir_dict']['chease_dir'])

    os.system('cp ' + main_template + ' MacMainD3D_current.m')
    SDIR_newline = "SDIR='"+ master['dir_dict']['chease_dir']+"';"

    #modify Matlab file
    modify_input_file('MacMainD3D_current.m', 'SDIR=', SDIR_newline)
    replace_value('MacMainD3D_current.m','Mac.resetCoil', ';', str(1))
    replace_value('MacMainD3D_current.m','Mac.Nm2', ';', str(1+int(abs(master['MARS_settings']['<<M1>>'])+abs(master['MARS_settings']['<<M2>>']))))
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


def mars_link_files(directory, special_dir = ''):
    os.chdir(directory)
    #if special_dir == '':
    os.system('ln -sf ../../../../efit/' + special_dir + '/PROFDEN PROFDEN')
    os.system('ln -sf ../../../../efit/' + special_dir + '/PROFROT PROFROT')
    #else:
    #os.system('ln -sf ../../../efit/' + special_dir + '/PROFDEN PROFDEN')
    #os.system('ln -sf ../../../efit/' + special_dir + '/PROFROT PROFROT')
    os.system('ln -sf ../../cheaserun/OUTRMAR OUTRMAR')
    os.system('ln -sf ../../cheaserun/OUTVMAR OUTVMAR')
    os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F')
    os.system('ln -sf ../../cheaserun/log_chease log_chease')

#--------- Mars Vacuum : setup -------------------
#Link the PROFDEN, PROFROT, and chease outputs into the mars directory
def mars_setup_files(master, special_dir = '', upper_and_lower = 0):
    if upper_and_lower==1:
        master['dir_dict']['mars_dir'] = master['dir_dict']['exp_dir']+'/RES{:.2f}_ROTE{:.4f}/'.format(master['MARS_settings']['<<ETA>>']*1e8,master['MARS_settings']['<<ROTE>>']*100)
        master['dir_dict']['mars_upper_plasma_dir']=master['dir_dict']['mars_dir']+'RUN_rfa_upper.p'
        master['dir_dict']['mars_lower_plasma_dir']=master['dir_dict']['mars_dir']+'RUN_rfa_lower.p'
        master['dir_dict']['mars_upper_vacuum_dir']=master['dir_dict']['mars_dir']+'RUN_rfa_upper.vac'
        master['dir_dict']['mars_lower_vacuum_dir']=master['dir_dict']['mars_dir']+'RUN_rfa_lower.vac'
        os.system('mkdir -p ' + master['dir_dict']['mars_upper_plasma_dir'])
        os.system('mkdir -p ' + master['dir_dict']['mars_lower_plasma_dir'])
        os.system('mkdir -p ' + master['dir_dict']['mars_lower_vacuum_dir'])
        os.system('mkdir -p ' + master['dir_dict']['mars_upper_vacuum_dir'])
        mars_link_files(master['dir_dict']['mars_upper_plasma_dir'], special_dir = special_dir)
        mars_link_files(master['dir_dict']['mars_lower_plasma_dir'], special_dir = special_dir)
        mars_link_files(master['dir_dict']['mars_lower_vacuum_dir'], special_dir = special_dir)
        mars_link_files(master['dir_dict']['mars_upper_vacuum_dir'], special_dir = special_dir)
    else:
        master['dir_dict']['mars_plasma_dir']=master['dir_dict']['mars_dir']+'RUN_rfa_RES{:.2f}_ROTE{:.2f}.p'.format(master['MARS_settings']['<<ETA>>']*1e8,master['MARS_settings']['<<ROTE>>'])
        master['dir_dict']['mars_vac_dir']=master['dir_dict']['mars_dir']+'RUN_rfa_RES{:.2f}_ROTE{:.2f}.vac'.format(master['MARS_settings']['<<ETA>>']*1e8,master['MARS_settings']['<<ROTE>>'])
        mars_link_files(master['dir_dict']['mars_plasma_dir'], special_dir = special_dir)
        mars_link_files(master['dir_dict']['mars_vac_dir'], special_dir = special_dir)



#--------- Calculate Alvfen velociy  -------------------
# and related alfven velocity scaled values :
# for mars input file : ROTE, OMEGA_NORM, TAUWM, v0a
def mars_setup_alfven(master, input_frequency, upper_and_lower=0):
    mu0 = 4e-7 * num.pi
    mi = 1.6726e-27 + 1.6749e-27 #rest mass of deuterium kg
    e = 1.60217e-19 #coulombs

    if upper_and_lower==0:
        os.chdir(master['dir_dict']['mars_vac_dir'])
    else:
        os.chdir(master['dir_dict']['mars_upper_vacuum_dir'])


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

    #omega = ichz/f_v0a
    #Fixed based on email from Matt on 8/1/2014
    omega = 2.0*np.pi*ichz*taua

    output_string = ['From /u/lanctot/mars/utils/MARSplot/write_mars_params.pro with mod from 8/1/2014']
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

def mars_edit_run_file(directory, settings, template_file):
    os.chdir(directory)
    RUN_file = open(template_file,'r')
    RUN_text = RUN_file.read()
    RUN_file.close()
    for current in settings.keys():
        #print current, master[dict_key][current]
        RUN_text = RUN_text.replace(current,str(settings[current]))
    RUN_file = open('RUN','w')
    RUN_file.write(RUN_text)
    RUN_file.close()
    
def mars_setup_run_file_new(master, template_file, upper_and_lower=0):
    #print 'Modify Mars run file'
    dict_key = 'MARS_settings'
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
        print '   IFEED set to :%d'%(master[dict_key]['<<IFEED>>'])

    else:
        master[dict_key]['<<IFEED>>'] = master['IFEED'][0]
        master[dict_key]['<<ISENS>>'] = master['IFEED'][0]


    if upper_and_lower == 1:
        master[dict_key]['<<INCFEED>>'] = 4
        master[dict_key]['<<FEEDI>>'] = '(1.0,0.0),(0.0, 0.0)'
        mars_edit_run_file(master['dir_dict']['mars_upper_vacuum_dir'], master[dict_key], template_file)
        master[dict_key]['<<INCFEED>>'] = 8
        mars_edit_run_file(master['dir_dict']['mars_upper_plasma_dir'], master[dict_key], template_file)
        master[dict_key]['<<FEEDI>>'] = '(0.0,0.0),(1.0, 0.0)'
        mars_edit_run_file(master['dir_dict']['mars_lower_plasma_dir'], master[dict_key], template_file)
        master[dict_key]['<<INCFEED>>'] = 4
        mars_edit_run_file(master['dir_dict']['mars_lower_vacuum_dir'], master[dict_key], template_file)
    else:
        master[dict_key]['<<INCFEED>>'] = 4
        mars_edit_run_file(master['dir_dict']['mars_vac_dir'], master[dict_key], template_file)
        master[dict_key]['<<INCFEED>>'] = 8
        mars_edit_run_file(master['dir_dict']['mars_plasma_dir'], master[dict_key], template_file)

    return master




#----------- Generate Job file for MARS batch run -----------
def generate_job_file(master,MARS_execution_script, id_string = 'MARS', rm_files = 'OUTDATA  JPLASMA VPLASMA PPLASMA JACOBIAN', rm_files2 = '', upper_and_lower=0):
    os.chdir(master['dir_dict']['mars_dir'])

    if upper_and_lower == 1:
        execution_txt = 'cd '+ master['dir_dict']['mars_upper_vacuum_dir'] + '\n'
        execution_txt += MARS_execution_script + ' > log_runmars\n'
        execution_txt += 'rm ' + rm_files +'\n'
        execution_txt += 'cd '+ master['dir_dict']['mars_lower_vacuum_dir'] + '\n'
        execution_txt += MARS_execution_script + ' > log_runmars\n'
        execution_txt += 'rm ' + rm_files +'\n'

        execution_txt += 'cd '+ master['dir_dict']['mars_upper_plasma_dir'] + '\n'
        execution_txt += MARS_execution_script + ' > log_runmars\n'
        execution_txt += 'rm ' + rm_files +'\n'
        execution_txt += 'cd '+ master['dir_dict']['mars_lower_plasma_dir'] + '\n'
        execution_txt += MARS_execution_script+ ' > log_runmars\n'
        execution_txt += 'rm ' + rm_files +'\n'


    else:
        execution_txt = 'cd '+ master['dir_dict']['mars_vac_dir'] + '\n'
        execution_txt += MARS_execution_script +' > log_runmars\n'
        execution_txt += 'rm ' + rm_files +'\n'

        execution_txt += 'cd '+ master['dir_dict']['mars_plasma_dir'] + '\n'
        execution_txt += MARS_execution_script + ' > log_runmars\n'
        execution_txt += 'rm ' + rm_files +'\n'

    execution_txt += 'cd '+ master['dir_dict']['chease_dir'] + '\n'
    execution_txt += 'rm ' + rm_files2 +'\n'

    job_string = '#!/bin/bash\n'
    job_string = job_string + '#$ -N ' + id_string + '_p%d_q%d\n'%(int(round(master['PMULT']*100)),int(round(master['QMULT']*100)))
    job_string = job_string + '#$ -q all.q\n'
    job_string = job_string + '#$ -o %s\n'%('sge_output.dat')
    job_string = job_string + '#$ -e %s\n'%('sge_error.dat')
    job_string = job_string + '#$ -cwd\n'
    job_string = job_string + 'echo $PATH\n'
    job_string += execution_txt
    #job_string = job_string + 'runmarsf > log_runmars\n'
    #job_string = job_string + 'rm ' + rm_files +'\n'
    file = open('mars_venus.job','w')
    file.write(job_string)
    file.close()


#------------ Generate job file for Chease batch run ----------
def generate_chease_job_file(master,CHEASE_execution_script, PEST=0, fxrun = 0, id_string = 'Chease_', rm_files = ''):
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
    job_string = job_string + CHEASE_execution_script + '\n'
    if fxrun==1:
        job_string += 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/c/source/PGI/pgi/linux86/8.0-2/lib\n'

        job_string += '/u/reimerde/mars/MarsF20060714/FourierRF/FourierRF.x < fxin > fxrun_log.log\n'
        #job_string += 'fxrun > fxrun_log.log\n'

    job_string += 'rm '+ rm_files +'\n'
    file_name = open('chease.job','w')
    file_name.write(job_string)
    file_name.close()

def pickup_interp_points(R_probe, Z_probe, l_probe, t_probe, type_probe, Navg):
    if type_probe == 1:
        Rprobek=R_probe + l_probe/2.*num.cos(t_probe)*num.linspace(-1,1,num = Navg)
        Zprobek=Z_probe + l_probe/2.*num.sin(t_probe)*num.linspace(-1,1,num = Navg)
    else:
        Rprobek=R_probe + l_probe/2.*num.sin(t_probe)*num.linspace(-1,1,num = Navg)
        Zprobek=Z_probe - l_probe/2.*num.cos(t_probe)*num.linspace(-1,1,num = Navg)
    return Rprobek, Zprobek

def pickup_field_interpolation(r_array, z_array, Br, Bz, Bphi, R_tot_array, Z_tot_array):
    #must be linear interpolation, otherwise there are sometimes problems
    Brprobek = num.resize(scipy_griddata((r_array.flatten(),z_array.flatten()), Br.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear'),R_tot_array.shape)
    Bzprobek = num.resize(scipy_griddata((r_array.flatten(),z_array.flatten()), Bz.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear'),Z_tot_array.shape)
    
    return Brprobek, Bzprobek
    


def coil_responses6(r_array,z_array,Br,Bz,Bphi, probe, probe_type, Rprobe,Zprobe,tprobe,lprobe, Navg=1000, default=0):
    if default==1:
        probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
        # probe type 1: poloidal field, 2: radial field
        probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
         # Poloidal geometry
        Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
        Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
        tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
        lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

    Nprobe = len(probe)

    Bprobem = []; Rprobek_total = []; Zprobek_total = []

    start_time = time.time()
    for k in range(0, Nprobe):
        Rprobek, Zprobek = pickup_interp_points(Rprobe[k], Zprobe[k], lprobe[k], tprobe[k], probe_type[k], Navg)
        Rprobek_total.append(Rprobek)
        Zprobek_total.append(Zprobek)

    R_tot_array = num.array(Rprobek_total)
    Z_tot_array = num.array(Zprobek_total)

    Brprobek, Bzprobek = pickup_field_interpolation(r_array, z_array, Br, Bz, Bphi, R_tot_array, Z_tot_array)

    Bprobem = []
    
    #calculate normal to coil and average over data points
    for k in range(0, Nprobe):
        Bprobem.append(num.average((num.sin(tprobe[k])*num.real(Bzprobek[k,:]) + num.cos(tprobe[k])*num.real(Brprobek[k,:])) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek[k,:]) +num.cos(tprobe[k])*num.imag(Brprobek[k,:]))))
    print 'total time :', time.time() - start_time 
    return Bprobem



def coil_responses6_backup(r_array,z_array,Br,Bz,Bphi, probe, probe_type, Rprobe,Zprobe,tprobe,lprobe, Navg=1000, default=0):
    if default==1:
        probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
        # probe type 1: poloidal field, 2: radial field
        probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
         # Poloidal geometry
        Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
        Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
        tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
        lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

    Nprobe = len(probe)

    Bprobem = []; Rprobek_total = []; Zprobek_total = []

    start_time = time.time()
    for k in range(0, Nprobe):
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



def modify_pickle(old_base_dir, new_base_dir, old_pickle_name, new_pickle_name):
    #old_base_dir = '/u/haskeysr/mars/project1/'
    #new_base_dir = '/u/haskeysr/mars/'
    #old_pickle_name = '5_post_RMZM.pickle'
    #new_pickle_name = '5_project_vary_ROTEblah.pickle'

    dir_dict = pickle.load(open(old_base_dir + old_pickle_name))

    def edit_string(string, old_base_dir, new_base_dir):
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
