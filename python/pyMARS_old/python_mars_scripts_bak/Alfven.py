import numpy as num
import os
import time
import re
import pickle
#constants

def mars_setup_alfven(master, vac):
    mu0 = 4e-7 * num.pi
    mi = 1.6726e-27 + 1.6749e-27
    e = 1.60217e-19

    if vac ==1 :
        os.chdir(master['dir_dict']['mars_vac_dir'])
    else:
        os.chdir(master['dir_dict']['mars_plasma_dir'])

        
    PROFDEN_file = open('PROFDEN','r')
    PROFDEN_data = PROFDEN_file.readlines()
    print PROFDEN_data[1]

    pattern = ''
    PROFDEN_data[1]
    re.search(pattern, PROFDEN_data[1])
    pattern = '\d+.\d+e*\+*\-*\d*'#e*E*+*-*[0-9]*'
    string1 = re.search(pattern, PROFDEN_data[1])
    string2 = re.search(pattern, PROFDEN_data[1][string1.end()+1:])
    ne0_r = float(PROFDEN_data[1][string1.start():string1.end()])
    ne0 = float(PROFDEN_data[1][string2.start()+string1.end()+1:string2.end()+string1.end()+1])
    print '|'+str(ne0_r)+'|'
    print '|'+str(ne0)+'|'

    B0EXP = master['B0EXP']
    R0EXP = master['R0EXP']

    v0a     = B0EXP/num.sqrt(mu0*mi*ne0)
    taua    = R0EXP/v0a
    tauwp   = 0.01405       # mu0*h*d/eta from /u/reimerde/mars/shot127838/README
    tauwm   = tauwp /taua


    f_v0a = v0a/R0EXP       # alfven frequency (1/s)
    fcio  = (e*R0EXP*num.sqrt(mu0*ne0))/num.sqrt(mi)

    ichz = num.array([1,2,5,10,20,40,60,100,120,160,200,500,1000,5000,10000.],dtype=float)
    ichz = num.array([10],dtype = float)#num.array([master['OMEGA']])

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

    master['OMEGA_Norm'] = omega[0]
    master['tauwm'] = tauwm
    master['v0a'] = v0a
    return master
pickle_file = open('/u/haskeysr/mars/test2/2_setup_directories.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

for i in project_dict['sims'].keys():
    project_dict['sims'][i] = mars_setup_alfven(project_dict['sims'][i], vac = 1)
