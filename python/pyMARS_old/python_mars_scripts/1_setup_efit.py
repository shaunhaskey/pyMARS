#!/usr/bin/env Python
import time, os, sys, pickle
project_name = sys.argv[1]

overall_start = time.time()
os.system('mkdir /scratch/haskeysr/mars/'+ project_name)
project_dict={}
project_dict['details']={}
project_dict['details']['base_dir'] = '/scratch/haskeysr/mars/'+ project_name +'/' #this directory must already exist
print project_dict['details']['base_dir']


project_dict['details']['template_dir'] = '/u/haskeysr/mars/templates/'
project_dict['details']['efit_master'] = '/u/haskeysr/mars/eq_from_matt/efit_files/'
project_dict['details']['thetac'] = 0.003
project_dict['details']['shot'] = 138344
project_dict['details']['M1'] = -29
project_dict['details']['M2'] = 29
project_dict['details']['shot_time'] = 2306
project_dict['details']['NTOR'] = 2
project_dict['details']['FEEDI']= '(1.0,0.0),(-0.5, 0.86603),'
project_dict['details']['ICOIL_FREQ'] = 20

#grid resolutions - NPSI, NCHI, NV, REXT, 


project_dict['sims']={}

def generate_master_dir(master):
    master['shot_dir'] =  project_dict['details']['base_dir'] + 'shot' + str(master['shot'])+'/'
    os.system('mkdir ' + master['shot_dir'])
    master['thetac_dir'] = master['shot_dir'] + 'tc_%03d/'%(master['thetac']*1000)
    os.system('mkdir ' + master['thetac_dir'])
    master['efit_dir'] =  master['thetac_dir']  + 'efit/'
    os.system('mkdir ' + master['efit_dir'])
    return master

#Generate the master directories
project_dict['details'] = generate_master_dir(project_dict['details'])
os.system('cp ' + project_dict['details']['efit_master'] + '* ' + project_dict['details']['efit_dir'])

pickle.dump(project_dict,open(project_dict['details']['base_dir']+'1_'+project_name+'_initial_setup.pickle','w'))
#Ready to run Caltrans now


print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)
