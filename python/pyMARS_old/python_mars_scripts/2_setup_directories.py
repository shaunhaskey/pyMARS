#!/usr/bin/env Python
from PythonMARS_funcs import *
import Chease_Batch_Launcher as ch_launch
import pickle, time

project_name = sys.argv[1]
overall_start = time.time()

##########MUST SET THESE VALUES BEFORE RUNNING!!!!!!!#########
q95_range = [2, 7]
Bn_Div_Li_range = [0.75, 3]
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
##############################################################


pickle_file = open(project_dir + '1_'+project_name+'_initial_setup.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()


file_location = project_dict['details']['efit_dir']+'/stab_setup_results.dat'
base_dir = project_dict['details']['base_dir']

project_dict['sims'] = read_stab_results(file_location)


#Filter out so only relevant q95 and Bn_Div_Li remain
print 'Filtering out irrelevant q95 and Bn_Div_Li'
print 'Could include DCON'
removed_q95 = 0
removed_Bn = 0
removed_DCON = 0
removed_read_error = 0

for i in project_dict['sims'].keys():
    current_q95 = project_dict['sims'][i]['Q95']
    current_Bn_Div_Li = project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI']
    try:
        WTOTN1 = project_dict['sims'][i]['WTOTN1']
        WTOTN2 = project_dict['sims'][i]['WTOTN2']
        WTOTN3 = project_dict['sims'][i]['WTOTN3']
        WWTOTN1 = project_dict['sims'][i]['WWTOTN1']
        if (current_q95<q95_range[0]) or (current_q95>q95_range[1]):
            del project_dict['sims'][i]
            print 'removed item q95 out of range'
            removed_q95 += 1
        elif (current_Bn_Div_Li<Bn_Div_Li_range[0]) or (current_Bn_Div_Li>Bn_Div_Li_range[1]):
            del project_dict['sims'][i]
            print 'removed item Bn_Div_Li out of range'
            removed_Bn += 1
        elif (WTOTN1<=0) or  (WTOTN2<=0)  or (WTOTN3<=0) or (WWTOTN1<=0):
            del project_dict['sims'][i]
            print 'removed item due to stability'
            print WTOTN1, WTOTN2,WTOTN3, WWTOTN1
            removed_DCON += 1
        else:
            pass
    except:
        del project_dict['sims'][i]
        print 'removed due to error reading'
        removed_read_error += 1


print 'Removed %d for q95, %d for Bn_Div_Li, %d for stability, and %d due to read error'%(removed_q95, removed_Bn, removed_DCON, removed_read_error)
print 'Remaining eq : %d'%(len(project_dict['sims'].keys()))
time.sleep(3)

#Generate all the directories
for i in project_dict['sims'].keys():
    print i
    project_dict['sims'][i]['shot']=project_dict['details']['shot']
    project_dict['sims'][i]['shot_time']=project_dict['details']['shot_time']
    project_dict['sims'][i]['thetac']=project_dict['details']['thetac']
#    project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%.3d_q%.3d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*100)),int(round(project_dict['sims'][i]['QMULT']*100)))
    project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%d_q%d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*1000)),int(round(project_dict['sims'][i]['QMULT']*1000)))
    project_dict['sims'][i]['M1'] = project_dict['details']['M1']
    project_dict['sims'][i]['M2'] = project_dict['details']['M2']
    project_dict['sims'][i]['NTOR'] = project_dict['details']['NTOR']

    project_dict['sims'][i]['ICOIL_FREQ'] = project_dict['details']['ICOIL_FREQ']
    project_dict['sims'][i]['FEEDI'] = project_dict['details']['FEEDI']

    project_dict['sims'][i] = generate_directories(project_dict['sims'][i],base_dir)

print 'dumping data to pickle file'
pickle_file = open(project_dict['details']['base_dir']+ '2_'+project_name+'_setup_directories.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)

