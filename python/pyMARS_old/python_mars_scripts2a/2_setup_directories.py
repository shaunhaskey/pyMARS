#!/usr/bin/env Python
## Run this script after you have created many EXPEQ files for the different equilibria.
## There has to be a stab_setup_results.dat file which is output by Matt Lanctot's scripts
## for running CORSICA to obtain the pressure and density scalings. The line items in the
## stab_setup_results.dat each describe a single equilibria including the name of the
## filename through qmult and pmult values.
## Set the values for the filters before you run! this will prohibit certain equilibria from
## making it past this step. Run as follows:
##
## ## bash $: ./2_setup_directories.py project_name
## bash $: nohup 2_setup_directories.py project_name > step2_log &
##
## All the EXPEQ equilibria and the stab results file need to be in the efit dir of this project
## where project_name is a label you already gave to this project - it is important for
## all the other scripts you run as it identifies where everything is and is the
## parent directory name for the 'project'.
## Note the section where the EXPEQ filename is generated for each serial number.
## *** YOU NEED TO SET THIS FORMAT TO THE SAME AS THE OUTPUT FROM CORSICA******
## Shaun Haskey Sept 28 2011

from PythonMARS_funcs import *
import pickle, time

project_name = sys.argv[1]
overall_start = time.time()

########## MUST SET THESE VALUES BEFORE RUNNING!!!!!!! #########
q95_range = [2, 7]
Bn_Div_Li_range = [0.75, 3]
#Bn_Div_Li_range = [0.0, 3]
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
#filters on the equilibria that are 'accepted'
filter_WTOTN1 = 1
filter_WTOTN2 = 1
filter_WTOTN3 = 1
filter_WWTOTN1 = 1
##############################################################


#Open the datastructure from step1 so it knows where to find all the files
pickle_file = open(project_dir + '1_'+project_name+'_initial_setup.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()


file_location = project_dict['details']['efit_dir']+'/stab_setup_results.dat'
base_dir = project_dict['details']['base_dir']


#Read the stab_results file and create the serial numbers in the data structure
#for each equilibria.
project_dict['sims'] = read_stab_results(file_location)


#Filter  only equilibria without relevant q95 and Bn_Div_Li and stability values

removed_q95 = 0;removed_Bn = 0;removed_DCON = 0;removed_read_error = 0

for i in project_dict['sims'].keys():
    #current_q95 = 3 #edit for benchmark test
    current_q95 = project_dict['sims'][i]['Q95']
    current_Bn_Div_Li = project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI']
    #current_Bn_Div_Li = 2 #edit for benchmark test

    #This is so that the script doesn't die on some strange error, must disable these if you are trying
    #to debug what is happening
    try:
        WTOTN1 = project_dict['sims'][i]['WTOTN1']
        WTOTN2 = project_dict['sims'][i]['WTOTN2']
        WTOTN3 = project_dict['sims'][i]['WTOTN3']
        WWTOTN1 = project_dict['sims'][i]['WWTOTN1']

        #project_dict['sims'][i]['QMULT']= 1 #TEMP EDIT!!! for benchmark
        #project_dict['sims'][i]['Q95']= 4 #TEMP EDIT!!! for benchmark
        
        #Filter for relevant q95
        if (current_q95<q95_range[0]) or (current_q95>q95_range[1]):
            del project_dict['sims'][i]
            print 'removed item q95 out of range'
            removed_q95 += 1

        #Filter for relevant Beta_n/Li
        elif (current_Bn_Div_Li<Bn_Div_Li_range[0]) or (current_Bn_Div_Li>Bn_Div_Li_range[1]):
            del project_dict['sims'][i]
            print 'removed item Bn_Div_Li out of range'
            removed_Bn += 1

        #Filter for DCON failure
        elif (WTOTN1<=0 and filter_WTOTN1==1) or  (WTOTN2<=0 and filter_WTOTN2==1)  or (WTOTN3<=0 and filter_WTOTN3==1) or (WWTOTN1<=0 and filter_WWTOTN1==1):
            del project_dict['sims'][i]
            print 'removed item due to stability'
            removed_DCON += 1
        else:
            pass
    except:
        del project_dict['sims'][i]
        print 'removed due to error reading'
        removed_read_error += 1


print 'Removed %d for q95, %d for Bn_Div_Li, %d for stability, and %d due to read error'%(removed_q95, removed_Bn, removed_DCON, removed_read_error)
print 'Remaining eq : %d'%(len(project_dict['sims'].keys()))
time.sleep(10) #So someone can read what happened

# Give each equilibria serial number some information (global info -> local info) - will improve this at some point
# Note change the structure of the EXPEQ file name to fit with your convention if its different

for i in project_dict['sims'].keys():
    print i
    project_dict['sims'][i]['shot']=project_dict['details']['shot']
    project_dict['sims'][i]['shot_time']=project_dict['details']['shot_time']
    project_dict['sims'][i]['thetac']=project_dict['details']['thetac']
    #project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%.3d_q%.3d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*100)),int(round(project_dict['sims'][i]['QMULT']*100)))
    project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%d_q%d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*1000)),int(round(project_dict['sims'][i]['QMULT']*1000)))
    #project_dict['sims'][i]['EXPEQ_name']='EXPEQ_%d.%.5d_p%03d'%(project_dict['sims'][i]['shot'], project_dict['sims'][i]['shot_time'],int(round(project_dict['sims'][i]['PMULT']*100))) #For Benchmark

    project_dict['sims'][i]['M1'] = project_dict['details']['M1']
    project_dict['sims'][i]['M2'] = project_dict['details']['M2']
    project_dict['sims'][i]['NTOR'] = project_dict['details']['NTOR']
    project_dict['sims'][i]['ICOIL_FREQ'] = project_dict['details']['ICOIL_FREQ']
    project_dict['sims'][i]['FEEDI'] = project_dict['details']['FEEDI']
    project_dict['sims'][i] = generate_directories(project_dict['sims'][i],base_dir)


#Dump the data structure so it can be read by the next step
print 'dumping data to pickle file'
#pickle_file = open(project_dict['details']['base_dir']+ '2_'+project_name+'_setup_directories.pickle','w')
pickle_file = open(project_dict['details']['base_dir']+ '2_'+project_name+'_setup_directories.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()

print 'Total Time for this step : %.2f'%((time.time()-overall_start)/60)

