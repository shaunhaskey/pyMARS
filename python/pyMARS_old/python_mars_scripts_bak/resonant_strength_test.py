import pickle
import numpy as num
import os
from PythonMARS_funcs import *
import RZfuncs as post_func
from results_class import *
PEST = 1
def find_serial(project_dict, Bn_Div_Li_target,q95_target):
    passes = 0
    fails = 0
    overall_distance = 1000.
    for jjj in project_dict['sims'].keys():
        try:
            current_Bn_Div_Li = project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI']
            current_q95 = project_dict['sims'][jjj]['Q95']
            distance = (current_Bn_Div_Li-Bn_Div_Li_target)**2+(current_q95-q95_target)**2
            if distance < overall_distance:
                overall_distance = distance *1.
                current_serial = jjj*1
            passes+=1
        except:
            fails+=1
            print 'failed'
    return current_serial


def change_phasings(combined_data,phasing_list,min_s,power):
    #combined_data.comb_p = copy.deepcopy(combined_data.data_p_l)
    combined_data.comb_v = copy.deepcopy(combined_data.data_v_l)
    #combined_data.comb_p_only = copy.deepcopy(combined_data.data_p_l)
    answer_list = []
    for phasing in phasing_list:
        #combined_data.comb_p.R,combined_data.comb_p.Z, combined_data.comb_p.B1, combined_data.comb_p.B2, combined_data.comb_p.B3, combined_data.comb_p.Bn, combined_data.comb_p.BMn,combined_data.comb_p.BnPEST = combine_data(combined_data.data_p_u, combined_data.data_p_l, phasing)
        combined_data.comb_v.R,combined_data.comb_v.Z, combined_data.comb_v.B1, combined_data.comb_v.B2, combined_data.comb_v.B3, combined_data.comb_v.Bn, combined_data.comb_v.BMn,combined_data.comb_v.BnPEST = combine_data(combined_data.data_v_u, combined_data.data_v_l, phasing)
        #print 'getting pest'
        #temp_storage = copy.deepcopy(combined_data.comb_p.BnPEST)
        #combined_data.comb_p.get_PEST()
        #combined_data.comb_v.get_PEST()
        #print '*************check new get_PEST routine'
        #print temp_storage.shape, combined_data.comb_p.BnPEST.shape
        #print num.sum(num.abs(temp_storage - combined_data.comb_p.BnPEST))
        answer = combined_data.comb_v.resonant_strength(min_s=min_s,power=power)
        answer_list.append((phasing,answer))
    print answer_list
    return answer_list


def get_combined_data(project_dict, current_serial):
    mars_dir = project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']
    mars_base_dir = project_dict['sims'][current_serial]['dir_dict']['mars_dir']
    dir_base = mars_base_dir

    print current_serial
    dir1_v_u = dir_base + 'RUNrfa_COILupper.vac/'
    dir1_v_l = dir_base + 'RUNrfa_COILlower.vac/'
    dir1_p_u = dir_base + 'RUNrfa_COILupper.p/'
    dir1_p_l = dir_base + 'RUNrfa_COILlower.p/'
    combined_data = results_combine(dir1_v_u, dir1_v_l, dir1_p_u, dir1_p_l)
    #print 'getting plasma data'
    #combined_data.get_plasma_data_upper()
    #combined_data.get_plasma_data_lower()
    print 'getting vac data'
    combined_data.get_vac_data_upper()
    combined_data.get_vac_data_lower()
    print 'get PEST'
    #combined_data.data_p_l.get_PEST()
    #combined_data.data_p_u.get_PEST()
    combined_data.data_v_l.get_PEST()
    combined_data.data_v_u.get_PEST()

    print 'combining data'
    return combined_data

ROTE_value = 0
#q95_target = 3.5
project_name = 'project1_new_eq'
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'
#phasing_list = [-300, -240, -180, -120, -60,0]
#phasing_list = [0]
phasing_list = range(0,-360,-10)
#phasing_list = [0, -60]
name = project_dir + '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
project_dict = pickle.load(open(name))
#Bn_values = num.arange(1,3,0.5)#1,3,0.1)
Bn_values = [1.5]
q95_values = num.arange(2,7,0.2)
#Bn_target = 2.8
#for i in Bn_values:

min_s1=0.8
min_s2=0.9
min_s3=0.95
min_s4 = 0

answer_list_total = {}
answer_list_total1 = {}
answer_list_total2 = {}
answer_list_total3 = {}
answer_list_total4 = {}

power = 2
for Bn_target in Bn_values:
    for q95_target in q95_values:
        print Bn_target, q95_target
        current_serial = find_serial(project_dict, Bn_target, q95_target)
        combined_data = get_combined_data(project_dict, current_serial)
        answer_list_total1[q95_target] = change_phasings(combined_data,phasing_list,min_s1,power)
        answer_list_total2[q95_target] = change_phasings(combined_data,phasing_list,min_s2,power)
        answer_list_total3[q95_target] = change_phasings(combined_data,phasing_list,min_s3,power)
        answer_list_total4[q95_target] = change_phasings(combined_data,phasing_list,min_s4,power)

pickle_file = open(project_dir+'resonant_phasing_0-8_s2.pickle','w')
pickle.dump(answer_list_total1,pickle_file)
pickle_file.close()
pickle_file = open(project_dir+'resonant_phasing_0-9_s2.pickle','w')
pickle.dump(answer_list_total2,pickle_file)
pickle_file.close()
pickle_file = open(project_dir+'resonant_phasing_0-95_s2.pickle','w')
pickle.dump(answer_list_total3,pickle_file)
pickle_file.close()
pickle_file = open(project_dir+'resonant_phasing_0_s2.pickle','w')
pickle.dump(answer_list_total4,pickle_file)
pickle_file.close()


'''
total = len(project_dict['sims'].keys())
finished = 0

for current_key in project_dict['sims'].keys():
    combined_data = get_combined_data(project_dict, current_key)
    project_dict['sims'][current_key]['resonant'] = change_phasings(combined_data,phasing_list)
    finished+=1
    print 'finished %d of %d'%(finished,total)

pickle_file = open(project_dir+'test_output.pickle','w')
pickle.dump(project_dict,pickle_file)
pickle_file.close()
'''

