#!/usr/bin/env Python
import results_class
import pickle,sys,copy
import numpy as num
import PythonMARS_funcs as pyMARS
import RZfuncs as RZfuncs

project_name = sys.argv[1]
upper_and_lower = int(sys.argv[2])

def kink_resonant_response(project_dict, upper_and_lower=0, facn = 1.0, psi_list = [0.92], q_range = [2,6]):
    link_RMZM = 0
    for i in project_dict['sims'].keys():
        project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
        Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
        I0EXP = project_dict['sims'][i]['I0EXP']

        print 'working on serial : ', i
        n = num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
        I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])

        if upper_and_lower == 1:
            directory = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
            upper_data_tot = results_class.data(directory, I0EXP=I0EXP)
            directory = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']
            lower_data_tot = results_class.data(directory, I0EXP=I0EXP)
            directory = project_dict['sims'][i]['dir_dict']['mars_upper_vacuum_dir']
            upper_data_vac = results_class.data(directory, I0EXP=I0EXP)
            directory = project_dict['sims'][i]['dir_dict']['mars_lower_vacuum_dir']
            lower_data_vac = results_class.data(directory, I0EXP=I0EXP)

            print 'getting PEST data'
            upper_data_tot.get_PEST(facn = facn)
            lower_data_tot.get_PEST(facn = facn)
            upper_data_vac.get_PEST(facn = facn)
            lower_data_vac.get_PEST(facn = facn)

            print 'getting resonant_strength data'
            upper_vac_res_integral, upper_vac_res_discrete = upper_data_vac.resonant_strength(n = n)
            lower_vac_res_integral, lower_vac_res_discrete = lower_data_vac.resonant_strength(n = n)
            upper_tot_res_integral, upper_tot_res_discrete = upper_data_tot.resonant_strength(n = n)
            lower_tot_res_integral, lower_tot_res_discrete = lower_data_tot.resonant_strength(n = n)

            project_dict['sims'][i]['responses']={}

            #record the vacuum results (psi independent)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_upper'] = copy.deepcopy(upper_vac_res_discrete)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_lower'] = copy.deepcopy(lower_vac_res_discrete)
            project_dict['sims'][i]['responses']['total_resonant_response_upper'] = copy.deepcopy(upper_tot_res_discrete)
            project_dict['sims'][i]['responses']['total_resonant_response_lower'] = copy.deepcopy(lower_tot_res_discrete)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_upper_integral'] = copy.deepcopy(upper_vac_res_integral)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_lower_integral'] = copy.deepcopy(lower_vac_res_integral)
            project_dict['sims'][i]['responses']['total_resonant_response_upper_integral'] = copy.deepcopy(upper_tot_res_integral)
            project_dict['sims'][i]['responses']['total_resonant_response_lower_integral'] = copy.deepcopy(lower_tot_res_integral)

            project_dict['sims'][i]['responses']['resonant_response_mq'] = copy.deepcopy(upper_data_tot.mq)
            project_dict['sims'][i]['responses']['resonant_response_qn'] = copy.deepcopy(upper_data_tot.qn)
            project_dict['sims'][i]['responses']['resonant_response_sq'] = copy.deepcopy(upper_data_tot.sq)

            for psi in psi_list:
                current_label = str(psi)
                project_dict['sims'][i]['responses'][current_label]={}
                print 'getting kink data'
                mk_upper, ss_upper, relevant_values_upper_tot = upper_data_tot.kink_amp(psi, q_range, n = n)
                mk_lower, ss_lower, relevant_values_lower_tot = lower_data_tot.kink_amp(psi, q_range, n = n)
                mk_upper, ss_upper, relevant_values_upper_vac = upper_data_vac.kink_amp(psi, q_range, n = n)
                mk_lower, ss_lower, relevant_values_lower_vac = lower_data_vac.kink_amp(psi, q_range, n = n)

                #record the kink results
                project_dict['sims'][i]['responses'][current_label]['vacuum_kink_response_upper'] = copy.deepcopy(relevant_values_upper_vac)
                project_dict['sims'][i]['responses'][current_label]['vacuum_kink_response_lower'] = copy.deepcopy(relevant_values_lower_vac)
                project_dict['sims'][i]['responses'][current_label]['total_kink_response_upper'] = copy.deepcopy(relevant_values_upper_tot)
                project_dict['sims'][i]['responses'][current_label]['total_kink_response_lower'] = copy.deepcopy(relevant_values_lower_tot)
                project_dict['sims'][i]['responses'][current_label]['mk'] = copy.deepcopy(mk_upper)
                project_dict['sims'][i]['responses'][current_label]['ss'] = copy.deepcopy(ss_upper)

        else:
            #print 'hello2'
            directory = project_dict['sims'][i]['dir_dict']['mars_vac_dir']
            single_data_vac = results_class.data(directory, I0EXP=I0EXP)
            directory = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
            single_data_tot = results_class.data(directory, I0EXP=I0EXP)

            print 'getting single PEST data'
            single_data_tot.get_PEST(facn = facn)
            single_data_vac.get_PEST(facn = facn)

            project_dict['sims'][i]['responses']={}
            print 'getting resonant_strength data'
            a, single_vac_res = single_data_vac.resonant_strength()
            a, single_tot_res = single_data_tot.resonant_strength()
            
            #record the kink results
            project_dict['sims'][i]['responses']['vacuum_resonant_response_single'] = copy.deepcopy(single_vac_res)
            project_dict['sims'][i]['responses']['total_resonant_response_single'] = copy.deepcopy(single_tot_res)
            project_dict['sims'][i]['responses']['resonant_response_mq'] = copy.deepcopy(upper_data_tot.mq)
            project_dict['sims'][i]['responses']['resonant_response_qn'] = copy.deepcopy(upper_data_tot.qn)

            for psi in psi_list:
                current_label = str(psi)
                project_dict['sims'][i]['responses'][current_label]={}

                print 'getting kink data'
                mk_upper, ss_upper, relevant_values_single_tot = single_data_tot.kink_amp(psi, q_range, n = n)
                mk_upper, ss_upper, relevant_values_single_vac = single_data_vac.kink_amp(psi, q_range, n = n)

                #record the kink results
                project_dict['sims'][i]['responses'][current_label]['vacuum_kink_response_single'] = copy.deepcopy(relevant_values_single_vac)
                project_dict['sims'][i]['responses'][current_label]['total_kink_response_single'] = copy.deepcopy(relevant_values_single_tot)

    return project_dict

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()
psi_list = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
project_dict = kink_resonant_response(project_dict, upper_and_lower = upper_and_lower, psi_list = psi_list)
output_name = project_name + 'output'
pickle_file = open(output_name,'w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()
