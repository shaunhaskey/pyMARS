#!/usr/bin/env Python
import results_class
import pickle,sys
import numpy as num
import PythonMARS_funcs as pyMARS
import RZfuncs as RZfuncs

project_name = sys.argv[1]
upper_and_lower = int(sys.argv[2])


def kink_resonant_response(project_dict, upper_and_lower=0, facn = 1.0, psi = 0.92, q_range = [2,6]):
    link_RMZM = 0
    for i in project_dict['sims'].keys():
        project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
        Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
        I0EXP = project_dict['sims'][i]['I0EXP']
        print 'working on serial : ', i
        n = num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
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

            print 'getting kink data'
            mk_upper, ss_upper, relevant_values_upper_tot = upper_data_tot.kink_amp(psi, q_range, n = n)
            mk_lower, ss_lower, relevant_values_lower_tot = lower_data_tot.kink_amp(psi, q_range, n = n)
            mk_upper, ss_upper, relevant_values_upper_vac = upper_data_vac.kink_amp(psi, q_range, n = n)
            mk_lower, ss_lower, relevant_values_lower_vac = lower_data_vac.kink_amp(psi, q_range, n = n)

            #record the kink results
            project_dict['sims'][i]['vacuum_kink_response_upper'] = relevant_values_upper_vac
            project_dict['sims'][i]['vacuum_kink_response_lower'] = relevant_values_lower_vac
            project_dict['sims'][i]['total_kink_response_upper'] = relevant_values_upper_tot
            project_dict['sims'][i]['total_kink_response_lower'] = relevant_values_lower_tot
            project_dict['sims'][i]['kink_response_psi'] = psi

            print 'getting resonant_strength data'
            a, upper_vac_res = upper_data_vac.resonant_strength()
            a, lower_vac_res = lower_data_vac.resonant_strength()
            a, upper_tot_res = upper_data_tot.resonant_strength()
            a, lower_tot_res = lower_data_tot.resonant_strength()
            
            #record the kink results
            project_dict['sims'][i]['vacuum_resonant_response_upper'] = upper_vac_res
            project_dict['sims'][i]['vacuum_resonant_response_lower'] = lower_vac_res
            project_dict['sims'][i]['total_resonant_response_upper'] = upper_tot_res
            project_dict['sims'][i]['total_resonant_response_lower'] = lower_tot_res
            project_dict['sims'][i]['resonant_response_mq'] = upper_data_tot.mq
            project_dict['sims'][i]['resonant_response_qn'] = upper_data_tot.qn

        else:
            #print 'hello2'
            directory = project_dict['sims'][i]['dir_dict']['mars_vac_dir']
            single_data_vac = results_class.data(directory, I0EXP=I0EXP)
            directory = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
            single_data_tot = results_class.data(directory, I0EXP=I0EXP)

            print 'getting single PEST data'
            single_data_tot.get_PEST(facn = facn)
            single_data_vac.get_PEST(facn = facn)

            print 'getting kink data'
            mk_upper, ss_upper, relevant_values_single_tot = single_data_tot.kink_amp(psi, q_range, n = n)
            mk_upper, ss_upper, relevant_values_single_vac = single_data_vac.kink_amp(psi, q_range, n = n)

            #record the kink results
            project_dict['sims'][i]['vacuum_kink_response_single'] = relevant_values_single_vac
            project_dict['sims'][i]['total_kink_response_single'] = relevant_values_single_tot
            project_dict['sims'][i]['kink_response_psi'] = psi

            print 'getting resonant_strength data'
            a, single_vac_res = single_data_vac.resonant_strength()
            a, single_tot_res = single_data_tot.resonant_strength()
            
            #record the kink results
            project_dict['sims'][i]['vacuum_resonant_response_single'] = single_vac_res
            project_dict['sims'][i]['total_resonant_response_single'] = single_tot_res
            project_dict['sims'][i]['resonant_response_mq'] = upper_data_tot.mq
            project_dict['sims'][i]['resonant_response_qn'] = upper_data_tot.qn

    return project_dict

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

project_dict = kink_resonant_response(project_dict, upper_and_lower = upper_and_lower)

output_name = project_name + 'output'
pickle_file = open(output_name,'w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()
