#!/usr/bin/env Python
'''
This does the post processing of the MARS-F runs to get dBkink and dBres information
SH: 26Feb2013
'''

import pyMARS.results_class as results_class
import pickle,sys,copy
import numpy as np
import pyMARS.RZfuncs as RZfuncs

project_name = sys.argv[1]
upper_and_lower = int(sys.argv[2])

def kink_resonant_response(project_dict, upper_and_lower=0, facn = 1.0, s_surface_list = [0.92], SURFMN_coords = 1):
    link_RMZM = 0
    disp_calcs = 1
    for i in project_dict['sims'].keys():
        print 'working on serial : ', i
        n = np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
        q_range = [0, n+4]
        I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
        #project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
        Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
        project_dict['sims'][i]['I0EXP'] = I0EXP
        #I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])

        if upper_and_lower == 1:
            #create the data classes for all 4 cases
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

            if disp_calcs:
                disp_run_list = [lower_data_tot, lower_data_vac, upper_data_tot, upper_data_vac]
                for tmp_cur in disp_run_list: tmp_cur.get_VPLASMA()
                print 'got VPLASMA, calculating displacements'
                out = results_class.disp_calcs(disp_run_list, n_zones = 20, phasing_vals = [0,45,90,135,180,225,270,315], ul = upper_and_lower)
                project_dict['sims'][i]['displacement_responses'] = copy.deepcopy(out)

            print 'getting resonant_strength data'
            upper_vac_res_integral, upper_vac_res_discrete = upper_data_vac.resonant_strength(n = n, SURFMN_coords=SURFMN_coords)
            lower_vac_res_integral, lower_vac_res_discrete = lower_data_vac.resonant_strength(n = n, SURFMN_coords=SURFMN_coords)
            upper_tot_res_integral, upper_tot_res_discrete = upper_data_tot.resonant_strength(n = n, SURFMN_coords=SURFMN_coords)
            lower_tot_res_integral, lower_tot_res_discrete = lower_data_tot.resonant_strength(n = n, SURFMN_coords=SURFMN_coords)

            project_dict['sims'][i]['responses']={}

            #record the vacuum results (s_surface independent)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_upper'] = copy.deepcopy(upper_vac_res_discrete)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_lower'] = copy.deepcopy(lower_vac_res_discrete)
            project_dict['sims'][i]['responses']['total_resonant_response_upper'] = copy.deepcopy(upper_tot_res_discrete)
            project_dict['sims'][i]['responses']['total_resonant_response_lower'] = copy.deepcopy(lower_tot_res_discrete)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_upper_integral'] = copy.deepcopy(upper_vac_res_integral)
            project_dict['sims'][i]['responses']['vacuum_resonant_response_lower_integral'] = copy.deepcopy(lower_vac_res_integral)
            project_dict['sims'][i]['responses']['total_resonant_response_upper_integral'] = copy.deepcopy(upper_tot_res_integral)
            project_dict['sims'][i]['responses']['total_resonant_response_lower_integral'] = copy.deepcopy(lower_tot_res_integral)

            #mq, qn, sq are m values, q values and s values for the resonant surfaces
            project_dict['sims'][i]['responses']['resonant_response_mq'] = copy.deepcopy(upper_data_tot.mq)
            project_dict['sims'][i]['responses']['resonant_response_qn'] = copy.deepcopy(upper_data_tot.qn)
            project_dict['sims'][i]['responses']['resonant_response_sq'] = copy.deepcopy(upper_data_tot.sq)


            #start doing the kink amplitude calculations - this is s_surface dependent
            #do it for all items in s_surface_list
            for s_surface in s_surface_list:
                current_label = str(s_surface)
                project_dict['sims'][i]['responses'][current_label]={}
                print 'getting kink data'
                mk_upper, ss_upper, relevant_values_upper_tot, q_val_upper_tot = upper_data_tot.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
                mk_lower, ss_lower, relevant_values_lower_tot, q_val_lower_tot = lower_data_tot.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
                mk_upper, ss_upper, relevant_values_upper_vac, q_val_upper_vac = upper_data_vac.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
                mk_lower, ss_lower, relevant_values_lower_vac, q_val_lower_vac = lower_data_vac.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)

                #record the kink results
                project_dict['sims'][i]['responses'][current_label]['vacuum_kink_response_upper'] = copy.deepcopy(relevant_values_upper_vac)
                project_dict['sims'][i]['responses'][current_label]['vacuum_kink_response_lower'] = copy.deepcopy(relevant_values_lower_vac)
                project_dict['sims'][i]['responses'][current_label]['total_kink_response_upper'] = copy.deepcopy(relevant_values_upper_tot)
                project_dict['sims'][i]['responses'][current_label]['total_kink_response_lower'] = copy.deepcopy(relevant_values_lower_tot)
                
                #mk are the m values associated with the amplitudes, ss_upper is the surface, q_val_upper_tot is the 
                #q value at the surface of interest
                project_dict['sims'][i]['responses'][current_label]['mk'] = copy.deepcopy(mk_upper)
                project_dict['sims'][i]['responses'][current_label]['ss'] = copy.deepcopy(ss_upper)
                project_dict['sims'][i]['responses'][current_label]['q_val'] = copy.deepcopy(q_val_upper_tot)

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
            a, single_vac_res = single_data_vac.resonant_strength(n=n, SURFMN_coords=SURFMN_coords)
            a, single_tot_res = single_data_tot.resonant_strength(n=n, SURFMN_coords=SURFMN_coords)
            
            #record the dBres information
            project_dict['sims'][i]['responses']['vacuum_resonant_response_single'] = copy.deepcopy(single_vac_res)
            project_dict['sims'][i]['responses']['total_resonant_response_single'] = copy.deepcopy(single_tot_res)

            #details of the surfaces where the resonance happens
            project_dict['sims'][i]['responses']['resonant_response_mq'] = copy.deepcopy(single_data_tot.mq)
            project_dict['sims'][i]['responses']['resonant_response_qn'] = copy.deepcopy(single_data_tot.qn)
            project_dict['sims'][i]['responses']['resonant_response_sq'] = copy.deepcopy(sing_data_tot.sq)

            #Record the dBkink information
            for s_surface in s_surface_list:
                current_label = str(s_surface)
                project_dict['sims'][i]['responses'][current_label]={}

                print 'getting kink data'
                mk_upper, ss_upper, relevant_values_single_tot, q_val = single_data_tot.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
                mk_upper, ss_upper, relevant_values_single_vac, q_val = single_data_vac.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)

                #record the kink results
                project_dict['sims'][i]['responses'][current_label]['vacuum_kink_response_single'] = copy.deepcopy(relevant_values_single_vac)
                project_dict['sims'][i]['responses'][current_label]['total_kink_response_single'] = copy.deepcopy(relevant_values_single_tot)
                project_dict['sims'][i]['responses'][current_label]['mk'] = copy.deepcopy(mk_upper)
                project_dict['sims'][i]['responses'][current_label]['ss'] = copy.deepcopy(ss_upper)
                project_dict['sims'][i]['responses'][current_label]['q_val'] = copy.deepcopy(q_val)
                #project_dict['sims'][i]['responses'][current_label]['q_val'] = q_val
    return project_dict

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()
s_surface_list = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
project_dict = kink_resonant_response(project_dict, upper_and_lower = upper_and_lower, s_surface_list = s_surface_list)
output_name = project_name + 'output'
pickle_file = open(output_name,'w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()
