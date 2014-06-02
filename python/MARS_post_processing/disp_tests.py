
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_funcs
import copy
import pyMARS.results_class as results_class
import pickle,sys,copy
import numpy as np
import pyMARS.RZfuncs as RZfuncs

file_name = '/u/haskeysr/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_spitz/shot_142614_expt_scan_NC_const_eq_const_rot_prof_spitz_post_processing_PEST.pickle'
project_dict = pickle.load(file(file_name,'r'))
upper_and_lower = 1
s_surface_list = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
#out = results_class.disp_calcs(disp_run_list, n_zones = 20, phasing_vals = phasing_vals, ul = upper_and_lower)
#project_dict['sims'][i]['displacement_responses'] = copy.deepcopy(out)
#disp_run_list = [results_classes['lower_plasma'], results_classes['lower_vacuum'], results_classes['upper_plasma'], results_classes['upper_vacuum']]
phasing_vals = [0,45,90,135,180,225,270,315]
i = 10

#for i in project_dict['sims'].keys():
print 'working on serial : ', i
n = np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
q_range = [0, n+4]
I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
#project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
project_dict['sims'][i]['I0EXP'] = I0EXP
#I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])

project_dict_copy = copy.deepcopy(project_dict)
for i in project_dict['sims'].keys():
    locs = ['upper', 'lower'] if upper_and_lower else ['']
    project_dict['sims'][i]['responses']={}
    results_classes = {}
    for s_surface in s_surface_list:
        current_label = str(s_surface)
        project_dict['sims'][i]['responses'][current_label]={}
    facn = 1
    for loc in locs:
        for type1, type2 in zip(['plasma', 'vacuum'], ['total','vacuum']):
            directory = project_dict['sims'][i]['dir_dict']['mars_{}_{}_dir'.format(loc, type1)]
            curr_data = results_class.data(directory, I0EXP=I0EXP)
            #curr_data.get_PEST(facn = facn)
            #res_integral, res_discrete = curr_data.resonant_strength(n = n, SURFMN_coords=SURFMN_coords)
            #project_dict['sims'][i]['responses']['{}_resonant_response_{}'.format(type2, loc)] = copy.deepcopy(res_discrete)
            #project_dict['sims'][i]['responses']['{}_resonant_response_{}_integral'.format(type2, loc)] = copy.deepcopy(res_integral)
            #for tmp_type in ['mq','qn','sq']:  project_dict['sims'][i]['responses']['resonant_response_{}'.format(tmp_type)] = copy.deepcopy(getattr(curr_data, tmp_type))
            #for s_surface in s_surface_list:
            #    current_label = str(s_surface)
            #    print 'getting kink data'
            #    mk, ss, relevant_values, q_val = curr_data.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
            #    project_dict['sims'][i]['responses'][current_label]['{}_kink_response_{}'.format(type2, loc)] = copy.deepcopy(relevant_values)
            #    project_dict['sims'][i]['responses'][current_label]['mk'] = copy.deepcopy(mk)
            #    project_dict['sims'][i]['responses'][current_label]['ss'] = copy.deepcopy(ss)
            #    project_dict['sims'][i]['responses'][current_label]['q_val'] = copy.deepcopy(q_val)

            results_classes['{}_{}'.format(loc, type1)] = copy.deepcopy(curr_data)

    #disp_run_list = [lower_data_tot, lower_data_vac, upper_data_tot, upper_data_vac]
    if upper_and_lower:
        disp_run_list = [results_classes['lower_plasma'], results_classes['lower_vacuum'], results_classes['upper_plasma'], results_classes['upper_vacuum']]
        phasing_vals = [0,45,90,135,180,225,270,315]
    else:
        disp_run_list = [results_classes['_plasma'], results_classes['_vacuum']]
        phasing_vals = [0]
    for tmp_cur in disp_run_list: tmp_cur.get_VPLASMA()
    print 'got VPLASMA, calculating displacements'

    out = results_class.disp_calcs(disp_run_list, n_zones = 20, phasing_vals = phasing_vals, ul = upper_and_lower)
    project_dict['sims'][i]['displacement_responses'] = copy.deepcopy(out)

xpoint = dBres_dBkink.x_point_displacement_calcs(a, 0)
xpoint.plot_single_phasing(0, 'shot_time', field = 'plasma', multiplier = 1)

a = dBres_dBkink.post_processing_results(file_name, 0.92, 0, 0, fixed_harmonic = 3, reference_offset = [4,0], reference_dB_kink = 'plasma', sort_name = 'time_list', try_many_phasings = False)
