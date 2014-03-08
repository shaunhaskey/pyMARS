import results_class, copy
import RZfuncs
import numpy as np
import matplotlib.pyplot as pt
import PythonMARS_funcs as pyMARS
from scipy.interpolate import griddata
import pickle
import matplotlib.cm as cm
import time as time_module
import pyMARS.dBres_dBkink_funcs as dBres_dBkink


file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan_n2_lower_BetaN/detailed_q95_scan_n2_lower_BetaN_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/146382_thetac_003/146382_thetac_003_post_processing_PEST.pickle'
N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
#facn = 1.0; 
s_surface = 0.92 #0.97
ylim = [0,1.4]
phasing = 0
fixed_harmonic = 3
phase_machine_ntor = 0
#this is for dBkink calculation for selecting the relevant m to choose from
#(n+reference_offset[1])q+reference_offset[0] < m
reference_offset = [2,0]
#reference to calculate the relevant m
reference_dB_kink = 'plas'
upper_lower = False
project_dict = pickle.load(file(file_name,'r'))
key_list = project_dict['sims'].keys()

n = np.abs(project_dict['details']['MARS_settings']['<<RNTOR>>'])
q95_list, Bn_Li_list, time_list = dBres_dBkink.extract_q95_Bn(project_dict, bn_li = 1)

#res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower = dBres_dBkink.extract_dB_res(project_dict)





fig, ax = pt.subplots();

for i in [3,6,10]:
    percent = (1. - i/1000.)*100.
    file_name = '/home/srh112/NAMP_datafiles/mars/146382_thetac_0{:02d}/146382_thetac_0{:02d}_post_processing_PEST.pickle'.format(i,i)
    project_dict = pickle.load(file(file_name,'r'))
    print file_name
    ax.plot(np.abs(project_dict['sims'][1]['responses']['0.92']['total_kink_response_single']), label='Plasma + Vacuum {:.2f} %'.format(percent))
ax.plot(np.abs(project_dict['sims'][1]['responses']['0.92']['vacuum_kink_response_single']), label='Vacuum')
ax.set_title('Poloidal harmonics at s = 0.92')
ax.set_xlabel('m')
ax.set_ylabel('Harmonic amplitude (a.u)')
ax.legend()
fig.canvas.draw(); fig.show()
1/0



amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close = dBres_dBkink.extract_dB_kink(project_dict, s_surface, upper_lower = upper_lower)
fig_harm_select, ax_harm_select = pt.subplots()
ax_harm_select.plot(q95_list, np.array(q_val_list)*(n+reference_offset[1])+reference_offset[0], label='(n+%d)q+%d'%(reference_offset[1],reference_offset[0]))
ax_harm_select.plot(q95_list, np.array(q_val_list)*n, label='m=nq')
#Create the fixed phasing cases (as set by phasing)
amps_vac_comp = dBres_dBkink.apply_phasing(amps_vac_comp_upper, amps_vac_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
amps_plas_comp = dBres_dBkink.apply_phasing(amps_plas_comp_upper, amps_plas_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
amps_tot_comp = dBres_dBkink.apply_phasing(amps_tot_comp_upper, amps_tot_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)


#Get the reference which we use to find the maximum harmonic for dBkink
if reference_dB_kink=='plas':
    reference = dBres_dBkink.get_reference(amps_plas_comp_upper, amps_plas_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)
elif reference_dB_kink=='tot':
    reference = dBres_dBkink.get_reference(amps_tot_comp_upper, amps_tot_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)

#Note the returned values are simply a 1D array containing the complex amplitude of the max harmonic
#Do it for the single cases
plot_quantity_vac, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_vac_comp, reference_offset = reference_offset)
ax_harm_select.plot(q95_list,max_loc_list, label='max_harmonic')
plot_quantity_plas, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_plas_comp, reference_offset = reference_offset)
plot_quantity_tot, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_tot_comp, reference_offset = reference_offset)

#Do it for the upper/lower cases
upper_values_plasma, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_plas_comp_upper, reference_offset = reference_offset)
lower_values_plasma, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_plas_comp_lower, reference_offset = reference_offset)
upper_values_tot, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_tot_comp_upper, reference_offset = reference_offset)
lower_values_tot, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_tot_comp_lower, reference_offset = reference_offset)

upper_values_vac, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_vac_comp_upper, reference_offset = reference_offset)
lower_values_vac, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, amps_vac_comp_lower, reference_offset = reference_offset)
ax_harm_select.legend(loc='best')
ax_harm_select.set_xlabel('q95')
ax_harm_select.set_ylabel('m')
ax_harm_select.set_title('%s used to select m'%(reference_dB_kink))

ax_harm_select.set_ylim([0,np.max(q_val_list)*n+5])
fig_harm_select.canvas.draw(); fig_harm_select.show()
#Calculate fixed harmonic dBkink based only on vacuum fields, again upper_values.... are 1D array containing the complex amplitude of fixed harmonic
upper_values_vac_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_vac_comp_upper, fixed_harmonic)
lower_values_vac_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_vac_comp_lower, fixed_harmonic)
upper_values_plas_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_plas_comp_upper, fixed_harmonic)
lower_values_plas_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, amps_plas_comp_lower, fixed_harmonic)

