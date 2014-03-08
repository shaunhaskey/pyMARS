'''
Generates plots of 'kink amplification' as a function of phasing
Will also create the files for an animation of plasma, vac, and total 
components in PEST co-ordinates

SH 29/12/2012 Started to make this code more modular with functions
'''

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

#file_name = '/home/srh112/NAMP_datafiles/mars/shot146382_scan/shot146382_scan_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot146394_3000_q95/shot146394_3000_q95_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/q95_scan/q95_scan_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/detailed_q95_scan3_n4/detailed_q95_scan3_n4_post_processing_PEST.pickle'
#file_name2 = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan3_n4/detailed_q95_scan3_n4_post_processing_PEST.pickle'
file_name2 = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan_n4_lower_BetaN/detailed_q95_scan_n4_lower_BetaN_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan_n2_lower_BetaN/detailed_q95_scan_n2_lower_BetaN_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan_n2_146382/detailed_q95_scan_n2_146382_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan_n2_146382_NVEXP_4/detailed_q95_scan_n2_146382_NVEXP_4_post_processing_PEST.pickle'

#file_name = '/u/haskeysr/mars/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'

N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
#facn = 1.0; 
s_surface = 0.92 #0.97
ylim = [0,1.4]

#phasing_range = [-180.,180.]
#phasing_range = [0.,360.]

#For dBkink based on vacuum alone - picked nq+fixed_harmonic
fixed_harmonic = 3
phase_machine_ntor = 0
#this is for dBkink calculation for selecting the relevant m to choose from
#(n+reference_offset[1])q+reference_offset[0] < m
reference_offset = [2,0]
#reference to calculate the relevant m
reference_dB_kink = 'plas'
make_animations = 0
include_discrete_comparison = 0
seperate_res_plot = 0
include_vert_lines = 0
plot_text = 1

#Plot selections
various_line_plots = 0
#dB_kink and the vacuum harmonic strength of the same m.. amplitude and phase
dB_kink_vac_plot = 0
dB_kink_vac_plot_phase = 0
#dB_kink and the nq+5 for vacuum case - in the paper
dB_kink_fixed_vac = 1
#plot of the Bn/Li vs q95 to check that the scan is good
Bn_Li_q95_plot = 0
#(dBn=2)/(dBn=2+dBn=4) and the average equivalent
n2_n4_dBres_comparison = 0
#dBkinkn=2/(dBkinkn=2 + dBkinkn=4) single plot
single_kink_proportion=1
#dBresn=2/(dBresn=2 + dBresn=4) and the dBres ave equivalent also
db_res_dB_res_ave_proportion = 0
#dBresn=2 and (dBresn=2 + dBresn=4)
dB_res_n2_dB_res_sum = 0
#plot of dBkink2 and dBkink2 + dBkink4
dBkink2_dBkink_sum = 0
#plot for paper for dBres n2 and n4, dBres n=2, dBres n=4, dBres sum and dBres proportion
#This figure is in the paper
dBres_n2_n4_complete_comparison = 1
#plot for paper for dBres n2 and n4
#dBres n=2, dBres n=4, dBres sum and dBres proportion
#This figure is in the paper
dBres_n2_n4_complete_comparison = 1
#line plot for paper for dBkink n2 and n4 for delta_phi_ul = 0
phi_0_q95_slice = 1
#phasing sweep experiment simulation for paper
phasing_sweep_simulation =1

#plot_type = 'best_harmonic'
#plot_type = 'normalised'
#plot_type = 'normalised_average'
#plot_type = 'standard_average'

project_dict = pickle.load(file(file_name,'r'))
phasing = 0.
#phasing = np.arange(0.,360.,1)
phasing = phasing/180.*np.pi
key_list = project_dict['sims'].keys()
#extract the relevant data from the pickle file and put it into lists


def do_everything(file_name, s_surface, phasing,phase_machine_ntor, fixed_harmonic = 5, reference_offset=[2,0], reference_dB_kink='plas'):
    project_dict = pickle.load(file(file_name,'r'))
    key_list = project_dict['sims'].keys()

    n = np.abs(project_dict['details']['MARS_settings']['<<RNTOR>>'])
    q95_list, Bn_Li_list, time_list = dBres_dBkink.extract_q95_Bn(project_dict, bn_li = 1)
    res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower = dBres_dBkink.extract_dB_res(project_dict)

    amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close = dBres_dBkink.extract_dB_kink(project_dict, s_surface)
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

    #Convert the complex number into an amplitude and phase
    plot_quantity_vac_phase = np.angle(plot_quantity_vac,deg=True).tolist()
    plot_quantity_plas_phase = np.angle(plot_quantity_plas,deg=True).tolist()
    plot_quantity_tot_phase = np.angle(plot_quantity_tot,deg=True).tolist()
    plot_quantity_vac = np.abs(plot_quantity_vac).tolist()
    plot_quantity_plas = np.abs(plot_quantity_plas).tolist()
    plot_quantity_tot = np.abs(plot_quantity_tot).tolist()

    #create copies before everything is arranged
    q95_list_copy = copy.deepcopy(q95_list)
    Bn_Li_list_copy = copy.deepcopy(Bn_Li_list)

    #create the sorted lists - sorted by q95....
    tmp = zip(*sorted(zip(q95_list, Bn_Li_list, plot_quantity_plas,plot_quantity_vac, plot_quantity_tot,
                          plot_quantity_plas_phase, plot_quantity_vac_phase, plot_quantity_tot_phase, 
                          mode_list, time_list, key_list, resonant_close)))
    q95_list_arranged, Bn_Li_list_arranged, plot_quantity_plas_arranged, plot_quantity_vac_arranged, plot_quantity_tot_arranged, plot_quantity_plas_phase_arranged, plot_quantity_vac_phase_arranged, plot_quantity_tot_phase_arranged, mode_list_arranged, time_list_arranged, key_list_arranged, resonant_close_arranged = tmp

    #Calculate the phasing dependence of dBkink
    plot_array_plasma, plot_array_vac, plot_array_tot, plot_array_vac_fixed, q95_array, phasing_array, plot_array_plasma_fixed, plot_array_plasma_phase, plot_array_vac_phase, plot_array_vac_fixed_phase, plot_array_plasma_fixed_phase = dBres_dBkink.dB_kink_phasing_dependence(q95_list_copy, lower_values_plasma, upper_values_plasma, lower_values_vac, upper_values_vac, lower_values_tot, upper_values_tot, lower_values_vac_fixed, upper_values_vac_fixed, phase_machine_ntor, upper_values_plas_fixed, lower_values_plas_fixed, n, n_phases = 360)

    #Calculate the phasing dependence of dBres
    plot_array_vac_res, plot_array_plas_res, plot_array_vac_res_ave, plot_array_plas_res_ave = dBres_dBkink.dB_res_phasing_dependence(phasing_array, q95_array, res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, phase_machine_ntor, n)

    string_list = ['q95_list_arranged', 'Bn_Li_list_arranged', 'plot_quantity_plas_arranged', 'plot_quantity_vac_arranged', 'plot_quantity_tot_arranged', 'plot_quantity_plas_phase_arranged', 'plot_quantity_vac_phase_arranged', 'plot_quantity_tot_phase_arranged', 'mode_list_arranged', 'time_list_arranged', 'key_list_arranged', 'resonant_close_arranged','lower_values_plasma', 'upper_values_plasma', 'lower_values_vac', 'upper_values_vac', 'lower_values_vac_fixed', 'upper_values_vac_fixed', 'q95_list_copy', 'plot_array_plasma', 'plot_array_vac', 'plot_array_tot',  'plot_array_vac_fixed', 'q95_array', 'phasing_array', 'plot_array_vac_res', 'plot_array_plas_res', 'plot_array_vac_res_ave', 'plot_array_plas_res_ave','n', 'plot_array_plasma_fixed', 'plot_array_plasma_phase', 'plot_array_vac_phase', 'plot_array_vac_fixed_phase', 'plot_array_plasma_fixed_phase', 'max_loc_list']
    output_dict = {}
    for i in string_list:
        output_dict[i] = eval(i)
    return output_dict


if Bn_Li_q95_plot:
    q95_list, Bn_Li_list, Beta_N, time_list = dBres_dBkink.extract_q95_Bn2(project_dict)
    fig, ax = pt.subplots()
    ax.plot(q95_list, Bn_Li_list, '--')
    ax.plot(q95_list, Beta_N, '-')
    fig.canvas.draw(); fig.show()

answers = do_everything(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink)
answers2 = do_everything(file_name2, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink)


if various_line_plots:
    #Shows the location of all the peeling spikes relative to the resonant q surfaces
    fig_single, ax_single = pt.subplots(nrows = 3, sharex = True)
    ax_single[0].plot(answers['q95_list_arranged'], answers['resonant_close_arranged'], '.-')
    ax_single[1].plot(answers['q95_list_arranged'], answers['plot_quantity_plas_arranged'], 'o-', label = 'plasma')
    #ax_single[1].plot(answers['q95_array'], answers['plot_array_plasma_fixed'][0,:], 'o-')
    #ax_single[1].plot(answers['q95_array'], answers['plot_array_vac_fixed'][0,:], 'o-')
    ax_single[2].plot(answers['q95_list_arranged'], answers['plot_quantity_plas_phase_arranged'], 'o-', label = 'plasma')
    fig_single.canvas.draw(); fig_single.show()

    #amplitude and phase versus q95 and time
    fig, ax = pt.subplots(ncols = 2, nrows = 2)
    ax[0,0].plot(answers['q95_list_arranged'], answers['plot_quantity_plas_arranged'], 'o-', label = 'plasma')
    ax[0,0].plot(answers['q95_list_arranged'], answers['plot_quantity_vac_arranged'], 'o-', label = 'vacuum')
    ax[0,0].plot(answers['q95_list_arranged'], answers['plot_quantity_tot_arranged'], 'o-',label = 'total')
    ax[0,0].plot(answers['q95_list_arranged'], np.array(answers['mode_list_arranged'])/2., 'x-',label = 'm/2')
    if plot_text ==1:
        for i in range(0,len(answers['key_list_arranged'])):
            ax[0,0].text(answers['q95_list_arranged'][i], answers['plot_quantity_plas_arranged'][i], str(answers['key_list_arranged'][i]), fontsize = 8.5)
    leg = ax[0,0].legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax[0,0].set_ylabel('mode amplitude')
    ax[0,0].set_title('sqrt(psi)=%.2f'%(s_surface))
    ax[0,1].plot(answers['time_list_arranged'], answers['plot_quantity_plas_arranged'], 'o', label = 'plasma')
    ax[0,1].plot(answers['time_list_arranged'], answers['plot_quantity_vac_arranged'], 'o', label = 'vacuum')
    ax[0,1].plot(answers['time_list_arranged'], answers['plot_quantity_tot_arranged'], 'o',label = 'total')
    ax[1,0].plot(answers['q95_list_arranged'], answers['plot_quantity_plas_phase_arranged'], 'o-', label = 'plasma')
    ax[1,0].plot(answers['q95_list_arranged'], answers['plot_quantity_vac_phase_arranged'], 'o-', label = 'vacuum')
    ax[1,0].plot(answers['q95_list_arranged'], answers['plot_quantity_tot_phase_arranged'], 'o-',label = 'total')
    leg = ax[1,0].legend(loc='best', fancybox = True)
    leg.get_frame().set_alpha(0.5)
    ax[1,0].set_xlabel('q95')
    ax[1,0].set_ylabel('phase (deg)')
    ax[1,1].plot(answers['time_list_arranged'], answers['plot_quantity_plas_phase_arranged'], 'o', label = 'plasma')
    ax[1,1].plot(answers['time_list_arranged'], answers['plot_quantity_vac_phase_arranged'], 'o', label = 'vacuum')
    ax[1,1].plot(answers['time_list_arranged'], answers['plot_quantity_tot_phase_arranged'], 'o',label = 'total')
    ax[1,1].set_xlabel('time (ms)')
    fig.suptitle(file_name,fontsize=8)
    fig.canvas.draw()
    fig.show()

    #plot q95 versus Bn/Li
    fig, ax = pt.subplots(ncols = 2, sharey=True)
    ax[0].plot(answers['q95_list_arranged'], answers['Bn_Li_list_arranged'], 'o-')
    ax[0].set_xlabel('q95')
    ax[0].set_ylabel('Bn/Li')
    ax[0].set_ylim([0,3.5])
    ax[1].plot(answers['time_list_arranged'], answers['Bn_Li_list_arranged'], 'o')
    ax[1].set_xlabel('time (ms)')
    fig.suptitle(file_name,fontsize=8)
    fig.canvas.draw(); fig.show()

if dB_kink_vac_plot:
    #dB_kink and the vacuum harmonic strength of the same m
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True)
    color_plot = ax[0].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_plasma'], cmap='hot')
    color_plot.set_clim([0, 1.5])
    color_plot2 = ax[1].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_vac'], cmap='hot')
    #color_plot2.set_clim([0.002, 3])
    ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmax(answers['plot_array_plasma'],axis=0)],'k.')
    ax[1].plot(answers['q95_array'], answers['phasing_array'][np.argmax(answers['plot_array_vac'],axis=0)],'k.')
    ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_plasma'],axis=0)],'b.')
    ax[1].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_vac'],axis=0)],'b.')
    #color_plot.set_clim()
    #ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
    ax[0].set_ylabel('Phasing (deg)')
    ax[1].set_ylabel('Phasing (deg)')
    ax[0].set_title('Kink Amplitude - Plasma')
    ax[1].set_title('Kink Amplitude - Vacuum')
    ax[0].set_xlim([2.5,6.0])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
    tmp_xaxis = np.arange(1,10,0.1)
    tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
    cbar = pt.colorbar(color_plot, ax = ax[0])
    ax[1].set_xlabel(r'$q_{95}$', fontsize = 20)
    cbar.ax.set_ylabel(r'$\delta B_{kink}^{n=%d}$ G/kA'%(answers['n']),fontsize=20)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel(r'$\delta B_{kink,vac}^{n=%d}$ G/kA'%(answers['n'],),fontsize=20)
    fig.canvas.draw(); fig.show()

if dB_kink_vac_plot_phase:
    #dB_kink and the vacuum harmonic strength of the same m
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True)
    tmp = answers['plot_array_plasma_phase'] - answers['plot_array_vac_phase']
    lower_limit = -20
    tmp[tmp<-10]+=360;tmp[tmp<-10]+=360;tmp[tmp<-10]+=360
    tmp[tmp>350]-=360
    #color_plot = ax[0].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_plasma_phase'], cmap='hot')
    color_plot = ax[0].pcolor(answers['q95_array'], answers['phasing_array'], tmp, cmap='hot')
    #color_plot.set_clim([0, 1.5])
    color_plot2 = ax[1].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_vac_fixed_phase'], cmap='hot')
    #color_plot2.set_clim([0.002, 3])
    ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmax(answers['plot_array_tot'],axis=0)],'k.')
    ax[1].plot(answers['q95_array'], answers['phasing_array'][np.argmax(answers['plot_array_vac'],axis=0)],'k.')
    ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_tot'],axis=0)],'b.')
    ax[1].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_vac'],axis=0)],'b.')
    #color_plot.set_clim()
    #ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
    ax[0].set_ylabel('Phasing (deg)')
    ax[1].set_ylabel('Phasing (deg)')
    ax[0].set_title('Arg(Kink Amplitude) - Plasma')
    ax[1].set_title('Arg(Kink Amplitude) - Vacuum')
    ax[0].set_xlim([2.5,6.0])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
    tmp_xaxis = np.arange(1,10,0.1)
    tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
    cbar = pt.colorbar(color_plot, ax = ax[0])
    ax[1].set_xlabel(r'$q_{95}$', fontsize = 20)
    cbar.ax.set_ylabel(r'$\delta B_{kink}$ deg',fontsize=20)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+5,n=%d}$ deg'%(answers['n'],),fontsize=20)
    fig.canvas.draw(); fig.show()


publication_images = 1
if publication_images:
    import matplotlib as mpl
    mpl.rcParams['font.size']=9.0
    mpl.rcParams['axes.titlesize']=9.0#'medium'
    mpl.rcParams['xtick.labelsize']=7.0
    mpl.rcParams['ytick.labelsize']=7.0
    mpl.rcParams['lines.markersize']=4.0
    mpl.rcParams['savefig.dpi']=300

if dB_kink_fixed_vac:
    #########################
    #Plot for the paper
    clim1 = [0,1.5]
    clim2 = [0,0.55]
    cm_to_inch=0.393701
    fig, ax = pt.subplots(nrows = 2, sharex =True, sharey = True)
    if publication_images:
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*cm_to_inch)
    color_plot = ax[0].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_plasma'], cmap='hot', rasterized= 'True')
    #color_plot = ax[0].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_tot_fixed'], cmap='hot', rasterized= 'True')
    color_plot.set_clim(clim1)
    color_plot2 = ax[1].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_vac_fixed'], cmap='hot', rasterized = 'True')
    color_plot2.set_clim(clim2)
    #color_plot2.set_clim([0.002, 3])
    ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmax(answers['plot_array_tot'],axis=0)],'kx')
    #ax[1].plot(answers['q95_array'], answers['phasing_array'][np.argmax(answers['plot_array_vac'],axis=0)],'k.')
    ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_tot'],axis=0)],'b.')

    suppressed_regions = [[3.81,-30,0.01],[3.48,15,0.1],[3.72,15,0.025],[3.75,0,0.025]]
    for i in range(0,len(suppressed_regions)):
        curr_tmp = suppressed_regions[i]
        tmp_angle = curr_tmp[1]*-2.
        if tmp_angle<0:tmp_angle+=360
        if tmp_angle>360:tmp_angle-=360

        ax[0].errorbar(curr_tmp[0], tmp_angle, xerr=curr_tmp[2], yerr=0, ecolor='g')
    #ax[1].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_vac'],axis=0)],'b.')
    #color_plot.set_clim()
    #ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)')#,fontsize = 20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)')#,fontsize = 20)

    ax[0].set_xlim([2.5,6.0])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    #ax[0].plot(np.arange(1,10), np.arange(1,10)*(-55.)+180+180,'b-')
    #ax[1].plot(np.arange(1,10), np.arange(1,10)*(-55.)+180+180,'b-')
    ax[0].locator_params(nbins=4)
    ax[1].locator_params(nbins=4)
    tmp_xaxis = np.arange(1,10,0.1)
    tmp_yaxis = np.arange(1,10,0.1)*(-55.)+180

    cbar = pt.colorbar(color_plot, ax = ax[0])
    ax[1].set_xlabel(r'$q_{95}$')#, fontsize = 20)
    cbar.ax.set_ylabel(r'$\delta B_{kink}^{n=%d}$ G/kA'%(answers['n'],))#,fontsize=20)
    cbar.ax.set_title('(a)')
    
    cbar.set_ticks(np.round(np.linspace(clim1[0], clim1[1],5),decimals=2))

    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+%d,n=%d}$ G/kA'%(fixed_harmonic,answers['n']))#,fontsize=20)
    cbar.ax.set_title('(b)')
    cbar.set_ticks(np.round(np.linspace(clim2[0], clim2[1],5),decimals=2))
    #cbar.locator.nbins=4
    #cbar.set_ticks(cbar.ax.get_yticks()[::2])
    fig.canvas.draw();
    fig.savefig('tmp2.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig('tmp2.pdf', bbox_inches='tight', pad_inches=0)
    fig.show()
    #fig.savefig('tmp2.svg', bbox_inches='tight', pad_inches=0)

    #fig.savefig('tmp3.eps', bbox='tight', pad_inches=0)
    #fig.savefig('tmp3.pdf', bbox='tight', pad_inches=0)
    #fig.savefig('tmp4.eps', bbox='tight', pad_inches=0)
    #fig.savefig('tmp4.pdf', bbox='tight', pad_inches=0)


if n2_n4_dBres_comparison:
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    color_plot = ax[0].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_vac_res'], cmap='hot', rasterized=True)
    ax[0].contour(answers['q95_array'],answers['phasing_array'], answers['plot_array_vac_res'], colors='white')
    color_plot2 = ax[1].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_vac_res_ave'], cmap='hot', rasterized=True)
    ax[1].contour(answers['q95_array'],answers['phasing_array'], answers['plot_array_vac_res_ave'], colors='white')
    color_plot.set_clim([0,10])
    color_plot2.set_clim([0,0.75])

    title_string1 = 'Total Forcing'
    title_string2 = 'Average Forcing'

    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)

    ax[0].set_title(r'$\delta B_{res}^{n=2}$/($\delta B_{res}^{n=2}$ + $\delta B_{res}^{n=4}$)',fontsize=20)
    ax[1].set_title(r'$\overline{\delta B}_{res}^{n=2}$/($\overline{\delta B}_{res}^{n=2}$ + $\overline{\delta B}_{res}^{n=4}$)',fontsize=20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    #ax[0].set_ylabel('Phasing (deg)')
    #ax[1].set_ylabel('Phasing (deg)')

    cbar = pt.colorbar(color_plot, ax = ax[0])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    fig.canvas.draw(); fig.show()


#Compare dBres calculated from the plasma and vacuum fields
#THIS IS NEW FOR NOW
if n2_n4_dBres_comparison:
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    color_plot = ax[0].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_vac_res'], cmap='hot', rasterized=True)
    ax[0].contour(answers['q95_array'],answers['phasing_array'], answers['plot_array_vac_res'], colors='white')
    color_plot2 = ax[1].pcolor(answers['q95_array'], answers['phasing_array'], answers['plot_array_plas_res'], cmap='hot', rasterized=True)
    ax[1].contour(answers['q95_array'],answers['phasing_array'], answers['plot_array_plas_res'], colors='white')
    color_plot.set_clim([0,10])
    color_plot2.set_clim([0,10])

    title_string1 = 'Total Forcing (from vac calc)'
    title_string2 = 'Total Forcing (from plas calc)'

    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)

    ax[0].set_title(r'$\delta B_{res}^{n=2}$/($\delta B_{res}^{n=2}$ + $\delta B_{res}^{n=4}$)',fontsize=20)
    ax[1].set_title(r'$\overline{\delta B}_{res}^{n=2}$/($\overline{\delta B}_{res}^{n=2}$ + $\overline{\delta B}_{res}^{n=4}$)',fontsize=20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    #ax[0].set_ylabel('Phasing (deg)')
    #ax[1].set_ylabel('Phasing (deg)')

    cbar = pt.colorbar(color_plot, ax = ax[0])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    fig.canvas.draw(); fig.show()


#Start making the comparisons between two different n values....
#need to check if we have the same q95 values first....
#This needs to be seriously changed....
if len(answers['q95_array']) > len(answers2['q95_array']):
    truth_array = answers['q95_array'] * 0
    for i in answers2['q95_array']:
        tmp_loc = np.argmin(np.abs(answers['q95_array'] - i))
        if np.abs(answers['q95_array'][tmp_loc] - i)<0.0001:
            truth_array[tmp_loc] = 1
elif len(answers['q95_array']) == len(answers2['q95_array']):
    truth_array = answers['q95_array'] * 0+1

#convert to boolean
truth_array = (truth_array==1)

#calculate the proportions, sums and normalised sums
quant1 = answers['plot_array_vac_res'][:,truth_array]/(answers['plot_array_vac_res'][:,truth_array] + answers2['plot_array_vac_res'])
quant2 = answers['plot_array_vac_res_ave'][:,truth_array]/(answers['plot_array_vac_res_ave'][:,truth_array] + answers2['plot_array_vac_res_ave'])
dB_res_sum = answers['plot_array_vac_res'][:,truth_array] + answers2['plot_array_vac_res']
dB_res_sum2 = answers['plot_array_vac_res_ave'][:,truth_array] + answers2['plot_array_vac_res_ave']
dB_kink_sum = answers['plot_array_plasma'][:,truth_array] + answers2['plot_array_plasma']
dB_kink_sum_norm = answers['plot_array_plasma'][:,truth_array]/(answers['plot_array_plasma'][:,truth_array] + answers2['plot_array_plasma'])

#dBkinkn=2/(dBkinkn=2 + dBkinkn=4) single plot
single_kink_proportion = 1
if single_kink_proportion:
    pub_fig = 1
    if pub_fig:
        cm_to_inch=0.393701
        import matplotlib as mpl
        old_rc_Params = mpl.rcParams
        mpl.rcParams['font.size']=8.0
        mpl.rcParams['axes.titlesize']=8.0#'medium'
        mpl.rcParams['xtick.labelsize']=8.0
        mpl.rcParams['ytick.labelsize']=8.0
        mpl.rcParams['lines.markersize']=1.0
        mpl.rcParams['savefig.dpi']=300
    fig, ax = pt.subplots();ax = [ax]
    if pub_fig:
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*cm_to_inch*0.8)
    
    color_plot = ax[0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], dB_kink_sum_norm, cmap='hot', rasterized=True)
    color_plot.set_clim([0,1])
    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[0].set_xlabel(r'$q_{95}$', fontsize=9)
    ax[0].set_title(r'$\delta B_{kink}^{n=2}$/($\delta B_{kink}^{n=2}$ + $\delta B_{kink}^{n=4}$)',fontsize=9)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 9)
    cbar = pt.colorbar(color_plot, ax = ax[0],use_gridspec=True)
    #fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
    fig.tight_layout()
    #fig.subplots_adjust(hspace=0.02, wspace=0.01)#,left=0.10, bottom=0.10,top=0.95, right=0.95)
    #fig.savefig('hello.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig('figure16.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig('figure16.eps', bbox_inches='tight', pad_inches=0.01)
    fig.canvas.draw(); fig.show()

#dBresn=2/(dBresn=2 + dBresn=4) single plot
    #ax[0,0].contour(answers['q95_array'][truth_array],answers['phasing_array'], answers['plot_array_vac_res_ave'][:,truth_array], colors='white')

single_res_proportion = 1
if single_res_proportion:
    pub_fig = 1
    if pub_fig:
        cm_to_inch=0.393701
        import matplotlib as mpl
        old_rc_Params = mpl.rcParams
        mpl.rcParams['font.size']=8.0
        mpl.rcParams['axes.titlesize']=8.0#'medium'
        mpl.rcParams['xtick.labelsize']=8.0
        mpl.rcParams['ytick.labelsize']=8.0
        mpl.rcParams['lines.markersize']=1.0
        mpl.rcParams['savefig.dpi']=300
    fig, ax = pt.subplots();ax = [ax]
    if pub_fig:
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*cm_to_inch*0.8)
    
    color_plot3 = ax[0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], quant2, cmap='hot', rasterized=True)
    color_plot3.set_clim([0, 1.])
    cbar = pt.colorbar(color_plot3, ax = ax[0],use_gridspec=True)
    cbar.set_ticks(np.round(np.linspace(clim1[0], clim1[1],6),decimals=1))
    ax[0].set_title(r'$\overline{\delta B}_{res}^{n=%d}$/($\overline{\delta B}_{res}^{n=%d}$ + $\overline{\delta B}_{res}^{n=%d}$)'%(answers['n'],answers['n'],answers2['n']))
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)')
    ax[0].set_xlabel(r'$q_{95}$')
    ax[0].locator_params(nbins=4)
    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    #fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
    fig.tight_layout()
    #fig.savefig('hello.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig('figure15.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig('figure15.eps', bbox_inches='tight', pad_inches=0.01)
    fig.canvas.draw(); fig.show()




#dBresn=2/(dBresn=2 + dBresn=4) and the dBres ave equivalent also
db_res_dB_res_ave_proportion = 1
if db_res_dB_res_ave_proportion:
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    color_plot = ax[0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], quant1, cmap='hot', rasterized=True)
    ax[0].contour(answers['q95_array'][truth_array],answers['phasing_array'], quant2, colors='white')
    color_plot2 = ax[1].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], quant2, cmap='hot', rasterized=True)
    ax[1].contour(answers['q95_array'][truth_array],answers['phasing_array'], quant2, colors='white')
    color_plot.set_clim([0,1])
    color_plot2.set_clim([0,1])
    title_string1 = 'Total Forcing'
    title_string2 = 'Average Forcing'
    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)
    ax[0].set_title(r'$\delta B_{res}^{n=2}$/($\delta B_{res}^{n=2}$ + $\delta B_{res}^{n=4}$)',fontsize=20)
    ax[1].set_title(r'$\overline{\delta B}_{res}^{n=2}$/($\overline{\delta B}_{res}^{n=2}$ + $\overline{\delta B}_{res}^{n=4}$)',fontsize=20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    #ax[0].set_ylabel('Phasing (deg)')
    #ax[1].set_ylabel('Phasing (deg)')
    cbar = pt.colorbar(color_plot, ax = ax[0])
    #cbar.ax.set_ylabel('G/kA',fontsize = 16)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    #cbar.ax.set_ylabel('G/kA',fontsize = 16)
    fig.canvas.draw(); fig.show()



#plot of dBkink2 and dBkink2 + dBkink4 
#plot of db_res n=2 and db_res n=2 + db_res n=4
if dB_res_n2_dB_res_sum:
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    color_plot = ax[0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], answers['plot_array_vac_res'][:,truth_array], cmap='hot', rasterized=True)
    ax[0].contour(answers['q95_array'][truth_array],answers['phasing_array'], answers['plot_array_vac_res'][:,truth_array], colors='white')
    color_plot2 = ax[1].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], dB_res_sum, cmap='hot', rasterized=True)
    ax[1].contour(answers['q95_array'][truth_array],answers['phasing_array'], dB_res_sum, colors='white')
    #color_plot.set_clim([0,1])
    #color_plot2.set_clim([0,1])
    title_string1 = 'Total Forcing'
    title_string2 = 'Average Forcing'
    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)
    ax[0].set_title(r'$\delta B_{res}^{n=2}$',fontsize=20)
    ax[1].set_title(r'$\delta B_{res}^{n=2} + \delta B_{res}^{n=4}$',fontsize=20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    #ax[0].set_ylabel('Phasing (deg)')
    #ax[1].set_ylabel('Phasing (deg)')
    cbar = pt.colorbar(color_plot, ax = ax[0])
    #cbar.ax.set_ylabel('G/kA',fontsize = 16)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    #cbar.ax.set_ylabel('G/kA',fontsize = 16)
    fig.canvas.draw(); fig.show()

#plot of dBkink2 and dBkink2 + dBkink4
if dBkink2_dBkink_sum:
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    color_plot = ax[0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], answers['plot_array_tot'][:,truth_array], cmap='hot', rasterized=True)
    #ax[0].contour(answers['q95_array'][truth_array],answers['phasing_array'], answers['plot_array_tot'][:,truth_array], colors='white')
    color_plot2 = ax[1].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], dB_kink_sum, cmap='hot', rasterized=True)
    #ax[1].contour(answers['q95_array'][truth_array],answers['phasing_array'], dB_kink_sum, colors='white')
    color_plot.set_clim([0, 1.5])
    color_plot2.set_clim([0, 1.5])
    title_string1 = 'Total Forcing'
    title_string2 = 'Average Forcing'
    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)
    ax[0].set_title(r'$\delta B_{kink}^{n=%d}$'%(answers['n']),fontsize=20)
    ax[1].set_title(r'$\delta B_{kink}^{n=%d} + \delta B_{kink}^{n=%d}$'%(answers['n'],answers2['n']),fontsize=20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    #ax[0].set_ylabel('Phasing (deg)')
    #ax[1].set_ylabel('Phasing (deg)')
    cbar = pt.colorbar(color_plot, ax = ax[0])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    #cbar.ax.set_ylabel('G/kA',fontsize = 16)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    #cbar.ax.set_ylabel('G/kA',fontsize = 16)
    fig.canvas.draw(); fig.show()

#plot for paper for dBres n2 and n4
if dBres_n2_n4_complete_comparison:
    #dBres n=2, dBres n=4, dBres sum and dBres proportion
    #This figure is in the paper
    clim1 = [0,1]

    fig, ax = pt.subplots(nrows = 2, ncols=2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    pt.subplots_adjust(wspace = .1,hspace=0.25)
    if publication_images:
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*cm_to_inch*1.2)

    color_plot = ax[1,0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], answers['plot_array_vac_res_ave'][:,truth_array], cmap='hot', rasterized=True)
    #ax[0,0].contour(answers['q95_array'][truth_array],answers['phasing_array'], answers['plot_array_vac_res_ave'][:,truth_array], colors='white')
    cbar = pt.colorbar(color_plot, ax = ax[1,0])
    cbar.ax.set_title('G/kA')
    color_plot.set_clim(clim1)
    cbar.set_ticks(np.round(np.linspace(clim1[0], clim1[1],6),decimals=1))
    ax[1,0].set_title(r'(a) $\overline{\delta B}_{res}^{n=%d}$'%(answers['n']))
    ax[1,0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)')

    color_plot2 = ax[1,1].pcolor(answers2['q95_array'][truth_array], answers2['phasing_array'], answers2['plot_array_vac_res_ave'], cmap='hot', rasterized=True)
    #ax[0,1].contour(answers['q95_array'][truth_array],answers['phasing_array'], answers2['plot_array_vac_res_ave'], colors='white')
    cbar = pt.colorbar(color_plot2, ax = ax[1,1])
    cbar.ax.set_title('G/kA')
    color_plot2.set_clim(clim1)
    cbar.set_ticks(np.round(np.linspace(clim1[0], clim1[1],6),decimals=1))
    ax[1,1].set_title(r'(b) $\overline{\delta B}_{res}^{n=%d}$'%(answers2['n']))

    color_plot3 = ax[0,0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], quant2, cmap='hot', rasterized=True)
    #ax[1,0].contour(answers['q95_array'][truth_array], answers['phasing_array'], quant2, colors='white')
    color_plot3.set_clim([0, 1.])
    cbar = pt.colorbar(color_plot3, ax = ax[0,0])
    color_plot3.set_clim(clim1)
    cbar.set_ticks(np.round(np.linspace(clim1[0], clim1[1],6),decimals=1))
    #ax[1,0].set_title(r'(c) $\overline{\delta B}_{res}^{n=%d}$/($\overline{\delta B}_{res}^{n=%d}$ + $\overline{\delta B}_{res}^{n=%d}$)'%(answers['n'],answers['n'],answers2['n']))
    ax[0,0].set_title(r'(c) $\overline{\delta B}_{res}^{n=%d} /\sum^{n=%d,%d}\overline{\delta B}_{res}^{n}$'%(answers['n'],answers['n'],answers2['n']))
    ax[0,0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)')
    ax[1,0].set_xlabel(r'$q_{95}$')

    color_plot4 = ax[0,1].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], dB_res_sum2, cmap='hot', rasterized=True)
    #ax[1,1].contour(answers['q95_array'][truth_array],answers['phasing_array'], dB_res_sum, colors='white')
    cbar = pt.colorbar(color_plot4, ax = ax[0,1])
    color_plot4.set_clim(clim1)
    cbar.set_ticks(np.round(np.linspace(clim1[0], clim1[1],6),decimals=1))
    cbar.ax.set_title('G/kA')
    #ax[1,1].set_title(r'(d) $\overline{\delta B}_{res}^{n=%d} + \overline{\delta B}_{res}^{n=%d}$'%(answers['n'],answers2['n']))
    ax[0,1].set_title(r'(d) $\sum^{n=%d,%d}\overline{\delta B}_{res}^{n}$'%(answers['n'],answers2['n']))
    ax[1,1].set_xlabel(r'$q_{95}$')
    ax[0,0].set_xlim([2.6, 6])
    ax[0,0].set_ylim([0, 360])

    ax[0,0].locator_params(nbins=4)
    ax[1,0].locator_params(nbins=4)
    ax[0,1].locator_params(nbins=4)
    ax[1,1].locator_params(nbins=4)

    fig.savefig('tmp3b.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig('tmp3b.pdf', bbox_inches='tight', pad_inches=0)

    fig.canvas.draw(); fig.show()

#plot for paper for dBkink n2 and n4
if dBres_n2_n4_complete_comparison:
    #dBkink n=2, dBkink n=4, dBkink sum and dBkink proportion
    #This figure is in the paper
    fig, ax = pt.subplots(nrows = 2, ncols=2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    color_plot = ax[1,0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], answers['plot_array_plasma'][:,truth_array], cmap='hot', rasterized=True)
    #ax[0,0].contour(answers['q95_array'][truth_array],answers['phasing_array'], answers['plot_array_vac_res_ave'][:,truth_array], colors='white')
    cbar = pt.colorbar(color_plot, ax = ax[1,0])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    color_plot.set_clim([0, 1.5])
    ax[1,0].set_title(r'$\delta B_{kink}^{n=%d}$'%(answers['n']),fontsize=16)
    ax[1,0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 16)
    ax[1,0].set_xlabel(r'$q_{95}$', fontsize=16)

    color_plot2 = ax[1,1].pcolor(answers2['q95_array'][truth_array], answers2['phasing_array'], answers2['plot_array_plasma'], cmap='hot', rasterized=True)
    #ax[0,1].contour(answers['q95_array'][truth_array],answers['phasing_array'], answers2['plot_array_vac_res_ave'], colors='white')
    cbar = pt.colorbar(color_plot2, ax = ax[1,1])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    color_plot2.set_clim([0, 1.5])
    ax[1,1].set_title(r'$\delta B_{kink}^{n=%d}$'%(answers2['n']),fontsize=16)
    ax[1,1].set_xlabel(r'$q_{95}$', fontsize=16)

    color_plot3 = ax[0,0].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], dB_kink_sum_norm, cmap='hot', rasterized=True)
    #ax[1,0].contour(answers['q95_array'][truth_array], answers['phasing_array'], quant2, colors='white')
    cbar = pt.colorbar(color_plot3, ax = ax[0,0])
    color_plot3.set_clim([0, 1])
    ax[0,0].set_title(r'$\delta B_{kink}^{n=%d}$/($\delta B_{kink}^{n=%d}$ + $\delta B_{kink}^{n=%d}$)'%(answers['n'],answers['n'],answers2['n']),fontsize=16)
    ax[0,0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 16)

    color_plot4 = ax[0,1].pcolor(answers['q95_array'][truth_array], answers['phasing_array'], dB_kink_sum, cmap='hot', rasterized=True)
    #ax[1,1].contour(answers['q95_array'][truth_array],answers['phasing_array'], dB_res_sum, colors='white')
    cbar = pt.colorbar(color_plot4, ax = ax[0,1])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    color_plot4.set_clim([0, 1.5])
    ax[0,1].set_title(r'$\delta B_{kink}^{n=%d} + \delta B_{kink}^{n=%d}$'%(answers['n'], answers2['n']),fontsize=16)

    ax[0,0].set_xlim([2.6, 6])
    ax[0,0].set_ylim([0, 360])

    fig.canvas.draw(); fig.show()

#line plot for paper for dBkink n2 and n4 for delta_phi_ul = 0
if phi_0_q95_slice:
    fig,ax = pt.subplots()
    tmp_loc = np.argmin(np.abs(answers['phasing_array'] - 0))
    ax.plot(answers['q95_array'][truth_array],answers['plot_array_tot'][tmp_loc,truth_array],'x-', label = r'$\delta B_{kink}^{n=2}$')
    ax.plot(answers2['q95_array'][truth_array],answers2['plot_array_tot'][tmp_loc,truth_array],'.-', label = r'$\delta B_{kink}^{n=4}$')
    ax.set_ylabel('Amplitude (G/kA)')
    ax.set_xlabel(r'$q_{95}$', fontsize=16)
    ax.set_title(r'$\Delta \phi_{ul} = 0^o$, $\beta_N / \ell_i = 1.15$',fontsize = 16)
    ax.legend(loc='best')
    ax.set_xlim([2.6, 6])
    fig.canvas.draw(); fig.show()

#phasing sweep experiment simulation for paper
single_q95_value = 3.5
tmp_loc = np.argmin(np.abs(answers['q95_array']-single_q95_value))
single_db_res = answers['plot_array_vac_res'][:,tmp_loc]
single_db_res_ave = answers['plot_array_vac_res_ave'][:,tmp_loc]
single_db_kink = answers['plot_array_tot'][:,tmp_loc]
if phasing_sweep_simulation:
    fig, ax = pt.subplots();ax = [ax]
    ax[0].plot(answers['phasing_array'],single_db_res/np.max(single_db_res), '-',label=r'$\delta B_{res}^{n=%d}$'%(answers['n']))
    ax[0].plot(answers['phasing_array'],single_db_kink/np.max(single_db_kink), '-.',label=r'$\delta B_{kink}^{n=%d}$'%answers['n'])
    include_another_mode = 0
    if include_another_mode:
        tmp_loc = np.argmin(np.abs(answers2['q95_array']-single_q95_value))
        single_db_res2 = answers2['plot_array_vac_res'][:,tmp_loc]
        single_db_res_ave2 = answers2['plot_array_vac_res_ave'][:,tmp_loc]
        single_db_kink2 = answers2['plot_array_tot'][:,tmp_loc]
        ax[0].plot(answers2['phasing_array'],single_db_res2/np.max(single_db_res2), '-',label=r'$\delta B_{res}^{n=%d}$'%(answers2['n']))
        ax[0].plot(answers2['phasing_array'],single_db_kink2/np.max(single_db_kink2), '-.',label=r'$\delta B_{kink}^{n=%d}$'%(answers2['n']))

    #ax[0].plot(answers['phasing_array'],single_db_res_ave/np.max(single_db_res_ave), '--', label='db_res_ave')
    ax[0].legend(loc='best'); ax[0].grid()
    ax[0].set_xlabel(r'$\Delta \phi_{ul}$ (deg) or time (1/180 s)', fontsize = 20)
    ax[0].set_ylabel('Normalised Amplitude', fontsize = 16)
    tmp_ax = ax[0].twiny()
    tmp_ax.set_xlabel('Time (s)')
    tmp_ax.set_xlim([0,2])
    ax[0].set_xlim([0,360]); ax[0].set_ylim([0,1])
    fig.canvas.draw(); fig.show()

single_q95_values = [3.5]
single_q95_values = np.linspace(3,5,15)
include_line_plots_n2_n4 = 0
if include_line_plots_n2_n4:
    for single_q95_value in single_q95_values:
        fig, ax = pt.subplots(nrows = 2, sharex = True)
        tmp_loc = np.argmin(np.abs(answers['q95_array']-single_q95_value))
        single_db_res = answers['plot_array_vac_res'][:,tmp_loc]
        single_db_res_ave = answers['plot_array_vac_res_ave'][:,tmp_loc]
        single_db_kink = answers['plot_array_tot'][:,tmp_loc]
        ax[0].plot(answers['phasing_array'],single_db_res_ave, '-',label=r'$\overline{\delta B}_{res}^{n=%d}$'%(answers['n']))
        ax[1].plot(answers['phasing_array'],single_db_kink, '-',label=r'$\delta B_{kink}^{n=%d}$'%(answers['n']))
        include_another_mode = 1
        if include_another_mode:
            tmp_loc = np.argmin(np.abs(answers2['q95_array']-single_q95_value))
            single_db_res2 = answers2['plot_array_vac_res'][:,tmp_loc]
            single_db_res_ave2 = answers2['plot_array_vac_res_ave'][:,tmp_loc]
            single_db_kink2 = answers2['plot_array_tot'][:,tmp_loc]
            ax[0].plot(answers2['phasing_array'],single_db_res_ave2, '-.',label=r'$\overline{\delta B}_{res}^{n=%d}$'%(answers2['n']))
            ax[1].plot(answers2['phasing_array'],single_db_kink2, '-.',label=r'$\delta B_{kink}^{n=%d}$'%(answers2['n']))

        #ax[0].plot(answers['phasing_array'],single_db_res_ave/np.max(single_db_res_ave), '--', label='db_res_ave')
        ax[0].legend(loc='best', prop={'size':18}); ax[0].grid()
        ax[1].legend(loc='best',prop={'size':18}); ax[1].grid()
        ax[1].set_xlabel(r'$\Delta \phi_{ul}$ (deg)', fontsize = 18)
        ax[0].set_ylabel('Amplitude (G/kA)', fontsize = 16)
        ax[1].set_ylabel('Amplitude (G/kA)', fontsize = 16)
        ax[0].set_title('%.2f'%(single_q95_value))
        ax[0].set_xlim([0,360]); #ax[0].set_ylim([0,1])
        fig.canvas.draw(); fig.show()

plot_quantity = 'plasma'
plot_PEST_pics = 0
if plot_PEST_pics:
    for tmp_loc, i in enumerate(key_list_arranged):
        print i
        I0EXP = RZfuncs.I0EXP_calc_real(n,I)
        facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

        print '===========',i,'==========='
        if plot_quantity=='total' or plot_quantity=='plasma':
            upper_file_loc = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
            lower_file_loc = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']
        elif plot_quantity=='vacuum':
            upper_file_loc = project_dict['sims'][i]['dir_dict']['mars_upper_vacuum_dir']
            lower_file_loc = project_dict['sims'][i]['dir_dict']['mars_lower_vacuum_dir']
        elif plot_quantity=='plasma':
            upper_file_loc_vac = project_dict['sims'][i]['dir_dict']['mars_upper_vacuum_dir']
            lower_file_loc_vac = project_dict['sims'][i]['dir_dict']['mars_lower_vacuum_dir']
            upper_file_loc_plasma = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
            lower_file_loc_plasma = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']

        upper = results_class.data(upper_file_loc, I0EXP=I0EXP)
        lower = results_class.data(lower_file_loc, I0EXP=I0EXP)
        upper.get_PEST(facn = facn)
        lower.get_PEST(facn = facn)
        tmp_R, tmp_Z, upper.B1, upper.B2, upper.B3, upper.Bn, upper.BMn, upper.BnPEST = results_class.combine_data(upper, lower, 0)

        if plot_quantity=='plasma':
            #upper_file_loc = project_dict['sims'][i]['dir_dict']['mars_upper_vacuum_dir']
            #lower_file_loc = project_dict['sims'][i]['dir_dict']['mars_lower_vacuum_dir']
            upper_vac = results_class.data(upper_file_loc_vac, I0EXP=I0EXP)
            lower_vac = results_class.data(lower_file_loc_vac, I0EXP=I0EXP)
            upper_vac.get_PEST(facn = facn)
            lower_vac.get_PEST(facn = facn)
            tmp_R, tmp_Z, upper_vac.B1, upper_vac.B2, upper_vac.B3, upper_vac.Bn, upper_vac.BMn, upper_vac.BnPEST = results_class.combine_data(upper_vac, lower_vac, 0)

            upper.B1 = upper.B1 - upper_vac.B1
            upper.B2 = upper.B2 - upper_vac.B2
            upper.B3 = upper.B3 - upper_vac.B3
            upper.Bn = upper.Bn - upper_vac.Bn
            upper.BMn = upper.BMn - upper_vac.BMn
            upper.BnPEST = upper.BnPEST - upper_vac.BnPEST

        print plot_quantity, i, q95_list_arranged[i], plot_quantity_plas_arranged[i], s_surface, mode_list_arranged[i]
        suptitle = '%s key: %d, q95: %.2f, max_amp: %.2f, s_surface: %.2f, m_max: %d'%(plot_quantity, i, q95_list_arranged[i], plot_quantity_plas_arranged[i], s_surface, mode_list_arranged[i])
        include_phase = 1
        fig, ax = pt.subplots(nrows = include_phase + 1, sharex = True, sharey = True)
        if include_phase == 0: ax = [ax]
        if n==2:
            contour_levels = np.linspace(0,5.0,7)
        else:
            contour_levels = np.linspace(0,1.5, 7)
        color_plot = upper.plot_BnPEST(ax[0], n=n, inc_contours = 1, contour_levels = contour_levels)
        if n==2:
            color_plot.set_clim([0,5.])
        else:
            color_plot.set_clim([0,1.5])
        ax[0].set_title(suptitle)
        cbar = pt.colorbar(color_plot, ax = ax[0])
        if include_phase:
            min_phase = -130
            color_plot2 = upper.plot_BnPEST(ax[1], n=n, inc_contours = 0, contour_levels = contour_levels, phase=1, min_phase = min_phase)
            color_plot2.set_clim([min_phase,min_phase+360])
            cbar = pt.colorbar(color_plot2, ax = ax[1])
            cbar.ax.set_ylabel('Phase (deg)')
            ax[1].plot(mode_list_arranged[tmp_loc], s_surface,'bo')
        ax[0].plot(mode_list_arranged[tmp_loc], s_surface,'bo')
        ax[0].plot([-29,29],[s_surface,s_surface], 'b--')
        ax[0].set_xlabel('m')
        ax[0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
        cbar.ax.set_ylabel(r'$\delta B_r$ (G/kA)')
        ax[0].set_xlim([-29,29])
        ax[0].set_ylim([0,1])
        fig_name='/u/haskeysr/tmp_pics_dir2/n%d_%03d_q95_scan.png'%(n,i)
        fig.savefig(fig_name)
        #fig.canvas.draw(); fig.show()
        fig.clf()
        pt.close('all')

        #upper.plot1(suptitle = suptitle,inc_phase=0, clim_value=[0,2], ss_squared = 0, fig_show=0,fig_name='/u/haskeysr/%03d_q95_scan.png'%(i))
