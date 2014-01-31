import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import dBres_dBkink_funcs as dBres_dBkink
import copy

file_name = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan/shot_142614_rote_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp/shot_142614_rote_scan_w_damp_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp2/shot_142614_rote_scan_w_damp2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp3/shot_142614_rote_scan_w_damp3_post_processing_PEST.pickle'
file_name='/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30/shot_142614_rote_res_scan_30x30_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_test/shot_142614_rote_res_scan_test_post_processing_PEST.pickle'
phasing = 0
n = 3
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [2,0]
sort_name = 'rote_list'

a = dBres_dBkink.test1(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
fig, ax = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax = ax.flatten()
fig2, ax2 = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax2 = ax2.flatten()
phasings_disp = [0,45,90,135,180,225,270,315]
for i in range(len(phasings_disp)):
    a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = ax[i], ax_matrix = ax2[i], clim = [0, 0.015])
fig.canvas.draw(); fig.show()
fig2.canvas.draw(); fig2.show()
a.eta_rote_matrix(phasing = 0)


1/0
def do_everything(file_name, s_surface, phasing,phase_machine_ntor, fixed_harmonic = 5, reference_offset=[2,0], reference_dB_kink='plas',sort_name = 'q95_list'):
    project_dict = pickle.load(file(file_name,'r'))
    key_list = project_dict['sims'].keys()

    n = np.abs(project_dict['details']['MARS_settings']['<<RNTOR>>'])
    q95_list, Bn_Li_list, time_list = dBres_dBkink.extract_q95_Bn(project_dict, bn_li = 1)
    eta_list, rote_list = dBres_dBkink.extract_eta_rote(project_dict)
    res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, res_tot_list_upper, res_tot_list_lower = dBres_dBkink.extract_dB_res(project_dict, return_total = True)

    amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close = dBres_dBkink.extract_dB_kink(project_dict, s_surface)
    fig_harm_select, ax_harm_select = pt.subplots()
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
    ax_harm_select.plot(eta_list,max_loc_list, label='max-harmonic')
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

    #create the sorted lists
    list_of_item_names = ['eta_list', 'rote_list', 'q95_list', 'Bn_Li_list', 'plot_quantity_plas','plot_quantity_vac', 'plot_quantity_tot', 'plot_quantity_plas_phase', 'plot_quantity_vac_phase', 'plot_quantity_tot_phase', 'mode_list', 'time_list', 'key_list', 'resonant_close']

    list_of_items = zip(*[eval(i) for i in list_of_item_names])
    sort_index = list_of_item_names.index(sort_name)
    print sort_index
    tmp = zip(*sorted(list_of_items, key = lambda sort_val:sort_val[sort_index]))
    output_dict2 = {}
    for loc, i in enumerate(list_of_item_names): output_dict2[i+'_arranged'] = tmp[loc]
    for loc, i in enumerate(list_of_item_names): output_dict2[i] = eval(i)

    name_list = ['plot_array_plasma', 'plot_array_vac', 'plot_array_tot', 'plot_array_vac_fixed', 'q95_array', 'phasing_array', 'plot_array_plasma_fixed', 'plot_array_plasma_phase', 'plot_array_vac_phase', 'plot_array_vac_fixed_phase', 'plot_array_plasma_fixed_phase']
    tmp1 = dBres_dBkink.dB_kink_phasing_dependence(q95_list_copy, lower_values_plasma, upper_values_plasma, lower_values_vac, upper_values_vac, lower_values_tot, upper_values_tot, lower_values_vac_fixed, upper_values_vac_fixed, phase_machine_ntor, upper_values_plas_fixed, lower_values_plas_fixed, n, n_phases = 360)
    for name, var in zip(name_list, tmp1): output_dict2[name]=var

    name_list = ['plot_array_vac_res', 'plot_array_plas_res', 'plot_array_vac_res_ave', 'plot_array_plas_res_ave']
    tmp1 = dBres_dBkink.dB_res_phasing_dependence(output_dict2['phasing_array'], output_dict2['q95_array'], res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, phase_machine_ntor, n)
    for name, var in zip(name_list, tmp1): output_dict2[name]=var

    name_list = ['q95_list_copy', 'max_loc_list', 'upper_values_vac_fixed', 'n', 'lower_values_plasma', 'lower_values_vac']
    for name in name_list: output_dict2[name]=eval(name)

    output_dict = output_dict2
    return output_dict

answers = do_everything(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name)
a = dBres_dBkink.test1(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name)


answers = a.output_dict
xaxis = np.array(answers[sort_name+'_arranged'])
dB_kink_fixed_vac = 1
if dB_kink_fixed_vac:
    a.plot_dB_kink_fixed_vac(clim1=[0,0.6],clim2=[0,1.4],xaxis_type='log',xaxis_label='rote')

dB_res_n2_dB_res_sum = 1
if dB_res_n2_dB_res_sum:
    a.dB_res_n2_dB_res_sum(clim1=None,clim2=None,xaxis_type='log',xaxis_label='rote')
    1/0
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    #color_plot = ax[0].pcolor(np.array(answers['eta_list']), answers['phasing_array'], answers['plot_array_vac_res'], cmap='hot', rasterized=True)
    #color_plot = ax[1].pcolor(np.array(answers['eta_list']), answers['phasing_array'], answers['plot_array_plas_res'], cmap='hot', rasterized=True)
    color_plot = ax[0].pcolor(xaxis, answers['phasing_array'], answers['plot_array_vac_res'], cmap='hot', rasterized=True)
    color_plot.set_clim([0,25])
    color_plot2 = ax[1].pcolor(xaxis, answers['phasing_array'], answers['plot_array_plas_res'], cmap='hot', rasterized=True)
    color_plot2.set_clim([0,50])
    #ax[0].contour(np.array(answers['eta_list']),answers['phasing_array'], answers['plot_array_vac_res'], colors='white')
    #color_plot2 = ax[1].pcolor(np.array(answers['eta_list']), answers['phasing_array'], dB_res_sum, cmap='hot', rasterized=True)
    #ax[1].contour(np.array(answers['eta_list']),answers['phasing_array'], dB_res_sum, colors='white')
    #color_plot.set_clim([0,1])
    #color_plot2.set_clim([0,1])
    title_string1 = 'Total Forcing'
    title_string2 = 'Average Forcing'
    ax[0].set_xlim([np.min(answers['eta_list']), np.max(answers['eta_list'])])
    ax[0].set_xlim([np.min(xaxis), np.max(xaxis)])
    ax[0].set_ylim([0,360])

    #ax[0].set_ylim([np.min(answers['phasing_array']), np.max(answers['phasing_array'])])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)
    ax[0].set_title(r'$\delta B_{res}^{n=2}$',fontsize=20)
    ax[1].set_title(r'$\delta B_{res}^{n=2} + \delta B_{res}^{n=4}$',fontsize=20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    #ax[0].set_ylabel('Phasing (deg)')
    #ax[1].set_ylabel('Phasing (deg)')
    fig2, ax2 = pt.subplots(nrows = 2, sharex = True, sharey = True)
    ax2[0].plot(xaxis, answers['plot_array_vac_res'][0,:], '-o',label='0deg res vac')
    ax2[0].plot(xaxis, answers['plot_array_plas_res'][0,:], '-o',label='0deg res plas')
    ax2[0].plot(xaxis, -answers['plot_array_plas_res'][0,:]+answers['plot_array_vac_res'][0,:], '-o',label='0deg total')
    ax2[1].plot(xaxis, answers['plot_array_vac_res'][180,:], '-o', label='180deg vac')
    ax2[1].plot(xaxis, answers['plot_array_plas_res'][180,:], '-o',label='180deg plas')
    ax2[1].plot(xaxis, -answers['plot_array_plas_res'][180,:]+answers['plot_array_vac_res'][180,:], '-o', label='180deg total')
    ax2[1].set_xscale('log')
    ax2[0].set_xscale('log')

    ax2[0].legend(loc='best')
    #ax2.plot(np.array(answers['eta_list']), answers['plot_array_total_res'][0,:], '-o')
    fig2.canvas.draw();fig2.show()
    cbar = pt.colorbar(color_plot, ax = ax[0])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel('G/kA',fontsize = 16)
    fig.canvas.draw(); fig.show()

1/0

res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower = dBresdBkink.extract_dB_res(rot_pickle)
amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close = dBresdBkink.extract_dB_kink(rot_pickle, s_surface)

amps_vac_comp = dBresdBkink.apply_phasing(amps_vac_comp_upper, amps_vac_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
amps_plas_comp = dBresdBkink.apply_phasing(amps_plas_comp_upper, amps_plas_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
amps_tot_comp = dBresdBkink.apply_phasing(amps_tot_comp_upper, amps_tot_comp_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
if reference_dB_kink=='plas':
    reference = dBres_dBkink.get_reference(amps_plas_comp_upper, amps_plas_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)
elif reference_dB_kink=='tot':
    reference = dBres_dBkink.get_reference(amps_tot_comp_upper, amps_tot_comp_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)

for i in rot_pickle['sims'].keys():
    rote_list.append(rot_pickle['sims'][i]['MARS_settings']['<<ROTE>>'])
    res_list.append(rot_pickle['sims'][i]['MARS_settings']['<<ETA>>'])
    total_res.append(rot_pickle['sims'][i]['responses']['total_resonant_response_upper_integral']+rot_pickle['sims'][i]['responses']['total_resonant_response_lower_integral'])
    vac_res.append(rot_pickle['sims'][i]['responses']['vacuum_resonant_response_upper_integral']+rot_pickle['sims'][i]['responses']['vacuum_resonant_response_lower_integral'])
    plas_res.append(total_res[-1] - vac_res[-1])

#for i in res_pickle['sims'].keys():
#    res_list.append(res_pickle['sims'][i]['MARS_settings']['<<ETA>>'])
#    total_res.append(res_pickle['sims'][1]['responses']['total_resonant_response_upper_integral']+res_pickle['sims'][1]['responses']['total_resonant_response_lower_integral'])
fig, ax = pt.subplots()
ax.plot(res_list, total_res, '-o', label='totl')
ax.plot(res_list, vac_res, '-o', label='vac')
ax.plot(res_list, plas_res, '-o', label='plas')
ax.legend(loc='best')
fig.canvas.draw(); fig.show()
