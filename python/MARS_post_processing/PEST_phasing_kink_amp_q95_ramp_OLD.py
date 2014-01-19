'''
Generates plots of 'kink amplification' as a function of phasing
Will also create the files for an animation of plasma, vac, and total 
components in PEST co-ordinates

'''

import results_class, copy
import RZfuncs
import numpy as np
import matplotlib.pyplot as pt
import PythonMARS_funcs as pyMARS
from scipy.interpolate import griddata
import pickle
import matplotlib.cm as cm
#file_name = '/home/srh112/NAMP_datafiles/mars/shot146382_scan/shot146382_scan_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot146394_3000_q95/shot146394_3000_q95_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/q95_scan/q95_scan_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/detailed_q95_scan3_n4/detailed_q95_scan3_n4_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/detailed_q95_scan3_n4/detailed_q95_scan3_n4_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'
#file_name = '/u/haskeysr/mars/detailed_q95_scan3/detailed_q95_scan3_post_processing_PEST.pickle'

N = 6; n = 4
I = np.array([1.,-1.,0.,1,-1.,0.])
facn = 1.0; s_surface = 0.97
ylim = [0,1.4]
#phasing_range = [-180.,180.]
#phasing_range = [0.,360.]
phase_machine_ntor = 0
make_animations = 0
include_discrete_comparison = 0
seperate_res_plot = 0
include_vert_lines = 0
plot_text = 1

plot_type = 'best_harmonic'
#plot_type = 'normalised'
#plot_type = 'normalised_average'
#plot_type = 'standard_average'

project_dict = pickle.load(file(file_name,'r'))
phasing = 0.
#phasing = np.arange(0.,360.,1)
print phasing
phasing = phasing/180.*np.pi
print phasing
q95_list = []; Bn_Li_list = []
amps_vac_comp = []; amps_tot_comp = []; amps_plas_comp=[]; mk_list = []; time_list = []; q_val_list = []
amps_plas_comp_upper = []; amps_plas_comp_lower = []
amps_vac_comp_upper = []; amps_vac_comp_lower = []
key_list = project_dict['sims'].keys()
resonant_close = []
#extract the relevant data from the pickle file and put it into lists
res_vac_list_upper = []; res_vac_list_lower = []
res_plas_list_upper = []; res_plas_list_lower = []


def extract_q95_Bn(tmp_dict, bn_li = 1):
    q95_list = []; Bn_Li_list = []
    for i in tmp_dict['sims'].keys()
        q95_list.append(tmp_dict['sims'][i]['Q95'])
        if bn_li == 1:
            Bn_Li_list.append(tmp_dict['sims'][i]['BETAN']/tmp_dict['sims'][i]['LI'])
        else:
            Bn_Li_list.append(tmp_dict['sims'][i]['BETAN']
    return q95_list, Bn_Li_list

q95_list, Bn_Li_list = extract_q95_Bn(project_dict, bn_li = 1)
print 'hello!'

for i in key_list:
    q95_list.append(project_dict['sims'][i]['Q95'])
    #q95_list.append((project_dict['sims'][i]['Q95']+2.*project_dict['sims'][i]['QMAX'])/3.)
    #q95_list.append(project_dict['sims'][i]['QMAX'])
    Bn_Li_list.append(project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI'])

    #tmp values
    relevant_values_upper_tot = project_dict['sims'][i]['responses'][str(s_surface)]['total_kink_response_upper']
    relevant_values_lower_tot = project_dict['sims'][i]['responses'][str(s_surface)]['total_kink_response_lower']
    relevant_values_upper_vac = project_dict['sims'][i]['responses'][str(s_surface)]['vacuum_kink_response_upper']
    relevant_values_lower_vac = project_dict['sims'][i]['responses'][str(s_surface)]['vacuum_kink_response_lower']

    mk_list.append(project_dict['sims'][i]['responses'][str(s_surface)]['mk'])
    q_val_list.append(project_dict['sims'][i]['responses'][str(s_surface)]['q_val'])

    #tmp values
    upper_tot_res = np.array(project_dict['sims'][i]['responses']['total_resonant_response_upper'])
    lower_tot_res = np.array(project_dict['sims'][i]['responses']['total_resonant_response_lower'])
    upper_vac_res = np.array(project_dict['sims'][i]['responses']['vacuum_resonant_response_upper'])
    lower_vac_res = np.array(project_dict['sims'][i]['responses']['vacuum_resonant_response_lower'])

    time_list.append(project_dict['sims'][i]['shot_time'])
    resonant_close.append(np.min(np.abs(project_dict['sims'][i]['responses']['resonant_response_sq']-s_surface)))

    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))

    amps_vac_comp.append(relevant_values_upper_vac + relevant_values_lower_vac*phasor)
    amps_tot_comp.append(relevant_values_upper_tot + relevant_values_lower_tot*phasor)
    amps_plas_comp.append(relevant_values_upper_tot-relevant_values_upper_vac + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)

    amps_plas_comp_upper.append(relevant_values_upper_tot-relevant_values_upper_vac)
    amps_plas_comp_lower.append(relevant_values_lower_tot-relevant_values_lower_vac)
    amps_vac_comp_upper.append(relevant_values_upper_vac)
    amps_vac_comp_lower.append(relevant_values_lower_vac)

    res_vac_list_upper.append(upper_vac_res)
    res_vac_list_lower.append(lower_vac_res)
    res_plas_list_upper.append(upper_tot_res - upper_vac_res)
    res_plas_list_lower.append(lower_tot_res - lower_vac_res)




plot_quantity_vac=[]; plot_quantity_plas=[]; plot_quantity_tot=[];
plot_quantity_vac_phase=[]; plot_quantity_plas_phase=[]; plot_quantity_tot_phase=[];

#Get the plot quantities out of the lists from the previous section
plot_quantity = 'max'
max_based_on_total = 1
max_loc_list = []; mode_list = []
upper_values_plasma = []; lower_values_plasma = []
upper_values_vac = []; lower_values_vac = []
upper_values_vac_fixed = []; lower_values_vac_fixed = []
for i in range(0,len(amps_vac_comp)):
    if plot_quantity == 'average':
        plot_quantity_vac.append(np.sum(np.abs(amps_vac_comp[i])**2)/len(amps_vac_comp[i]))
        plot_quantity_plas.append(np.sum(np.abs(amps_plas_comp[i])**2)/len(amps_vac_comp[i]))
        plot_quantity_tot.append(np.sum(np.abs(amps_tot_comp[i])**2)/len(amps_vac_comp[i]))
        mode_list.append(np.average(mk_list[i][:]))

        plot_quantity_vac_phase.append(0)
        plot_quantity_plas_phase.append(0)
        plot_quantity_tot_phase.append(0)

    elif plot_quantity == 'max':
        allowable_indices = np.array(mk_list[i])>(np.array(q_val_list[i])*(n+0))
        if max_based_on_total:
            maximum_val = np.max(np.abs(amps_tot_comp[i])[allowable_indices])
            max_loc = np.argmin(np.abs(np.abs(amps_tot_comp[i]) - maximum_val))
            print i, maximum_val, max_loc, np.abs(amps_tot_comp[i][max_loc]), q_val_list[i]*(n+0.5), mk_list[i][max_loc]
        else:
            maximum_val = np.max(np.abs(amps_plas_comp[i])[allowable_indices])
            max_loc = np.argmin(np.abs(np.abs(amps_plas_comp[i]) - maximum_val))
            print i, maximum_val, max_loc, np.abs(amps_plas_comp[i][max_loc]), q_val_list[i]*(n+0.5), mk_list[i][max_loc]
        max_loc_list.append(max_loc)

        mode_list.append(mk_list[i][max_loc])
        plot_quantity_vac.append(np.abs(amps_vac_comp[i][max_loc]))
        plot_quantity_plas.append(np.abs(amps_plas_comp[i][max_loc]))
        plot_quantity_tot.append(np.abs(amps_tot_comp[i][max_loc]))

        mode_list.append(mk_list[i][max_loc])
        plot_quantity_vac_phase.append(np.angle(amps_vac_comp[i][max_loc], deg = True))
        plot_quantity_plas_phase.append(np.angle(amps_plas_comp[i][max_loc], deg= True))
        plot_quantity_tot_phase.append(np.angle(amps_tot_comp[i][max_loc], deg = True))

        upper_values_plasma.append(amps_plas_comp_upper[i][max_loc])
        lower_values_plasma.append(amps_plas_comp_lower[i][max_loc])
        upper_values_vac.append(amps_vac_comp_upper[i][max_loc])
        lower_values_vac.append(amps_vac_comp_lower[i][max_loc])

        fixed_loc = np.min([np.argmin(np.abs(mk_list[i] - q_val_list[i]*n)) + 5, len(amps_vac_comp_upper[i])-1])
        upper_values_vac_fixed.append(amps_vac_comp_upper[i][fixed_loc])
        lower_values_vac_fixed.append(amps_vac_comp_lower[i][fixed_loc])


#arange the answers based on q95 value, there is a better way to do this.... using zip
plot_quantity_plas_arranged = []; plot_quantity_tot_arranged = []; plot_quantity_vac_arranged = []
plot_quantity_plas_phase_arranged = []; plot_quantity_tot_phase_arranged = []; plot_quantity_vac_phase_arranged = []
q95_list_arranged = []; Bn_Li_list_arranged = []; mode_list_arranged = []
time_list_arranged = []
q95_list_copy = copy.deepcopy(q95_list)
Bn_Li_list_copy = copy.deepcopy(Bn_Li_list)
key_list_arranged = []; resonant_close_arranged = []
for i in range(0,len(q95_list)):
    cur_loc = np.argmin(q95_list)
    q95_list_arranged.append(q95_list.pop(cur_loc))
    Bn_Li_list_arranged.append(Bn_Li_list.pop(cur_loc))
    plot_quantity_plas_arranged.append(plot_quantity_plas.pop(cur_loc))
    plot_quantity_vac_arranged.append(plot_quantity_vac.pop(cur_loc))
    plot_quantity_tot_arranged.append(plot_quantity_tot.pop(cur_loc))
    plot_quantity_plas_phase_arranged.append(plot_quantity_plas_phase.pop(cur_loc))
    plot_quantity_vac_phase_arranged.append(plot_quantity_vac_phase.pop(cur_loc))
    plot_quantity_tot_phase_arranged.append(plot_quantity_tot_phase.pop(cur_loc))
    mode_list_arranged.append(mode_list.pop(cur_loc))
    time_list_arranged.append(time_list.pop(cur_loc))
    key_list_arranged.append(key_list.pop(cur_loc))
    resonant_close_arranged.append(resonant_close.pop(cur_loc))

fig_single, ax_single = pt.subplots(nrows = 3)
ax_single[0].plot(q95_list_arranged, resonant_close_arranged, '.-')
ax_single[1].plot(q95_list_arranged, plot_quantity_plas_arranged, 'o-', label = 'plasma')
ax_single[2].plot(q95_list_arranged, plot_quantity_plas_phase_arranged, 'o-', label = 'plasma')
fig_single.canvas.draw(); fig_single.show()

#amplitude and phase versus q95 and time
fig, ax = pt.subplots(ncols = 2, nrows = 2)
ax[0,0].plot(q95_list_arranged, plot_quantity_plas_arranged, 'o-', label = 'plasma')
ax[0,0].plot(q95_list_arranged, plot_quantity_vac_arranged, 'o-', label = 'vacuum')
ax[0,0].plot(q95_list_arranged, plot_quantity_tot_arranged, 'o-',label = 'total')
ax[0,0].plot(q95_list_arranged, np.array(mode_list_arranged)/2., 'x-',label = 'm/2')
if plot_text ==1:
    for i in range(0,len(key_list_arranged)):
        ax[0,0].text(q95_list_arranged[i], plot_quantity_plas_arranged[i], str(key_list_arranged[i]), fontsize = 8.5)


leg = ax[0,0].legend(loc='best', fancybox=True)
leg.get_frame().set_alpha(0.5)
ax[0,0].set_ylabel('mode amplitude')
ax[0,0].set_title('sqrt(psi)=%.2f'%(s_surface))
ax[0,1].plot(time_list_arranged, plot_quantity_plas_arranged, 'o', label = 'plasma')
ax[0,1].plot(time_list_arranged, plot_quantity_vac_arranged, 'o', label = 'vacuum')
ax[0,1].plot(time_list_arranged, plot_quantity_tot_arranged, 'o',label = 'total')


ax[1,0].plot(q95_list_arranged, plot_quantity_plas_phase_arranged, 'o-', label = 'plasma')
ax[1,0].plot(q95_list_arranged, plot_quantity_vac_phase_arranged, 'o-', label = 'vacuum')
ax[1,0].plot(q95_list_arranged, plot_quantity_tot_phase_arranged, 'o-',label = 'total')
leg = ax[1,0].legend(loc='best', fancybox = True)
leg.get_frame().set_alpha(0.5)
ax[1,0].set_xlabel('q95')
ax[1,0].set_ylabel('phase (deg)')
ax[1,1].plot(time_list_arranged, plot_quantity_plas_phase_arranged, 'o', label = 'plasma')
ax[1,1].plot(time_list_arranged, plot_quantity_vac_phase_arranged, 'o', label = 'vacuum')
ax[1,1].plot(time_list_arranged, plot_quantity_tot_phase_arranged, 'o',label = 'total')
ax[1,1].set_xlabel('time (ms)')
fig.suptitle(file_name,fontsize=8)
fig.canvas.draw()
fig.show()

#plot q95 versus Bn/Li
fig, ax = pt.subplots(ncols = 2, sharey=1)
ax[0].plot(q95_list_arranged, Bn_Li_list_arranged, 'o-')
ax[0].set_xlabel('q95')
ax[0].set_ylabel('Bn/Li')
ax[0].set_ylim([0,3.5])
ax[1].plot(time_list_arranged, Bn_Li_list_arranged, 'o')
ax[1].set_xlabel('time (ms)')
fig.suptitle(file_name,fontsize=8)
fig.canvas.draw(); fig.show()


#Work on the phasing as a function of q95
phasing_array = np.linspace(0,360,360)
fig, ax = pt.subplots(nrows = 2, sharex = 1, sharey = 1)
q95_array = np.array(q95_list_copy)

rel_lower_vals_plasma = np.array(lower_values_plasma)
rel_upper_vals_plasma = np.array(upper_values_plasma)
rel_lower_vals_vac =  np.array(lower_values_vac)
rel_upper_vals_vac =  np.array(upper_values_vac)

rel_lower_vals_vac_fixed =  np.array(lower_values_vac_fixed)
rel_upper_vals_vac_fixed =  np.array(upper_values_vac_fixed)

plot_array_plasma = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
plot_array_vac = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
plot_array_vac_fixed = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)


for i, curr_phase in enumerate(phasing_array):
    phasing = curr_phase/180.*np.pi
    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
    plot_array_plasma[i,:] = np.abs(rel_upper_vals_plasma + rel_lower_vals_plasma*phasor)
    plot_array_vac[i,:] = np.abs(rel_upper_vals_vac + rel_lower_vals_vac*phasor)
    plot_array_vac_fixed[i,:] = np.abs(rel_upper_vals_vac_fixed + rel_lower_vals_vac_fixed*phasor)

color_plot = ax[0].pcolor(q95_array, phasing_array, plot_array_plasma, cmap='hot')
color_plot.set_clim([0, 3])
color_plot2 = ax[1].pcolor(q95_array, phasing_array, plot_array_vac, cmap='hot')
#color_plot2.set_clim([0.002, 3])
ax[0].plot(q95_array, phasing_array[np.argmax(plot_array_plasma,axis=0)],'k.')
ax[1].plot(q95_array, phasing_array[np.argmax(plot_array_vac,axis=0)],'k.')
ax[0].plot(q95_array, phasing_array[np.argmin(plot_array_plasma,axis=0)],'b.')
ax[1].plot(q95_array, phasing_array[np.argmin(plot_array_vac,axis=0)],'b.')
#color_plot.set_clim()
#ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
ax[0].set_ylabel('Phasing (deg)')
ax[1].set_ylabel('Phasing (deg)')
ax[0].set_title('Kink Amplitude - Plasma')
ax[1].set_title('Kink Amplitude - Vacuum')
ax[0].set_xlim([2.5,6.0])
ax[0].set_ylim([np.min(phasing_array), np.max(phasing_array)])
ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
tmp_xaxis = np.arange(1,10,0.1)
tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
cbar = pt.colorbar(color_plot, ax = ax[0])
ax[1].set_xlabel(r'$q_{95}$', fontsize = 20)
cbar.ax.set_ylabel(r'$\delta B_{kink}$ G/kA',fontsize=20)
cbar = pt.colorbar(color_plot2, ax = ax[1])
cbar.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+5,n=%d}$ G/kA'%(n,),fontsize=20)
fig.canvas.draw(); fig.show()


#########################
#Plot for the paper
fig, ax = pt.subplots(nrows = 2, sharex =1, sharey = 1)
color_plot = ax[0].pcolor(q95_array, phasing_array, plot_array_plasma, cmap='hot', rasterized= 'True')
color_plot.set_clim([0, 3])
color_plot2 = ax[1].pcolor(q95_array, phasing_array, plot_array_vac_fixed, cmap='hot', rasterized = 'True')
#color_plot2.set_clim([0.002, 3])
ax[0].plot(q95_array, phasing_array[np.argmax(plot_array_plasma,axis=0)],'k.')
#ax[1].plot(q95_array, phasing_array[np.argmax(plot_array_vac,axis=0)],'k.')
ax[0].plot(q95_array, phasing_array[np.argmin(plot_array_plasma,axis=0)],'b.')
#ax[1].plot(q95_array, phasing_array[np.argmin(plot_array_vac,axis=0)],'b.')
#color_plot.set_clim()
#ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
#ax[0].set_ylabel('Phasing (deg)')
#ax[1].set_ylabel('Phasing (deg)')

ax[0].set_title('Kink Amplitude - Plasma')
ax[1].set_title('Kink Amplitude - Vacuum')
ax[0].set_xlim([2.5,6.0])
ax[0].set_ylim([np.min(phasing_array), np.max(phasing_array)])
ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
ax[1].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
tmp_xaxis = np.arange(1,10,0.1)
tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
ax[0].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
ax[0].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')
ax[1].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
ax[1].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')

cbar = pt.colorbar(color_plot, ax = ax[0])
ax[1].set_xlabel(r'$q_{95}$', fontsize = 20)
cbar.ax.set_ylabel(r'$\delta B_{kink}$ G/kA',fontsize=20)
cbar = pt.colorbar(color_plot2, ax = ax[1])
cbar.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+5,n=2}$ G/kA',fontsize=20)
fig.canvas.draw(); fig.show()





plot_array_vac_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
plot_array_plas_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
plot_array_vac_res2 = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
plot_array_plas_res2 = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)

for i, curr_phase in enumerate(phasing_array):
    print 'phase :', curr_phase
    phasor = (np.cos(curr_phase/180.*np.pi)+1j*np.sin(curr_phase/180.*np.pi))
    tmp_vac_list = []; tmp_plas_list = []
    tmp_vac_list2 = []; tmp_plas_list2 = []
    for ii in range(0,len(res_vac_list_upper)):
        divisor = len(res_vac_list_upper[ii])
        tmp_vac_list2.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor))/divisor)
        tmp_plas_list2.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor))/divisor)
        tmp_vac_list.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor)))
        tmp_plas_list.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor)))

    plot_array_vac_res[i,:] = tmp_vac_list
    plot_array_plas_res[i,:] = tmp_plas_list
    plot_array_vac_res2[i,:] = tmp_vac_list2
    plot_array_plas_res2[i,:] = tmp_plas_list2



fig, ax = pt.subplots(nrows = 2, sharex = 1, sharey = 1); #ax = [ax]#nrows = 2, sharex = 1, sharey = 1)
color_plot = ax[0].pcolor(q95_array, phasing_array, plot_array_vac_res, cmap='hot', rasterized=True)
ax[0].contour(q95_array,phasing_array, plot_array_vac_res, colors='white')
color_plot2 = ax[1].pcolor(q95_array, phasing_array, plot_array_vac_res2, cmap='hot', rasterized=True)
ax[1].contour(q95_array,phasing_array, plot_array_vac_res2, colors='white')
color_plot.set_clim([0,10])
color_plot2.set_clim([0,0.75])

title_string1 = 'Total Forcing'
title_string2 = 'Average Forcing'
    
ax[0].set_xlim([2.6, 6])
ax[0].set_ylim([np.min(phasing_array), np.max(phasing_array)])
ax[1].set_xlabel(r'$q_{95}$', fontsize=14)

ax[0].set_title('n=%d, Pitch Resonant Forcing'%(n))
ax[0].set_ylabel('Phasing (deg)')
ax[1].set_ylabel('Phasing (deg)')

cbar = pt.colorbar(color_plot, ax = ax[0])
cbar.ax.set_ylabel('%s G/kA'%(title_string1))
cbar = pt.colorbar(color_plot2, ax = ax[1])
cbar.ax.set_ylabel('%s G/kA'%(title_string2))
#color_plot2 = ax[1].pcolor(q95_single, phasing_array, plot_array_plas_res, cmap='hot', rasterized=True)
#color_plot2.set_clim([0,10])
#pt.colorbar(color_plot2, ax = ax[1])
fig.canvas.draw(); fig.show()



fig, ax = pt.subplots();ax = [ax]
single_q95_value = 3.5
tmp_loc = np.argmin(np.abs(q95_array-single_q95_value))
single_db_res = plot_array_vac_res[:,tmp_loc]
single_db_res_ave = plot_array_vac_res2[:,tmp_loc]
single_db_kink = plot_array_plasma[:,tmp_loc]
#ax[0].plot(phasing_array,single_db_res_ave/np.max(single_db_res_ave), '--', label='db_res_ave')
ax[0].plot(phasing_array,single_db_res/np.max(single_db_res), '-',label=r'$\delta B_{res}$')
ax[0].plot(phasing_array,single_db_kink/np.max(single_db_kink), '-.',label=r'$\delta B_{kink}$')
ax[0].legend(loc='best'); ax[0].grid()
ax[0].set_xlabel(r'$\Delta \phi_{ul}$ (deg) or time (1/180 s)', fontsize = 20)
ax[0].set_ylabel('Normalised Amplitude')
ax[0].set_xlim([0,360]); ax[0].set_ylim([0,1])
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
        fig, ax = pt.subplots(nrows = include_phase + 1, sharex = 1, sharey = 1)
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


#plot of dBkink2 and dBkink2 + dBkink4 
#plot of db_res n=2 and db_res n=2 + db_res n=4
dB_res_n2_dB_res_sum = 1
if dB_res_n2_dB_res_sum:
    fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
    color_plot = ax[0].pcolor(answers['res_array'][truth_array], answers['phasing_array'], answers['plot_array_vac_res'][:,truth_array], cmap='hot', rasterized=True)
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
