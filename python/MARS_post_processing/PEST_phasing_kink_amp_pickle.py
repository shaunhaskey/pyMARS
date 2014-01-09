'''
Generates plots of 'kink amplification' as a function of phasing
Will also create the files for an animation of plasma, vac, and total 
components in PEST co-ordinates


SH:27Feb2013
Updated to start using dBres_dBkink_funcs module

'''

import dBres_dBkink_funcs as dBres_dBkink
import numpy as np
import matplotlib.pyplot as pt
from scipy.interpolate import griddata
import pickle

#Pickle file that has the results in it
#file_name = '/home/srh112/NAMP_datafiles/mars/shot146382_scan/shot146382_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot146394_3000_q95/shot146394_3000_q95_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/q95_scan_fine/shot146394_3000_q95_fine_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing/equal_spacing_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacingV2/equal_spacingV2_post_processing_PEST.pickle' #FOR THE PAPER
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_146394/equal_spacing_146394_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4/equal_spacing_n4_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4_V2/equal_spacing_n4_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4_V2/equal_spacing_n4_post_processing_PEST.pickle'

#Important analysis parameters
Bn_Li_value = 1.15 #Bn/Li value for the phasing analysis (this is constant)
s_surface = 0.92 # s=s_surface for the dBkink calculation 
phase_machine_ntor = 0 #using machine phasing like Matt's paper
beta_n_axis = 0 #0 for betaN, 1 for betaN/li
color_map = 'jet'

###### PLOT SWITCHES ###################
#for a fixed betaN/li, show a cut across q95 for peeling modes
show_single_cuts = 0
#Show the phase for vac, plas, tot on a color plot
show_phasings_color_plots = 0
#Show the m that is chosen for dBkink
show_max_mode_number = 0
#plot db_res and db_res_ave as a function of bn/li and q95
#This is in the paper
show_db_res_db_res_ave = 1
#plot dBkink as a function of bn/li and q95
#This plot is in the paper
show_dB_kink = 1
#plot dB_kink, amplitude of the vacuum harmonic chosen, and the amplification
show_db_kink_vac_amp_amplification = 0
#plot dBkink vs q95 and phasing for a fixed bn/li with the strength of the 
#vacuum harmonic also shown
dB_kink_phasing_q95 = 0
#plot dBkink vs q95 and phasing for a fixed bn/li with the strength of the 
#vacuum harmonic also shown
dB_kink_phasing_q95_fixed_vac = 0
#Plot showing dBres and dBres_ave vs phasing and q95
#This plot is in the paper
show_dBres_dBres_ave_phasing_q95 = 1
#Plot showing black dots where the simulations were 
show_simulation_locations = 0
fixed_mode = 3
#this is for dBkink calculation for selecting the relevant m to choose from
#(n+reference_offset[1])q+reference_offset[0] < m
reference_offset = [2,0]
#reference to calculate the relevant m
reference_dB_kink = 'plas'
phasing = 0.
phasing = phasing/180.*np.pi

#Extract data from the project dictionary
project_dict = pickle.load(file(file_name,'r'))
n = np.abs(project_dict['details']['MARS_settings']['<<RNTOR>>'])
q95_list, Bn_Li_list, time_list = dBres_dBkink.extract_q95_Bn(project_dict, bn_li = not beta_n_axis)
pmult_list, qmult_list = dBres_dBkink.extract_pmult_qmult(project_dict)
dBres_vac_list_upper, dBres_vac_list_lower, dBres_plas_list_upper, dBres_plas_list_lower = dBres_dBkink.extract_dB_res(project_dict)
dBkink_vac_list_upper, dBkink_vac_list_lower, dBkink_plas_list_upper, dBkink_plas_list_lower, dBkink_tot_list_upper, dBkink_tot_list_lower, mk_list, q_val_list, resonant_close = dBres_dBkink.extract_dB_kink(project_dict, s_surface)

#Apply the single phasing to dBkink
dBkink_vac_list_single = dBres_dBkink.apply_phasing(dBkink_vac_list_upper, dBkink_vac_list_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
dBkink_plas_list_single = dBres_dBkink.apply_phasing(dBkink_plas_list_upper, dBkink_plas_list_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)
dBkink_tot_list_single = dBres_dBkink.apply_phasing(dBkink_tot_list_upper, dBkink_tot_list_lower, phasing, n, phase_machine_ntor = phase_machine_ntor)

#Apply the single phasing to dBres
dBres_vac_single, dBres_plas_single, dBres_ave_vac_single, dBres_ave_plas_single = dBres_dBkink.dB_res_single_phasing(0,phase_machine_ntor, n, dBres_vac_list_upper, dBres_vac_list_lower, dBres_plas_list_upper, dBres_plas_list_lower)

#Get the location of the no-wall limit
x_axis_NW, y_axis_NW, y_axis_NW2 = dBres_dBkink.no_wall_limit(q95_list, Bn_Li_list)

#Obtain the reference for dBkink to find the maximum harmonic, then get the results for vac, plas, tot
if reference_dB_kink=='tot':
    reference = dBres_dBkink.get_reference(dBkink_tot_list_upper, dBkink_tot_list_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)
elif reference_dB_kink=='plas':
    reference = dBres_dBkink.get_reference(dBkink_plas_list_upper, dBkink_plas_list_lower, np.linspace(0,2.*np.pi,100), n, phase_machine_ntor = phase_machine_ntor)

dBkink_vac_single, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, dBkink_vac_list_single, reference_offset = reference_offset)
dBkink_plas_single, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, dBkink_plas_list_single, reference_offset = reference_offset)
dBkink_tot_single, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, dBkink_tot_list_single, reference_offset = reference_offset)

#Calculate the phase of the dBkink result
dBkink_vac_single_phase = np.angle(dBkink_vac_single,deg=True).tolist()
dBkink_plas_single_phase = np.angle(dBkink_plas_single,deg=True).tolist()
dBkink_tot_single_phase = np.angle(dBkink_tot_single,deg=True).tolist()

#Calculate the amplitude of the dBkink result
dBkink_vac_single = np.abs(dBkink_vac_single).tolist()
dBkink_plas_single = np.abs(dBkink_plas_single).tolist()
dBkink_tot_single = np.abs(dBkink_tot_single).tolist()

#Calculate the maximum dBkink for the upper and lower arrays only
#This is so they can be combined later
#Do the same using a fixed harmonic instead of reference
dBkink_value_plas_upper, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, dBkink_plas_list_upper, reference_offset = reference_offset)
dBkink_value_plas_lower, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, dBkink_plas_list_lower, reference_offset = reference_offset)
dBkink_value_vac_upper, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, dBkink_vac_list_upper, reference_offset = reference_offset)
dBkink_value_vac_lower, mode_list, max_loc_list = dBres_dBkink.calculate_db_kink2(mk_list, q_val_list, n, reference, dBkink_vac_list_lower, reference_offset = reference_offset)
dBkink_value_vac_upper_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, dBkink_vac_list_upper, fixed_mode)
dBkink_value_vac_lower_fixed = dBres_dBkink.calculate_db_kink_fixed(mk_list, q_val_list, n, dBkink_vac_list_lower, fixed_mode)

#Create a new grid to interpolate the results onto for plotting
xnew = np.linspace(2.,7.,200)
ynew = np.linspace(0.75,4.5,200)
xnew_grid, ynew_grid = np.meshgrid(xnew,ynew)

#Glueing these together is to make the griddata function happy
q95_Bn_array = np.zeros((len(q95_list),2),dtype=float)
q95_Bn_array[:,0] = q95_list[:]
q95_Bn_array[:,1] = Bn_Li_list[:]
q95_Bn_new = np.zeros((len(xnew),2),dtype=float)
q95_Bn_new[:,0] = xnew[:]
q95_Bn_new[:,1] = ynew[:]

#Interpolate the no-wall limit onto the new grid
y_axis_NW_interp = np.interp(xnew,x_axis_NW, y_axis_NW)
y_axis_NW_interp2 = np.interp(xnew,x_axis_NW, y_axis_NW2)

#Interpolate the dBkink plas, vac and tot data onto a grid - for fixed phasing
dBkink_plas_single_interp = griddata(q95_Bn_array, np.array(dBkink_plas_single), (xnew_grid, ynew_grid),method = 'linear')
dBkink_vac_single_interp = griddata(q95_Bn_array, np.array(dBkink_vac_single), (xnew_grid, ynew_grid), method = 'linear')
dBkink_tot_single_interp = griddata(q95_Bn_array, np.array(dBkink_tot_single), (xnew_grid, ynew_grid), method = 'linear')

#Interpolate the dBres plas, vac onto a grid - for fixed phasing. Also do dBres_ave for vac
dBres_plas_single_interp = griddata(q95_Bn_array, np.array(dBres_plas_single), (xnew_grid, ynew_grid),method = 'linear')
dBres_vac_single_interp = griddata(q95_Bn_array, np.array(dBres_vac_single), (xnew_grid, ynew_grid), method = 'linear')
dBres_ave_vac_single_interp = griddata(q95_Bn_array, np.array(dBres_ave_vac_single), (xnew_grid, ynew_grid), method = 'linear')

#Interpolate the dBkink max mode onto a regular grid
mode_data = griddata(q95_Bn_array, mode_list, (xnew_grid, ynew_grid), method = 'cubic')

#Interpolate the phase of dBkink onto a regular grid
dBkink_plas_single_phase_interp = griddata(q95_Bn_array, dBkink_plas_single_phase, (xnew_grid, ynew_grid), method = 'linear')
dBkink_vac_single_phase_interp = griddata(q95_Bn_array, dBkink_vac_single_phase, (xnew_grid, ynew_grid), method = 'linear')

#Create a mask that removes anything outside the no-wall limit
mask = np.isnan(dBkink_plas_single_interp)
for i in range(0,dBkink_plas_single_interp.shape[1]):
    mask[ynew>((y_axis_NW_interp[i]+y_axis_NW_interp2[i])/2),i]=True

#Start the section on upper-lower phasing vs q95
phasing_array = np.linspace(0,360,360)
q95_single = np.linspace(2.6,6,100)

#Interpolate all the single phasing results onto a regular grid
dBkink_value_plas_lower_interp = griddata(q95_Bn_array, np.array(dBkink_value_plas_lower), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
dBkink_value_plas_upper_interp = griddata(q95_Bn_array, np.array(dBkink_value_plas_upper), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
dBkink_value_vac_lower_interp = griddata(q95_Bn_array, np.array(dBkink_value_vac_lower), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
dBkink_value_vac_upper_interp = griddata(q95_Bn_array, np.array(dBkink_value_vac_upper), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
dBkink_value_vac_lower_interp_fixed = griddata(q95_Bn_array, np.array(dBkink_value_vac_lower_fixed), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
dBkink_value_vac_upper_interp_fixed = griddata(q95_Bn_array, np.array(dBkink_value_vac_upper_fixed), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')

#Create the empty arrays for the phasing dependence calculation
dBkink_plas_phasing = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
dBkink_vac_phasing = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
dBkink_vac_phasing_fixed = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
dBres_vac_phasing = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
dBres_plas_phasing = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
dBres_ave_vac_phasing = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
dBres_ave_plas_phasing = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)

#Calculate the effect of phasing on dBres - note we have to do the calculation for all points
#in q95 betaN/li space, and then interpolate onto the slice that we are interested in to generate
#the plots
#For dBres
for i, curr_phase in enumerate(phasing_array):
    tmp_vac_list, tmp_plas_list, tmp_vac_list2, tmp_plas_list2 = dBres_dBkink.dB_res_single_phasing(curr_phase,phase_machine_ntor, n,dBres_vac_list_upper, dBres_vac_list_lower, dBres_plas_list_upper, dBres_plas_list_lower)
    dBres_vac_phasing[i,:] = griddata(q95_Bn_array, np.array(tmp_vac_list), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
    dBres_plas_phasing[i,:] = griddata(q95_Bn_array, np.array(tmp_plas_list), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
    dBres_ave_vac_phasing[i,:] = griddata(q95_Bn_array, np.array(tmp_vac_list2), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
    dBres_ave_plas_phasing[i,:] = griddata(q95_Bn_array, np.array(tmp_plas_list2), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')

#For dBkink
for i, curr_phase in enumerate(phasing_array):
    phasing = curr_phase/180.*np.pi
    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
    dBkink_plas_phasing[i,:] = np.abs(dBkink_value_plas_upper_interp + dBkink_value_plas_lower_interp*phasor)
    dBkink_vac_phasing[i,:] = np.abs(dBkink_value_vac_upper_interp + dBkink_value_vac_lower_interp*phasor)
    dBkink_vac_phasing_fixed[i,:] = np.abs(dBkink_value_vac_upper_interp_fixed + dBkink_value_vac_lower_interp_fixed*phasor)

#calculate the best fit to the data maxima - have to be careful about wrapping around however...
max_phases = phasing_array[np.argmax(dBres_vac_phasing,axis=0)]
max_phases[max_phases>max_phases[0]]-=360
poly_max_res = np.polyfit(q95_single,max_phases,1)
best_fit_max_res = np.polyval(poly_max_res, q95_single)
best_fit_max_res[best_fit_max_res<0]+=360
best_fit_max_res[best_fit_max_res>360]-=360

#calculate the best fit to the data minima - have to be careful about wrapping around however...
min_phases = phasing_array[np.argmin(dBres_vac_phasing,axis=0)]
min_phases[min_phases>min_phases[0]]-=360
poly_min_res = np.polyfit(q95_single,min_phases,1)
best_fit_min_res = np.polyval(poly_min_res, q95_single)
best_fit_min_res[best_fit_min_res<0]+=360
best_fit_min_res[best_fit_min_res>360]-=360

#print out the vest fits
print '############ best fit min res ################'
print poly_min_res
print '############ best fit max res  ################'
print poly_max_res




#############################PLOTS########################
#for a fixed betaN/li, show a cut across q95 for peeling modes
if show_single_cuts:
    for interp_meth in ['linear', 'cubic']:
        q95_single = np.linspace(3.,5.5,1000)
        plas_data_single = griddata(q95_Bn_array, dBkink_plas_single, (q95_single, q95_single*0.+Bn_Li_value),method = interp_meth)
        vac_data_single = griddata(q95_Bn_array, dBkink_vac_single, (q95_single, q95_single*0.+Bn_Li_value), method = interp_meth)
        tot_data_single = griddata(q95_Bn_array, dBkink_tot_single, (q95_single, q95_single*0.+Bn_Li_value), method = interp_meth)
        mode_data_single = griddata(q95_Bn_array, mode_list, (q95_single, q95_single*0.+Bn_Li_value), method = interp_meth)

        fig,ax = pt.subplots()
        ax.plot(q95_single, plas_data_single, '.-', label='plas')
        ax.plot(q95_single, vac_data_single, '.-', label='vac')
        ax.plot(q95_single, tot_data_single, '.-', label='tot')
        ax.plot(q95_single, mode_data_single, '.-', label='m')
        ax.legend(loc='best')
        ax.set_title('Bn_Li:%.2f, %s interpolation, sqrt(psi)=%.2f'%(Bn_Li_value,interp_meth,s_surface))
        ax.set_xlabel('q95')
        ax.set_ylim([0,14])
        ax.set_ylabel('amplitude or mode number')
        fig.suptitle(file_name,fontsize=8)
        fig.canvas.draw(); fig.show()

#Show the phase for vac, plas, tot on a color plot
if show_phasings_color_plots:
    fig,ax = pt.subplots(nrows = 2,sharex = 1, sharey = 1)
    dBkink_plas_single_phase_interp[dBkink_plas_single_phase_interp<=-40]+=360
    color_fig = ax[0].pcolor(xnew, ynew, np.ma.array(dBkink_plas_single_phase_interp, mask=np.isnan(mode_data)))
    color_fig.set_clim([-40,-40+360])
    pt.colorbar(color_fig, ax=ax[0])
    ax[0].plot(q95_list, Bn_Li_list,'k.')
    ax[0].set_title('Plasma phase')
    dBkink_vac_single_phase_interp[dBkink_plas_single_phase_interp<=-40]+=360
    color_fig = ax[1].pcolor(xnew, ynew, np.ma.array(dBkink_vac_single_phase_interp, mask=np.isnan(mode_data)))
    color_fig.set_clim([-40,-40+360])
    ax[1].set_title('Vacuum phase')
    ax[1].plot(q95_list, Bn_Li_list,'k.')
    pt.colorbar(color_fig, ax=ax[1])
    include_text = 0
    if include_text:
        for ax_tmp in ax:
            for i in range(0,len(q95_list)):
                print 'ehllo'
                ax_tmp.text(q95_list[i], Bn_Li_list[i], str(serial_list[i]), fontsize = 7.5)
    print_phases = 1
    if print_phases:
        for i in range(0,len(q95_list)):
            print 'serial %d : phase %.2f deg'%(serial_list[i],dBkink_plas_single_phase[i])

    fig.canvas.draw(); fig.show()


#Show the m that is chosen for dBkink
if show_max_mode_number:
    fig,ax = pt.subplots()
    color_fig = ax.pcolor(xnew, ynew, np.ma.array(mode_data, mask=np.isnan(mode_data)))
    color_fig.set_clim([5,15])

    pt.colorbar(color_fig, ax=ax)
    ax.plot(q95_list, Bn_Li_list,'k.')
    ax.set_title('Max mode number, sqrt(psi)=%.2f'%(s_surface))
    if beta_n_axis:
        ax.set_ylabel(r'$\beta_N$', fontsize = 14)
    else:
        ax.set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
    ax.set_xlabel(r'$q_{95}$', fontsize = 14)
    fig.suptitle(file_name,fontsize=8)
    fig.canvas.draw(); fig.show()


#plot db_res and db_res_ave as a function of bn/li and q95
#This is in the paper
if show_db_res_db_res_ave:
    fig_JAW, ax_JAW = pt.subplots(nrows=2, sharex =1, sharey=1)
    color_fig_plas_JAW = ax_JAW[0].pcolor(xnew, ynew, np.ma.array(dBres_ave_vac_single_interp, mask=mask),cmap=color_map, rasterized=True)
    cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW[0])
    cbar.ax.set_ylabel(r'$\overline{\delta B}_{res}^{n=%d}$ G/kA'%(n),fontsize=20)
    color_fig_plas_JAW.set_clim([0, 0.55])
    #color_fig_plas_JAW = ax_JAW[1].pcolor(xnew, ynew, np.ma.array(vac_data_res, mask=np.isnan(dBkink_plas_single_interp)),cmap=color_map, rasterized=True)
    color_fig_plas_JAW = ax_JAW[1].pcolor(xnew, ynew, np.ma.array(dBres_vac_single_interp, mask=mask),cmap=color_map, rasterized=True)
    color_fig_plas_JAW.set_clim([0,8])
    cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW[1])
    cbar.ax.set_ylabel(r'$\delta B_{res}^{n=%d}$ G/kA'%(n),fontsize=20)
    ax_JAW[1].set_xlabel(r'$q_{95}$', fontsize = 20)
    if beta_n_axis:
        ax_JAW[0].set_ylabel(r'$\beta_N$', fontsize = 20)
        ax_JAW[1].set_ylabel(r'$\beta_N$', fontsize = 20)
    else:
        ax_JAW[0].set_ylabel(r'$\beta_N / L_i$', fontsize = 20)
        ax_JAW[1].set_ylabel(r'$\beta_N / L_i$', fontsize = 20)
    ax_JAW[0].plot(q95_list, Bn_Li_list,'k.')
    ax_JAW[1].plot(q95_list, Bn_Li_list,'k.')
    ax_JAW[0].fill_between(xnew,  y_axis_NW_interp, y_axis_NW_interp2, facecolor='black', alpha=1)
    ax_JAW[1].fill_between(xnew,  y_axis_NW_interp, y_axis_NW_interp2, facecolor='black', alpha=1)
    ax_JAW[0].set_xlim([2.5, 6.0])
    ax_JAW[0].set_ylim([0.75,4.5])
    fig_JAW.canvas.draw(); fig_JAW.show()

#plot dBkink as a function of bn/li and q95
#This plot is in the paper
if show_dB_kink:
    fig_JAW, ax_JAW = pt.subplots()
    #color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(dBkink_plas_single_interp, mask=np.isnan(dBkink_plas_single_interp)),cmap=color_map, rasterized=True)
    color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(dBkink_tot_single_interp, mask=mask),cmap=color_map, rasterized=True)
    #ax_JAW.contour(xnew, ynew, np.ma.array(dBkink_plas_single_interp, mask=np.isnan(dBkink_plas_single_interp)))
    cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW)
    cbar.ax.set_ylabel(r'$\delta B_{kink}^{n=%d}$ G/kA'%(n),fontsize=20)
    ax_JAW.set_xlabel(r'$q_{95}$', fontsize = 20)

    if beta_n_axis:
        ax_JAW.set_ylabel(r'$\beta_N/\ell_i$', fontsize = 20)
    else:
        ax_JAW.set_ylabel(r'$\beta_N$', fontsize = 20)
    ax_JAW.plot(q95_list, Bn_Li_list,'k.')
    ax_JAW.fill_between(xnew,  y_axis_NW_interp, y_axis_NW_interp2, facecolor='black', alpha=1)
    #ax_JAW.set_title(r'Plasma, $\psi_N=%.2f$'%(s_surface**2), fontsize = 18)
    ax_JAW.set_xlim([2.5, 6.0])
    ax_JAW.set_ylim([0.75,4.5])
    color_fig_plas_JAW.set_clim([0,2.5])
    fig_JAW.canvas.draw(); fig_JAW.show()

#plot dB_kink, amplitude of the vacuum harmonic chosen, and the amplification
if show_db_kink_vac_amp_amplification:
    fig,ax = pt.subplots(nrows = 3,sharex = 1, sharey = 1)
    color_fig_plas = ax[0].pcolor(xnew, ynew, np.ma.array(dBkink_tot_single_interp, mask=np.isnan(dBkink_tot_single_interp)),cmap=color_map)#, cmap = cmap)
    cbar = pt.colorbar(color_fig_plas, ax = ax[0])
    cbar.ax.set_ylabel('G/kA')

    if beta_n_axis:
        ax[0].set_ylabel(r'$\beta_N$', fontsize = 14)
    else:
        ax[0].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
    ax[0].set_title(r'Plasma, $\psi_N=%.2f$'%(s_surface**2))
    ax[0].plot(q95_list, Bn_Li_list,'k.')
    color_fig_vac = ax[1].pcolor(xnew, ynew, np.ma.array(dBkink_vac_single_interp, mask=np.isnan(dBkink_vac_single_interp)),cmap=color_map)#, cmap = cmap)
    color_fig_relative = ax[2].pcolor(xnew, ynew, np.ma.array(dBkink_plas_single_interp/dBkink_vac_single_interp, mask=np.isnan(dBkink_vac_single_interp)),cmap=color_map)#, cmap = cmap)
    color_fig_relative.set_clim([0,10])
    cbar = pt.colorbar(color_fig_relative, ax = ax[2])
    #cbar.ax.set_ylabel('G/kA')
    if plot_quantity=='average':
        color_fig_vac.set_clim([0,0.2])
    elif plot_quantity=='max':
        color_fig_vac.set_clim([0,0.9])

    cbar = pt.colorbar(color_fig_vac, ax = ax[1])
    cbar.ax.set_ylabel('G/kA')
    ax[1].set_title(r'Vacuum, $\psi_N=%.2f$'%(s_surface**2))
    if beta_n_axis:
        ax[1].set_ylabel(r'$\beta_N$', fontsize = 14)
        ax[2].set_ylabel(r'$\beta_N$', fontsize = 14)
    else:
        ax[1].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
        ax[2].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
    ax[-1].set_xlabel(r'$q_{95}$', fontsize = 14)
    ax[1].plot(q95_list, Bn_Li_list,'k.')
    ax[2].plot(q95_list, Bn_Li_list,'k.')
    ax[2].set_title('Plasma Amplification')
    fig.suptitle(file_name,fontsize=8)
    fig.canvas.draw(); fig.show()

#plot dBkink vs q95 and phasing for a fixed bn/li with the strength of the 
#vacuum harmonic also shown
if dB_kink_phasing_q95:
    fig, ax = pt.subplots(nrows =2 , sharex = 1, sharey = 1)
    color_plot = ax[0].pcolor(q95_single, phasing_array, dBkink_totn_phasing, cmap='hot', rasterized=True)
    color_plot2 = ax[1].pcolor(q95_single, phasing_array, dBkink_vac_phasing, cmap='hot', rasterized=True)
    ax[0].plot(q95_single, phasing_array[np.argmax(dBkink_plas_phasing,axis=0)],'k.')
    ax[1].plot(q95_single, phasing_array[np.argmax(dBkink_vac_phasinng,axis=0)],'k.')
    ax[0].plot(q95_single, phasing_array[np.argmin(dBkink_plas_phasing,axis=0)],'b.')
    ax[1].plot(q95_single, phasing_array[np.argmin(dBkink_vac_phasing,axis=0)],'b.')
    ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
    ax[0].set_title('Kink Amplitude - Plasma')
    ax[1].set_title('Kink Amplitude - Vacuum')
    ax[0].set_xlim([np.min(q95_single),np.max(q95_single)])
    ax[0].set_ylim([np.min(phasing_array),np.max(phasing_array)])
    color_plot.set_clim([0, 2])
    color_plot2.set_clim([0, 1])
    cb = pt.colorbar(color_plot, ax = ax[0])
    cb.ax.set_ylabel(r'$\delta B_{kink}^{n=%d}$ G/kA'%(n),fontsize=20)
    cb = pt.colorbar(color_plot2, ax = ax[1])
    cb.ax.set_ylabel(r'$\delta B_{kink}^{n=%d}$ G/kA'%(n),fontsize=20)
    fig.canvas.draw(); fig.show()

#plot dBkink vs q95 and phasing for a fixed bn/li with the strength of the 
#vacuum harmonic also shown
if dB_kink_phasing_q95_fixed_vac:
    fig, ax = pt.subplots(nrows=2,sharex = 1, sharey=1)
    color_plot = ax[0].pcolor(q95_single, phasing_array, dBkink_tot_phasing, cmap='hot', rasterized=True)
    cb = pt.colorbar(color_plot, ax = ax[0])
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
    color_plot.set_clim([0, 2])
    ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
    tmp_xaxis = np.arange(1,10,0.1)
    tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
    ax[0].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
    ax[0].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')
    cb.ax.set_ylabel(r'$\delta B_{kink}^{n=%d}$ G/kA'%(n),fontsize=20)
    color_plot = ax[1].pcolor(q95_single, phasing_array, dBkink_vac_phasing_fixed, cmap='hot', rasterized=True)
    cb = pt.colorbar(color_plot, ax = ax[1])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
    #ax[1].set_title('Kink Amplitude - Vacuum')
    ax[1].set_ylim([np.min(phasing_array),np.max(phasing_array)])
    #color_plot.set_clim([0, 2])
    ax[1].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
    tmp_xaxis = np.arange(1,10,0.1)
    tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
    ax[1].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
    ax[1].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')
    ax[1].set_xlim([2.6, 6])
    cb.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+%d,n=%d}$ G/kA'%(fixed_mode, n),fontsize=20)
    fig.canvas.draw(); fig.show()


#Plot showing dBres and dBres_ave vs phasing and q95
#This plot is in the paper
if show_dBres_dBres_ave_phasing_q95:
    fig, ax = pt.subplots(nrows = 2, sharex = 1, sharey = 1)
    color_plot = ax[0].pcolor(q95_single, phasing_array, dBres_vac_phasing, cmap='hot', rasterized=True)
    ax[0].contour(q95_single,phasing_array, dBres_vac_phasing, colors='white')
    color_plot2 = ax[1].pcolor(q95_single, phasing_array, dBres_ave_vac_phasing, cmap='hot', rasterized=True)
    ax[1].contour(q95_single,phasing_array, dBres_ave_vac_phasing, colors='white')
    color_plot.set_clim([0,10])
    color_plot2.set_clim([0,0.75])
    ax[0].plot(q95_single, best_fit_max_res, 'b.')
    ax[0].plot(q95_single, best_fit_min_res, 'b.')
    ax[1].plot(q95_single, best_fit_max_res, 'b.')
    ax[1].plot(q95_single, best_fit_min_res, 'b.')
    ax[0].set_xlim([2.6, 6])
    ax[0].set_ylim([np.min(phasing_array), np.max(phasing_array)])
    ax[1].set_xlabel(r'$q_{95}$', fontsize=20)
    ax[0].set_title('n=%d, Pitch Resonant Forcing'%(n))
    ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
    cbar = pt.colorbar(color_plot, ax = ax[0])
    cbar.ax.set_ylabel(r'$\delta B_{res}^{n=%d}$ G/kA'%(n),fontsize = 20)
    cbar = pt.colorbar(color_plot2, ax = ax[1])
    cbar.ax.set_ylabel(r'$\overline{\delta B}_{res}^{n=%d}$ G/kA'%(n), fontsize = 20)
    fig.canvas.draw(); fig.show()

#Plot showing black dots where the simulations were 
if show_simulation_locations:
    fig, ax = pt.subplots()
    ax.plot(q95_list, Bn_Li_list,'k.')
    if beta_n_axis:
        ax.set_ylabel(r'$\beta_N$', fontsize = 14)
        ax.set_ylabel(r'$\beta_N$', fontsize = 14)
    else:
        ax.set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
        ax.set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
    ax.set_xlabel(r'$q_{95}$', fontsize=14)
    fig.canvas.draw(); fig.show()


