#This calculates the phasing between a pickup array and an I-coil array at a given frequency
#The various vacuum couplings are calculated using a hdf5 containing the details
#Many of the worker functions are in magnetics_generic_funcs.py
#Mainly used for comparison with MARS results
#Shaun Haskey 22/06/2012
#
#Shaun Haskey 10/07/2012
#magnetics details has been plumed in
#option to read in a pickled file containing the data
#svd scan using lots of toroidal harmonics

#shot details
#shot 146382:
#1600-2250 - no I-coils
#2400-5100 - I-coils on and q95 dropping

#shot 146388:
#1700-2100 - no I-coils
#2250-2500 static coil on
#3000-5000 - sinusoidal I-coils steady q95 and betaN

#shot 146398
#2000-2400 - no I-coils
#2400-3000 - static I-coils
#3000-4400 - sinusoidal I-coils steady q95, increasing betaN

import h5py, pickle, copy, os
import time as timer_module
import numpy as num
try:
    import data
except:
    print 'data module not available'
    print 'expect an error if not using pickled data'

import magnetics_generic_funcs as mag_funcs
import magnetics_details as mag_details
import matplotlib.pyplot as pt

sensor_array_name = 'Bp_probes_R0_working'
#sensor_array_name = 'Br_probes_R0_working'
sensor_array_name = 'Br_probes_R0'
sensor_array_name = 'Br_probes_R2'
icoil_name =  'I_coils_upper'
bulk_read = 0
jeremy_idl_compare = 0
perform_svd_scan = 1
i_coil_freq = 10.

use_pickled_signals = 1
write_pickle = 0

sample_rate = 1000 #Hz
n_range = [3,15]
#n_list = [0,2,4]
n_list = [0,1,2,3,4]
remove_mean = 0
remove_trend = 1
timer_start = timer_module.time()
nd = 16
coil_name_list_upper = ['IU30','IU90','IU150','IU210','IU270','IU330','IL30','IL90','IL150','IL210','IL270','IL330']

I_coil_plots = 0
include_svd_plot_pickups = 1
include_svd_plot_coils = 0

#These give good SNR
shot = 146382;start_time = 2500; end_time = 4500 # this gives good SNR!
shot = 146388;start_time = 3020; end_time = 4400 #This gives a pretty good signal
shot = 146392;start_time = 3080; end_time = 4100 #gives pretty good signal
shot = 146397;start_time = 3100; end_time = 4300 #gives pretty good signal
shot = 146398;start_time = 3100; end_time = 3900 #gives pretty good signal
#shot = 148765;start_time = 2800; end_time = 4800 #gives pretty good signal

#shot = 146382;start_time = 2500; end_time = 4500 # this gives good SNR!
#shot = 146382;start_time = 3200; end_time = 3600
#shot = 146382;start_time = 3000; end_time = 3600
#shot = 146388;start_time = 3020; end_time = 4400 #This gives a pretty good signal
#shot = 146388;start_time = 3100; end_time = 3600
#shot = 146392;start_time = 3400; end_time = 3800#3800#4300
#shot = 146392;start_time = 3080; end_time = 4100 #gives pretty good signal
#shot = 146397;start_time = 3400; end_time = 3800#3800#4300
#shot = 146397;start_time = 3100; end_time = 4300 #gives pretty good signal
#shot = 146398;start_time = 3100; end_time = 3900 #gives pretty good signal
#shot = 146398;start_time = 3400; end_time = 3800#3800#4300

#single_transfers = '/u/hansonjm/var/data/transfers/tf2012_single.h5'
single_transfers = '/home/srh112/NAMP_datafiles/tf2012_single.h5'
vac_coupling = h5py.File(single_transfers, 'r')

#create the interpolated time axis
interp_time = num.arange(start_time, end_time, (1./sample_rate)*1000, dtype=float)

overall_fig, overall_ax = pt.subplots(nrows=3, sharex=1)
overall_fft_fig, overall_fft_ax = pt.subplots(nrows=3)

#file_name = 'pickled_signals4.pickle'
file_name = '/home/srh112/NAMP_datafiles/pickled_signals4.pickle'
if os.path.exists(file_name) & use_pickled_signals==1:
    print 'reading in pickled file'
    file_obj = file(file_name, 'r')
    existing_signals = pickle.load(file_obj)
    file_obj.close()
else:
    print 'not using existing pickled signals'
    existing_signals = {}

sensor_dict = {}
sensor_dict['name']=sensor_array_name
sensor_dict['pickup_names']=mag_details.pickups.names(sensor_array_name)
sensor_dict['phi'] = mag_details.pickups.phi(sensor_array_name)

icoil_dict = {}
icoil_dict['name']= icoil_name
icoil_dict['pickup_names']=mag_details.coils.names(icoil_name)
icoil_dict['phi'] = mag_details.coils.phi(icoil_name)

if bulk_read == 1:
    array_list = ['Bp_probes_R0_working', 'Br_probes_R0', 'Br_probes_R1', 'Br_probes_R2']
    #[MPI66M_array, UISL_array, LISL_array, MISL_array]
    for tmp_array_name in array_list:
        tmp_dict = {}
        tmp_dict['name'] = tmp_array_name
        tmp_dict['pickup_names'] = mag_details.pickups.names(tmp_array_name)
        tmp_dict['phi'] = mag_details.pickups.phi(tmp_array_name)
        tmp_dict = mag_funcs.extract_data(tmp_dict, shot, interp_time, vac_coupling, coil_name_list_upper, sample_rate, i_coil_freq, existing_signals, plotting=1, ax_ylim = [0,1.e-3], remove_mean = remove_mean, remove_trend = remove_trend)
        del tmp_dict

sensor_dict = mag_funcs.extract_data(sensor_dict, shot, interp_time, vac_coupling, coil_name_list_upper, sample_rate, i_coil_freq, existing_signals, plotting=1, ax_ylim = [0,20], remove_mean = remove_mean, remove_trend = remove_trend)
icoil_dict = mag_funcs.extract_data(icoil_dict, shot, interp_time, vac_coupling, [], sample_rate, i_coil_freq, existing_signals, plotting=I_coil_plots, ax_ylim = None, remove_mean = 0, remove_trend = 0, plot_all=1)

if jeremy_idl_compare == 1:
    raw_sensor_data = num.loadtxt('sensorData.csv',delimiter=',')
    fig_1, ax_1 = pt.subplots(nrows=3,ncols=3,sharex=1)
    ax_1 = ax_1.flatten()
    for i in range(0,len(sensor_dict['pickup_names'])):
        ax_1[i].plot(raw_sensor_data[i,:],'k')
        ax_1[i].plot(sensor_dict['signals'][i],'b')
    ax_1[0].set_title('hello')
    comp_sensor_data = num.loadtxt('sensorDataComp.csv',delimiter=',')
    for i in range(0,len(sensor_dict['pickup_names'])):
        ax_1[i].plot(comp_sensor_data[i,:],'r')
        ax_1[i].plot(sensor_dict['comp_signals'][i],'y')
    fig_1.canvas.draw();fig_1.show()

if perform_svd_scan == 1:
    title_tmp = 'n=2 amplitude (pickup)'
    mag_funcs.svd_scan(n_range, sensor_dict, title_tmp, include_plot = include_svd_plot_pickups)
    title_tmp = 'n=2 amplitude (icoil)'
    mag_funcs.svd_scan(n_range, icoil_dict, title_tmp, include_plot = include_svd_plot_coils)

for i in ['total', 'vac', 'plasma']:
    #perform the SVD analysis
    print '===========sensor ', i, '================'
    sensor_dict[i+'_alpha'], resid = mag_funcs.calculate_alpha(sensor_dict['phi'],n_list,sensor_dict[i+'_results_list'])
    print '===========icoil ', i, '================'
    icoil_dict[i+'_alpha'], resid = mag_funcs.calculate_alpha(icoil_dict['phi'],n_list,icoil_dict[i+'_results_list'])

#print out the results
mag_funcs.print_results(sensor_dict, icoil_dict, shot, start_time, end_time)

#print out the important results and a plot
print '='*10, 'n=2 SUMMARY','='*10
plot_list = ['total','vac','plasma']
plot_list = ['plasma']
plot_list = []
for i in plot_list:
    answer = sensor_dict[i+'_alpha']*1.e4/(icoil_dict['total_alpha']/1.e3) #convert to G/kA
    tmp = num.abs(answer)
    print '============= %d,%d-%dms, %s, %s ==================='%(shot, start_time, end_time, i, sensor_dict['name'])
    print '%-10s | %-10.2e |'%('rel_amp',tmp[2])
    tmp = num.angle(answer,deg=True)
    print '%-10s | %-10.1f |'%('rel_deg',tmp[2])

    #strange way to get the first item in the list to be the last for full 360deg plot....
    tmp_results = copy.deepcopy(sensor_dict[i+'_results_list'])
    tmp_results_phi = copy.deepcopy(sensor_dict['phi'])
    tmp_results = num.append(tmp_results, tmp_results[0])
    tmp_results_phi = num.append(tmp_results_phi, tmp_results_phi[0]+360)

    tmp_results_icoil =  copy.deepcopy(icoil_dict['total_results_list'])
    tmp_results_phi_icoil = copy.deepcopy(icoil_dict['phi'])
    tmp_results_phi_icoil = num.append(tmp_results_phi_icoil, tmp_results_phi_icoil[0]+360)
    tmp_results_icoil = num.append(tmp_results_icoil, tmp_results_icoil[0])

    #sensor_dict['results_list'] = num.array(sensor_dict['results_list'])
    fig_tmp,ax_tmp = pt.subplots(nrows=2,sharex=1)

    #plot of the amplitudes for each pickup
    ax_tmp[0].plot(sensor_dict['phi'], num.abs(sensor_dict[i+'_results_list']),'o')

    #plot of the phases for each of the pickups
    ax_tmp[1].plot(tmp_results_phi, num.angle(tmp_results, deg=True),'bo-')
    ax_tmp[1].plot(tmp_results_phi_icoil, num.angle(tmp_results_icoil, deg=True),'ko-')
    ax_tmp[0].set_xlim([num.min(sensor_dict['phi'])-20,num.min(sensor_dict['phi'])+360+20])
    ax_tmp[0].set_title(i)
    ax_tmp[0].set_ylim([0,15])
    fig_tmp.canvas.draw();fig_tmp.show()
    if i=='plasma':
        fig_tmp2, ax_tmp2 = pt.subplots()
        cumulative_phase = [0]
        tmp_results_ang = num.angle(tmp_results, deg=True)
        phase_diff = tmp_results_ang[1:]-tmp_results_ang[0:-1]

        for j in range(0,len(phase_diff)):
            if phase_diff[j]>0:
                cumulative_phase.append(cumulative_phase[-1]+phase_diff[j])
            else:
                cumulative_phase.append(cumulative_phase[-1]+phase_diff[j]+360)
        ax_tmp2.plot(tmp_results_phi, cumulative_phase, 'o-', label='Toroidal Pickup Array (dBp)')
        cumulative_phase = [0]
        tmp_results_icoil_ang = num.angle(tmp_results_icoil, deg=True)
        phase_diff = tmp_results_icoil_ang[1:]-tmp_results_icoil_ang[0:-1]
        for j in range(0,len(phase_diff)):
            if phase_diff[j]>0:
                cumulative_phase.append(cumulative_phase[-1]+phase_diff[j])
            else:
                cumulative_phase.append(cumulative_phase[-1]+phase_diff[j]+360)
        ax_tmp2.plot(tmp_results_phi_icoil, cumulative_phase, 'o-',label='I-Coil Perturbation')

        for tmp_loc, tmp_n in enumerate(n_list):
            tmp = num.angle(answer[tmp_loc]*num.exp(tmp_n*(tmp_results_phi/180.*num.pi)*1j), deg=True)
            phase_diff = tmp[1:]-tmp[0:-1]

            svd_cumul = [0]
            for j in range(0,len(phase_diff)):
                if phase_diff[j]>0:
                    svd_cumul.append(svd_cumul[-1]+phase_diff[j])
                else:
                    svd_cumul.append(svd_cumul[-1]+phase_diff[j]+360)

            ax_tmp2.plot(tmp_results_phi, svd_cumul, 'k-')
        ax_tmp2.set_xlabel('Toroidal Angle (deg)')
        ax_tmp2.set_ylabel('Cumulative Phase (deg)')
        ax_tmp2.set_title('Shot : %d, 10Hz Cumulative Phase (Plasma Response)'%(shot))
        ax_tmp2.grid()
        ax_tmp2.legend(loc='best')
        ax_tmp2.set_xlim([0,450])
        ax_tmp2.set_ylim([0,750])
        fig_tmp2.canvas.draw(); fig_tmp2.show()
        fig_tmp2.savefig('latest.pdf')

if write_pickle == 1:
    file_name = 'pickled_signals2.pickle'
    file_obj = file(file_name, 'w')
    pickle.dump(existing_signals, file_obj)
    file_obj.close()

print 'time to finish: %.2fs'%(timer_module.time() - timer_start)
