#This calculates the phasing between a pickup array and an I-coil array at a given frequency
#The various vacuum couplings are calculated using a hdf5 containing the details
#Many of the worker functions are in magnetics_generic_funcs.py
#Shaun Haskey 22/06/2012
#



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

sensor_array_name = 'Bp_probes_R0_working'
icoil_name =  'I_coils_upper'


i_coil_freq = 10.

use_pickled_signals = 1
write_pickle = 0

sample_rate = 1000 #Hz
#n_list = [0,2,4]

remove_mean = 0
remove_trend = 0
timer_start = timer_module.time()
nd = 16
coupling_coils = ['IU30','IU90','IU150','IU210','IU270','IU330','IL30','IL90','IL150','IL210','IL270','IL330']

I_coil_plots = 0

shot = 148765;start_time = 1000; end_time = 6000 
single_transfers = '/home/srh112/NAMP_datafiles/tf2012_single.h5'
vac_coupling = h5py.File(single_transfers, 'r')

#create the interpolated time axis
interp_time = num.arange(start_time, end_time, (1./sample_rate)*1000, dtype=float)

existing_signals = {}

sensor_dict = {}
sensor_dict['name']=sensor_array_name
sensor_dict['pickup_names']=mag_details.pickups.names(sensor_array_name)
sensor_dict['phi'] = mag_details.pickups.phi(sensor_array_name)

icoil_dict = {}
icoil_dict['name']= icoil_name
icoil_dict['pickup_names']=mag_details.coils.names(icoil_name)
icoil_dict['phi'] = mag_details.coils.phi(icoil_name)

sensor_dict = mag_funcs.extract_data(sensor_dict, shot, interp_time, vac_coupling, coupling_coils, sample_rate, i_coil_freq, existing_signals, plotting=1, ax_ylim = [0,20], remove_mean = remove_mean, remove_trend = remove_trend)

icoil_dict = mag_funcs.extract_data(icoil_dict, shot, interp_time, vac_coupling, [], sample_rate, i_coil_freq, existing_signals, plotting=I_coil_plots, ax_ylim = None, remove_mean = 0, remove_trend = 0, plot_all=1)

