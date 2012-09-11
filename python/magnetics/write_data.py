'''
reads in data from MDSplus on GA computers and dumps the file
as a pickle file so it can be used remotely or for faster processing
SH: 11Sept2012
'''

import h5py, pickle, os, data
import numpy as num
import magnetics_generic_funcs as mag_funcs
import magnetics_details as mag_details

file_name = 'pickled_signals3.pickle'
i_coil_freq = 10.
use_pickled_signals = 1
coil_name_list_upper = ['IU30','IU90','IU150','IU210','IU270','IU330','IL30','IL90','IL150','IL210','IL270','IL330']
single_transfers = '/u/hansonjm/var/data/transfers/tf2012_single.h5'
f = h5py.File(single_transfers, 'r')
start_time = 3000; end_time = 4000; sample_rate = 1000
time = num.arange(start_time, end_time, (1./sample_rate)*1000, dtype=float)

if os.path.exists(file_name) & use_pickled_signals==1:
    print 'pickle file exists, reading in pickled file'
    file_obj = file(file_name, 'r')
    existing_signals = pickle.load(file_obj)
    file_obj.close()
else:
    existing_signals = {}

array_list = ['Bp_probes_R0_working', 'Br_probes_R0', 'Br_probes_R1', 'Br_probes_R2']
shot_list = [146382, 146388, 146392, 146397, 146398]
for shot in shot_list:
    for tmp_array_name in array_list:
        tmp_dict = {}
        tmp_dict['name'] = tmp_array_name
        tmp_dict['pickup_names'] = mag_details.pickups.names(tmp_array_name)
        tmp_dict['phi'] = mag_details.pickups.phi(tmp_array_name)
        tmp_dict = mag_funcs.extract_data(tmp_dict, shot, time, f, coil_name_list_upper, sample_rate, i_coil_freq, existing_signals)
        del tmp_dict

file_obj = file(file_name, 'w')
pickle.dump(existing_signals, file_obj)
file_obj.close()


