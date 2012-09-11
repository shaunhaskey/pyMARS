import pickle
import numpy as num
import matplotlib.pyplot as pt
import h5py

load_hdf5 = 1
shot = 146397
load_pickle = 0
hdf5_filename = '/u/haskeysr/hdf5testfile2.h5'
if load_pickle == 1:
    tmp_filename = file('displacement_data%s.pickle'%(shot),'r')
    stored_data = pickle.load(tmp_filename)
    tmp_filename.close()
    n_time, n_x1, n_data = stored_data['n']
    I_coil_x, I_coil_y = stored_data['I_coil']
    boundary_x, boundary_y = stored_data['boundary']

elif load_hdf5 ==1:

    tmp_file = h5py.File(hdf5_filename,'r')
    stored_data = tmp_file.get(str(shot))
    n_data = stored_data[0][0]
    n_time = stored_data[0][1]
    n_r = stored_data[0][2]
    n_rho = stored_data[0][3]
    tmp_file.close()
    
    #tmp_filename = file('displacement_data%s.pickle'%(shot),'r')
    tmp_filename = file('displacement_data%s.pickle'%(shot),'r') #tmp hack to make it work
    stored_data = pickle.load(tmp_filename)
    tmp_filename.close()
    I_coil_x, I_coil_y = stored_data['I_coil']
    boundary_x, boundary_y = stored_data['boundary']
    
