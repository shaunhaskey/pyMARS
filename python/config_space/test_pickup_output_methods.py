#!/usr/bin/env Python
import results_class
import pickle,sys
import numpy as num
import numpy as np
import PythonMARS_funcs as pyMARS
import RZfuncs as RZfuncs
import matplotlib.pyplot as pt
from matplotlib.mlab import griddata

def coil_outputs_B(project_dict, upper_and_lower=0):
    probe = project_dict['details']['pickup_coils']['probe']
    probe_type = project_dict['details']['pickup_coils']['probe_type']
    Rprobe = project_dict['details']['pickup_coils']['Rprobe']
    Zprobe = project_dict['details']['pickup_coils']['Zprobe']
    tprobe = project_dict['details']['pickup_coils']['tprobe']
    lprobe = project_dict['details']['pickup_coils']['lprobe']
    link_RMZM = 0
    #Nchi = 240
    for i in project_dict['sims'].keys():
        project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
        Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
        print 'working on serial : ', i
        if upper_and_lower == 1:
            #print 'hello1'
            directory = project_dict['sims'][i]['dir_dict']['mars_upper_vacuum_dir']
            #print directory, Nchi, link_RMZM
            #print 'starting vacuum_upper_response'
            project_dict['sims'][i]['vacuum_upper_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
            directory = project_dict['sims'][i]['dir_dict']['mars_lower_vacuum_dir']
            project_dict['sims'][i]['vacuum_lower_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
            directory = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
            project_dict['sims'][i]['plasma_upper_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
            directory = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']
            project_dict['sims'][i]['plasma_lower_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
        else:
            #print 'hello2'
            directory = project_dict['sims'][i]['dir_dict']['mars_vac_dir']
            project_dict['sims'][i]['vacuum_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
            directory = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
            project_dict['sims'][i]['plasma_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])
            
    return project_dict

def coil_responses4(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL']
    # probe type 1: poloidal field, 2: radial field
    type   = np.array([     1,     1,     1,     0,     0,     0,     0])
    # Poloidal geometry
    Rprobe = np.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300])
    Zprobe = np.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714])
    tprobe = np.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6])*2*np.pi/360  #DTOR # poloidal inclination
    lprobe = np.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = [] # final output

    for k in range(0, Nprobe):

        #depending on poloidal/radial
        if type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*np.cos(tprobe[k])*np.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*np.sin(tprobe[k])*np.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*np.sin(tprobe[k])*np.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*np.cos(tprobe[k])*np.linspace(-1,1,num = Navg)

        Br_list=[];Bz_list=[];Bphi_list=[]

        r_filt = [];z_filt = [];Br_filt = [];Bz_filt = []
        Rprobek_min = np.min(Rprobek) - 0.1
        Rprobek_max = np.max(Rprobek) + 0.1
        Zprobek_min = np.min(Zprobek) - 0.1
        Zprobek_max = np.max(Zprobek) + 0.1

        #search box for interpolation to minimise computation

        #whittle down to poindts near the coil


        for iii in range(0,r_array.shape[0]):
            for jjj in range(0,r_array.shape[1]):
                if (((Rprobek_min <= r_array[iii,jjj]) and (r_array[iii,jjj] <= Rprobek_max)) and ((Zprobek_min <= z_array[iii,jjj]) and (z_array[iii,jjj] <= Zprobek_max))):
                    r_filt.append(r_array[iii,jjj])
                    z_filt.append(z_array[iii,jjj])
                    Br_filt.append(Br[iii,jjj])
                    Bz_filt.append(Bz[iii,jjj])


        r_filt_array = np.array(r_filt)
        z_filt_array = np.array(z_filt)
        Br_filt_array = np.array(Br_filt)
        Bz_filt_array = np.array(Bz_filt)

#        print r_filt_array.shape
#        print z_filt_array.shape
#        print Br_filt_array.shape


        #Create interpolated values

        Brprobek = griddata(r_filt_array, z_filt_array, np.real(Br_filt_array), Rprobek, Zprobek) + 1j*griddata(r_filt_array, z_filt_array, np.imag(Br_filt_array), Rprobek, Zprobek)
        Bzprobek = griddata(r_filt_array, z_filt_array, np.real(Bz_filt_array), Rprobek, Zprobek) + 1j*griddata(r_filt_array, z_filt_array, np.imag(Bz_filt_array), Rprobek, Zprobek)


        #Find perpendicular components
        Bprobek  =  (np.sin(tprobe[k])*np.real(Bzprobek) + np.cos(tprobe[k])*np.real(Brprobek)) + 1j * (np.sin(tprobe[k])*np.imag(Bzprobek) +np.cos(tprobe[k])*np.imag(Brprobek))

        Bprobem.append(np.average(Bprobek))

    return Bprobem



def perform_calcs2(directory, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP= 1.0e+3 * 3./np.pi):
    #print 'in perform_calcs'
    print directory, 'I0EXP=',I0EXP

    #I0EXP = RZfuncs.I0EXP_calc(N,n,I)
    new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP)
    #print 'results_class initialised'
    new_data_R = new_data.R*new_data.R0EXP
    new_data_Z = new_data.Z*new_data.R0EXP

    fig, ax = pt.subplots()
    ax.plot(new_data_R, new_data_Z, '-')
    ax.plot(Rprobe, Zprobe,'o')
    fig.canvas.draw(); fig.show()
    print 'hello!!!!'
    #print 'R and Z data obtained'
    new_answer1 = num.array(pyMARS.coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi,probe, probe_type, Rprobe,Zprobe,tprobe,lprobe))
    new_answer2 = num.array(pyMARS.coil_responses6_backup(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi,probe, probe_type, Rprobe,Zprobe,tprobe,lprobe))
    new_answer3 = num.array(coil_responses4(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi))
    #print 'finished calculation'
    return new_answer1, new_answer2, new_answer3


I0EXP = 826
probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
# probe type 1: poloidal field, 2: radial field
probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
# Poloidal geometry
Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

directory = '/home/srh112/NAMP_datafiles/mars/shot146398_0deg/qmult1.000/exp1.000/marsrun/RUNrfa.p'
plas_response = perform_calcs2(directory, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=I0EXP)
directory = '/home/srh112/NAMP_datafiles/mars/shot146398_0deg/qmult1.000/exp1.000/marsrun/RUNrfa.vac'
vac_response = perform_calcs2(directory, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=I0EXP)




# directory = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
# project_dict['sims'][i]['plasma_response4'] = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP=project_dict['sims'][i]['I0EXP'])

# #N = 6
# #n = 2
# #I = num.array([1.,-1.,0.,1,-1.,0.])

# pickle_file = open(project_name,'r')
# project_dict = pickle.load(pickle_file)
# pickle_file.close()
# print 'opened project_dict %d items'%(len(project_dict.keys()))
# project_dict = coil_outputs_B(project_dict, upper_and_lower = upper_and_lower)
# print 'finished calc'

# output_name = project_name + 'output'
# pickle_file = open(output_name,'w')
# pickle.dump(project_dict, pickle_file)
# pickle_file.close()
# print 'output file'
