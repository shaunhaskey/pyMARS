#!/usr/bin/env Python
import pyMARS.results_class as results_class
import pickle,sys
import numpy as np
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs

def perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP= 1.0e+3 * 3./np.pi):
    #print 'in perform_calcs'
    #print directory, 'I0EXP=',I0EXP

    #I0EXP = RZfuncs.I0EXP_calc(N,n,I)
    new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
    #print 'results_class initialised'
    new_data_R = new_data.R*new_data.R0EXP
    new_data_Z = new_data.Z*new_data.R0EXP
    #print 'R and Z data obtained'
    new_answer = np.array(pyMARS_funcs.coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi,probe, probe_type, Rprobe,Zprobe,tprobe,lprobe))
    #print 'finished calculation'
    return new_answer

upper_and_lower = 1
#project_dict = pickle.load(file('/home/srh112/NAMP_datafiles/mars/shot156746_02113_betaN_ramp_carlos_prl_n1/shot156746_02113_betaN_ramp_carlos_prl_n1_post_processing_PEST.pickle','r'))
project_dict = pickle.load(file('/home/srh112/NAMP_datafiles/mars/shot156746_04617_betaN_ramp_carlos_prlV2_n1/shot156746_04617_betaN_ramp_carlos_prlV2_n1_post_processing_PEST.pickle','r'))
#def coil_outputs_B(project_dict, upper_and_lower=0):
probe = project_dict['details']['pickup_coils']['probe']
probe_type = project_dict['details']['pickup_coils']['probe_type']
Rprobe = project_dict['details']['pickup_coils']['Rprobe']
Zprobe = project_dict['details']['pickup_coils']['Zprobe']
tprobe = project_dict['details']['pickup_coils']['tprobe']
lprobe = project_dict['details']['pickup_coils']['lprobe']
link_RMZM = 0

#Nchi = 240
#for i in project_dict['sims'].keys():
output_objs = []
output_names = []
for i in [12]:
    n = np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
    #q_range = [0, n+4]
    project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
    #project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
    #project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc(project_dict['sims'][i]['I-coils']['N_Icoils'],num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>']),project_dict['sims'][i]['I-coils']['I_coil_current'])
    #n = num.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
    #project_dict['sims'][i]['I0EXP'] = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
    Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
    print 'working on serial : ', i
    locs = ['upper','lower'] if upper_and_lower else ['']
    for loc in locs:
        for type in ['plasma', 'vacuum']:
            directory = project_dict['sims'][i]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
            print directory
            output_names.append('{}_{}_response4'.format(type, loc))
            tmp2 = project_dict['sims'][i]['{}_{}_response4'.format(type, loc)]
            I0EXP = project_dict['sims'][i]['I0EXP']
            #tmp = perform_calcs(directory, Nchi, link_RMZM, probe, probe_type, Rprobe, Zprobe,tprobe,lprobe, I0EXP)
            new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
            output_objs.append(new_data)
            new_data_R = new_data.R*new_data.R0EXP
            new_data_Z = new_data.Z*new_data.R0EXP
            tmp = np.array(pyMARS_funcs.coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi,probe, probe_type, Rprobe,Zprobe,tprobe,lprobe))
            for p, ii, jj in zip(probe, tmp,tmp2): print '{}:{:.4e}%,{:.4e}%'.format(p, (np.real(ii)- np.real(jj))/np.real(ii)*100, (np.imag(ii)- np.imag(jj))/np.imag(ii)*100)


import matplotlib.pyplot as pt
total_upper, vacuum_upper, total_lower, vacuum_lower = output_objs
quant = total_upper.Bz + total_lower.Bz - vacuum_upper.Bz - vacuum_lower.Bz
quant2 = total_upper.Br + total_lower.Br - vacuum_upper.Br - vacuum_lower.Br
fig, ax = pt.subplots(ncols = 2, sharex = True, sharey = True)
im1 = new_data.plot_Bn(np.real(quant), ax[0], plot_coils_switch=1, plot_boundaries=1, cmap='RdBu', end_surface=380)
im2 = new_data.plot_Bn(np.imag(quant), ax[1], plot_coils_switch = 1, plot_boundaries = 1, cmap = 'RdBu', end_surface = 380)
im1.set_clim([-10,10])
im2.set_clim([-10,10])
pt.colorbar(im1,ax = [ax[0],ax[1]])
print im1.get_clim(), im2.get_clim()
fig.canvas.draw(); fig.show()

fig, ax = pt.subplots(ncols = 2, sharex = True, sharey = True)
im1 = new_data.plot_Bn(np.real(quant2), ax[0], plot_coils_switch = 1, plot_boundaries = 1, cmap = 'RdBu')
im2 = new_data.plot_Bn(np.imag(quant2), ax[1], plot_coils_switch = 1, plot_boundaries = 1, cmap = 'RdBu')
im1.set_clim([-10,10])
im2.set_clim([-10,10])
pt.colorbar(im1,ax = [ax[0],ax[1]])
print im1.get_clim(), im2.get_clim()
fig.canvas.draw(); fig.show()

total_upper.get_PEST()
total_upper.resonant_strength(n=1)
fig, ax = pt.subplots()
ax.plot(total_upper.q_profile_s, total_upper.q_profile)
dqds = total_upper.q_profile_s * 0
dqds[1:] = np.diff(total_upper.q_profile) / np.diff(total_upper.q_profile_s)
dsdq_res = [dqds[np.argmin(np.abs(total_upper.q_profile_s - s))] for s in total_upper.sq]
ax.plot(total_upper.q_profile_s, dqds)
ax.plot(total_upper.sq, total_upper.qn,'o')
ax.plot(total_upper.sq, dsdq_res,'o')
fig.canvas.draw(); fig.show()

#N = 6
#n = 2
#I = np.array([1.,-1.,0.,1,-1.,0.])

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()
print 'opened project_dict %d items'%(len(project_dict.keys()))
project_dict = coil_outputs_B(project_dict, upper_and_lower = upper_and_lower)
print 'finished calc'

output_name = project_name + 'output'
pickle_file = open(output_name,'w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()
print 'output file'
