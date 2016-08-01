#!/usr/bin/env Python
'''SRH : 5 Oct 2015. Noticed some corrugation on the HFS vacuum calculation using MARS. Did not go away by reducing the frequency of the perturbation. Suspect it is due to insufficient numerical resolution, so increased the number of poloidal harmonics that are used in the analysis.

Found out that it was due to the poloidal harmonic resolution. Going from +-29 to +-45 fixed the problem, however, memory consumption increased from 15GB to 40, and execution time for MARS went from 26mins -> 91mins. 

Try to find some kind of optimum point
+-45 works well
+-35 still has the issue


'''


import pyMARS.results_class as results_class
import pickle,sys
import numpy as np
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs
import scipy.interpolate as interp
#import matplotlib.pyplot as pt
import time
import matplotlib.pyplot as pt
import numpy as np
import pickle

project_name = '/u/haskeysr/mars/shot153585_03795_q95_scan_josh_q95_scan_n3/shot153585_03795_q95_scan_josh_q95_scan_n3_post_processing_PEST.pickle'
project_name = "/u/haskeysr/mars/shot158115_04702_n2_q95_scan/shot158115_04702_n2_q95_scan_post_processing_PEST.pickle"
project_name = "/u/haskeysr/mars/shot158103_03796_q95_scan_carlos_thetac0-003_100/shot158103_03796_q95_scan_carlos_thetac0-003_100_post_processing_PEST.pickle"

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()
print 'opened project_dict %d items'%(len(project_dict.keys()))

start_time = time.time()
link_RMZM = 0
results = {}
Z = np.linspace(-0.5,0.5,150)
R = Z * 0 + 0.98

useful_keys = project_dict['sims'].keys()
#useful_keys = useful_keys[:5]
#for i in project_dict['sims'].keys():

fname_results = 'carlos_n2_q95scan_15Sept2015'
get_raw_data = False
if get_raw_data:
    for i in useful_keys:
        results[i] = {}
        n = np.abs(project_dict['sims'][i]['MARS_settings']['<<RNTOR>>'])
        I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
        Nchi = project_dict['sims'][i]['CHEASE_settings']['<<NCHI>>']
        print 'working on serial : ', i
        locs = ['upper','lower']
        for loc in locs:
            for type in ['plasma', 'vacuum']:
                directory = project_dict['sims'][i]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
                print directory, 'I0EXP=',I0EXP
                new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
                new_data_R = new_data.R*new_data.R0EXP
                new_data_Z = new_data.Z*new_data.R0EXP
                results[i]['{}_{}'.format(type, loc)] = interp.griddata((new_data_R.flatten(), new_data_Z.flatten()), new_data.Bz.flatten(),(R, Z))
        print '{}, {:.3f}s'.format(i, time.time() - start_time)

    #Dump the data that has upper/lower, plasma/vacuum data
    pickle.dump(results, file(fname_results+'.pickle','w'))

#Step two, get the real parts
results = pickle.load(file(fname_results+'.pickle','r'))
for type in ['plasma','vacuum']:
    q95 = []
    tmp = np.zeros((len(Z),len(project_dict['sims'].keys())), dtype = float)
    tmp = np.zeros((len(Z),len(useful_keys)), dtype = float)
    for j, key in enumerate(useful_keys):#project_dict['sims'].keys()):

    #for j, key in enumerate(range(1,26)):
        tmp[:,j] = np.real(results[key]['{}_{}'.format(type, 'upper')] + results[key]['{}_{}'.format(type, 'lower')])
        q95.append(project_dict['sims'][key]['Q95'])
    results[type] = +tmp
results['Q95'] = np.array(q95)
results['Z'] = Z
results['R'] = R
pickle.dump(results, file(fname_results+'.pickle','w'))

useful_keys = np.array(useful_keys)
order = np.argsort(results['Q95'])
keys_reordered = useful_keys[order]

locs = ['upper','lower']
out_dat = {}
for loc in locs:
    for type in ['plasma', 'vacuum']:
        for component in ['real','imag']:
            if type=='plasma':
                fname_tmp = 'total'
            else:
                fname_tmp = type

            fname = '{}_{}_{}_{}.txt'.format(fname_results, fname_tmp, loc, component)
            out_dat[fname] = []
            print fname
            lines = []
            cur_file = file(fname,'w')
            for key in keys_reordered:
                func = getattr(np,component)
                cur_row = func(results[key]['{}_{}'.format(type, loc)])
                out_dat[fname].append(cur_row.tolist())

for i in out_dat.keys():np.savetxt('{}'.format(i),out_dat[i])
#for key in keys_reordered: 
#    out_dat['Q95'].append(cur_row.tolist())
np.savetxt('{}_{}.txt'.format(fname_results, 'Q95'),results['Q95'][order])
for i in ['Z','R']: 
    np.savetxt('{}_{}.txt'.format(fname_results, i),results[i])

check_output = True
if check_output:
    q95_new = np.loadtxt('{}_{}.txt'.format(fname_results, 'Q95'))
    R_new = np.loadtxt('{}_{}.txt'.format(fname_results, 'R'))
    Z_new = np.loadtxt('{}_{}.txt'.format(fname_results, 'Z'))
    Z_grid, q95_grid = np.meshgrid(Z_new, q95_new)
    fig, ax = pt.subplots(ncols=3,nrows=3,sharex = True, sharey=True)
    fig2, ax2 = pt.subplots(ncols=3,nrows=3,sharex = True, sharey=True)
    ax = ax.flatten()
    ax2 = ax2.flatten()
    for ind, i in enumerate(out_dat.keys()):
        print i
        a = np.loadtxt(i)
        im = ax[ind].pcolormesh(q95_grid.T, Z_grid.T, a.T, cmap = 'RdBu')
        for j in range(a.shape[0]):
            clr = '{}'.format((q95_new[j]-np.min(q95_new))/(np.max(q95_new)+0.5 - np.min(q95_new)))
            print clr
            ax2[ind].plot(Z_new, a[j,:],color=clr)
        ax[ind].plot(q95_new,q95_new*0,'o')
        im.set_clim([-0.3,0.3])
        ax[ind].set_title(i)
        print a.shape
fig.canvas.draw();fig.show()
fig2.canvas.draw();fig2.show()
fwchi = []
fcchi = []
for i in keys_reordered:
    fwchi.append(project_dict['sims'][i]['FWCHI'])
    fcchi.append(project_dict['sims'][i]['FCCHI'])
fwchi = np.array(fwchi)
fcchi = np.array(fcchi)
fig, ax = pt.subplots(nrows = 2)
ax[0].plot(fwchi[:,0])
ax[0].plot(fwchi[:,1])
ax[1].plot(fcchi[:,0])
ax[1].plot(fcchi[:,1])
fig.canvas.draw();fig.show()


fig, ax = pt.subplots(nrows = 3)
fname_tmp = 'vacuum'
loc = 'upper'
fname_imag_u = '{}_{}_{}_imag.txt'.format(fname_results, fname_tmp, loc,)
fname_real_u = '{}_{}_{}_real.txt'.format(fname_results, fname_tmp, loc,)
loc = 'lower'
fname_imag_l = '{}_{}_{}_imag.txt'.format(fname_results, fname_tmp, loc,)
fname_real_l = '{}_{}_{}_real.txt'.format(fname_results, fname_tmp, loc,)
a_imag_u = np.loadtxt(fname_imag_u)
a_real_u = np.loadtxt(fname_real_u)
a_imag_l = np.loadtxt(fname_imag_l)
a_real_l = np.loadtxt(fname_real_l)
tot1 = np.abs(a_real_u)
tot2 = np.abs(a_real_l)
tot3 = np.abs(a_real_u + a_real_l + 1j*(a_imag_u + a_imag_l))
im1 = ax[0].pcolormesh(q95_grid.T, Z_grid.T, tot1.T, cmap = 'RdBu')
im2 = ax[1].pcolormesh(q95_grid.T, Z_grid.T, tot2.T, cmap = 'RdBu')
im3 = ax[2].pcolormesh(q95_grid.T, Z_grid.T, tot3.T, cmap = 'RdBu')
im2.set_clim(im1.get_clim())
im3.set_clim(im1.get_clim())
fig.canvas.draw();fig.show()

raise(ValueError)

results = pickle.load(file(fname_results+'.pickle','r'))
Z = np.linspace(-0.5,0.5,100)
R = Z * 0 + 0.98
fig, ax = pt.subplots()
#X, Y = np.meshgrid(Z,results['Q95'])
order = np.argsort(results['Q95'])
X, Y = np.meshgrid(results['Q95'][order],results['Z'])
mult = 1232.42963413606/1.9556788
mult = 1
im = ax.pcolormesh(X, Y, mult*(results['plasma'][:,order]-results['vacuum'][:,order]), cmap = 'RdBu')
cbar = pt.colorbar(im, ax = ax)
cbar.set_label('Re(Bz) G/kA')
lim = 0.6
im.set_clim([-lim,lim])
ax.set_xlim([np.min(results['Q95']),np.max(results['Q95'])])
ax.set_xlabel('q95')
ax.set_ylabel('HFS Height (m)')
#ax.set_xlim([3.1,3.8])
ax.set_ylim([-0.5,0.5])
fig.canvas.draw(); fig.show()

loc = '/u/haskeysr/'
np.savetxt(loc+fname_results+'_plasma_only.txt', mult*(results['plasma']-results['vacuum']))
np.savetxt(loc+fname_results+'_total.txt', mult*(results['plasma']))
np.savetxt(loc+fname_results+'_vacuum.txt', mult*(results['vacuum']))
np.savetxt(loc+fname_results+'_q95.txt', mult*(results['Q95']))
np.savetxt(loc+fname_results+'_Z.txt', mult*(results['Z']))

#R = 0.98
#im = ax.pcolormesh(new_data_R, new_data_Z, new_data.Bz, cmap = 'RdBu')
#im.set_clim([-0.2,0.2])
#ax.set_xlim([0.95,2.3])
#ax.set_ylim([-1.5,1.5])
#fig.canvas.draw(); fig.show()
#fig, ax = pt.subplots()
#ax.plot(Z, vals)
#fig.canvas.draw(); fig.show()

#print np.argmin((new_data_R - 0.98)**2 - (new_data_Z)**2, axis = 0)
#print np.argmin((new_data_R - 0.98)**2 - (new_data_Z)**2, axis = 1)
# project_dict = coil_outputs_B(project_dict, upper_and_lower = upper_and_lower)
# print 'finished calc'

# output_name = project_name + 'output'
# pickle_file = open(output_name,'w')
# pickle.dump(project_dict, pickle_file)
# pickle_file.close()
# print 'output file'
target_q95s = [4.58, 4.733]
target_q95s = [3.6, 4.75]
fig, ax = pt.subplots(nrows = 1, sharex =True, sharey=True)
ax = [ax]
target_q95s = [3.6]

for j, target_q95 in enumerate(target_q95s):
    #loc = np.argmin(np.abs(results['Q95'][order] - target_q95))
    #key = keys_reordered[loc]
    key = 20
    print target_q95, project_dict['sims'][key]['Q95']
    n = np.abs(project_dict['sims'][key]['MARS_settings']['<<RNTOR>>'])
    I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
    Nchi = project_dict['sims'][key]['CHEASE_settings']['<<NCHI>>']
    print 'working on serial : ', key
    type = 'vacuum'
    loc = 'upper'
    directory = project_dict['sims'][key]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
    print directory, 'I0EXP=',I0EXP
    new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
    new_data_R = new_data.R*new_data.R0EXP
    new_data_Z = new_data.Z*new_data.R0EXP
    im = new_data.plot_Bn(np.abs(new_data.Bz),axis = ax[j],cmap='spectral', end_surface = -1, plot_boundaries = 1, wall_grid = new_data.Ns2 - new_data.Ns1)
    im.set_clim([-.2,.2])
    pt.colorbar(im,ax=ax[j])
    ax[j].plot(R,Z,'.')
fig.canvas.draw();fig.show()



#Check the 10Hz problem??
import pyMARS.results_class as results_class
import pickle,sys
import numpy as np
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs
import scipy.interpolate as interp
#import matplotlib.pyplot as pt
import time
import matplotlib.pyplot as pt
import numpy as np
import pickle
project_name = '/u/haskeysr/mars/shot153585_03795_q95_scan_josh_q95_scan_n3/shot153585_03795_q95_scan_josh_q95_scan_n3_post_processing_PEST.pickle'
project_name = "/u/haskeysr/mars/shot158115_04702_n2_q95_scan/shot158115_04702_n2_q95_scan_post_processing_PEST.pickle"

ids = ['10','5','1','0-1']
fig, ax = pt.subplots(ncols = len(ids), sharex = True, sharey = True)
for freq, tmp_ax in zip(ids,ax):
    project_name = "/u/haskeysr/mars/shot158103_03796_q95_scan_carlos_check_vac_{}Hz/shot158103_03796_q95_scan_carlos_check_vac_{}Hz_post_processing_PEST.pickle".format(freq,freq)
    with file(project_name,'r') as pickle_file:
        project_dict = pickle.load(pickle_file)
    key = 1
    print 'working on serial : ', key
    type = 'vacuum'
    #type = 'plasma'
    loc = 'upper'
    directory = project_dict['sims'][key]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
    n = np.abs(project_dict['sims'][key]['MARS_settings']['<<RNTOR>>'])
    I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
    print directory, 'I0EXP=',I0EXP
    new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
    new_data_R = new_data.R*new_data.R0EXP
    new_data_Z = new_data.Z*new_data.R0EXP
    im = new_data.plot_Bn(np.abs(new_data.Bn),axis = tmp_ax,cmap='spectral', end_surface = -1, plot_boundaries = 1,)
    im.set_clim([0.,.5])
    tmp_ax.set_title('{},{}Hz'.format(project_dict['sims'][key]['MARS_settings']['<<AL0>>'], freq))
    pt.colorbar(im,ax=tmp_ax)
fig.canvas.draw();fig.show()


ids = ['10Hz_high_res','10Hz_40','10Hz_35','10Hz']
fig, ax = pt.subplots(ncols = len(ids), sharex = True, sharey = True)
for freq, tmp_ax in zip(ids,ax):
    project_name = "/u/haskeysr/mars/shot158103_03796_q95_scan_carlos_check_vac_{}/shot158103_03796_q95_scan_carlos_check_vac_{}_post_processing_PEST.pickle".format(freq,freq)
    with file(project_name,'r') as pickle_file:
        project_dict = pickle.load(pickle_file)
    key = 1
    print 'working on serial : ', key
    type = 'vacuum'
    type = 'plasma'
    loc = 'upper'
    directory = project_dict['sims'][key]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
    n = np.abs(project_dict['sims'][key]['MARS_settings']['<<RNTOR>>'])
    I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
    print directory, 'I0EXP=',I0EXP
    new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
    new_data_R = new_data.R*new_data.R0EXP
    new_data_Z = new_data.Z*new_data.R0EXP
    im = new_data.plot_Bn(np.abs(new_data.Bz),axis = tmp_ax,cmap='spectral', end_surface = -1, plot_boundaries = 1,)
    im.set_clim([0.,2])
    tmp_ax.set_title('{},{}Hz'.format(project_dict['sims'][key]['MARS_settings']['<<AL0>>'], freq))
    pt.colorbar(im,ax=tmp_ax)
    print '!!!!', project_dict['sims'][1]['MARS_settings']['<<M1>>']
    print '!!!!', new_data.Mm, new_data.Nm0
    print '!!!!', new_data.M1, new_data.M2
    new_data.get_PEST()
    fig2, ax2 = pt.subplots()
    new_data.plot_BnPEST(ax2)
    fig2.canvas.draw();fig2.show()
fig.canvas.draw();fig.show()
