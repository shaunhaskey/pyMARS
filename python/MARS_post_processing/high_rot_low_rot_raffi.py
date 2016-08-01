#!/usr/bin/env Python
import pyMARS.results_class as results_class
import pickle,sys
import numpy as np
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs
import scipy.interpolate as interp
#import matplotlib.pyplot as pt
import time

project_name = '/u/haskeysr/mars/shot158115_04702_n2_rot_scan/shot158115_04702_n2_rot_scan_post_processing_PEST.pickle'
project_dict = pickle.load(open(project_name,'r'))
print 'opened project_dict %d items'%(len(project_dict.keys()))

rote_list = []
sim_list = np.array(project_dict['sims'].keys())
rote_list = np.array([project_dict['sims'][i]['MARS_settings']['<<ROTE>>'] for i in sim_list])
print rote_list

vals = [5.e-2,1.e-3]
sim_care = [sim_list[np.argmin(np.abs(rote_list - val))] for val in vals]
start_time = time.time()
link_RMZM = 0
results_hfs = {}
Z = np.linspace(-0.5,0.5,150)
R = Z * 0 + 0.98
#for i in [1,2]:
Z_probe = 0
R_probe = 2.413

fig, ax = pt.subplots()
ax.plot(new_data_R[ind-1,:], new_data_Z[ind-1,:])
ax.plot(new_data_R[ind-1,val_inds], new_data_Z[ind-1,val_inds])
ax.plot(R_new, Z)
fig.canvas.draw(); fig.show()

results = {}
for j in ['hfs','lfs']:
    results[j] = {}
    for i in sim_care:
        results[j][i] = {}
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
                if j == 'lfs':
                    print 'lfs'
                    distance = [np.min((Z_probe - new_data_Z[ii,:])**2 + (R_probe - new_data_R[ii,:])**2) for ii in range(new_data_Z.shape[1])]
                    ind = np.argmin(distance)
                    val_inds = (new_data_Z[ind-1,:]<0.6) * (new_data_Z[ind-1,:]>-0.6) * (new_data_R[ind-1,:]>1.4)
                    R_use = np.interp(Z, new_data_Z[ind-1,val_inds], new_data_R[ind-1, val_inds])
                elif j == 'hfs':
                    R_use = +R
                results[j][i]['{}_{}'.format(type, loc)] = interp.griddata((new_data_R.flatten(), new_data_Z.flatten()), new_data.Bz.flatten(),(R_use, Z))
        print '{}, {:.3f}s'.format(i, time.time() - start_time)


fig, ax = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
clims = [[0,0.6],[0,9]]
for i,key in enumerate(results.keys()):
    tmp_results = results[key]
    ax_cur = ax[:,i]
    type = 'plasma'
    im_list = []
    clim = clims[i]
    import matplotlib.pyplot as pt
    for ax_tmp, key2, lab in zip(ax_cur,sim_care, vals):
        tmp = []
        ax_tmp.set_title('{}, {:.3f}\% Va'.format(key,lab*100))
        phasings = np.linspace(0,2.*np.pi,100)
        for phasing in phasings:
            tmp.append(np.abs(tmp_results[key2]['{}_{}'.format(type, 'upper')] + np.exp(1j*phasing)*tmp_results[key2]['{}_{}'.format(type, 'lower')]))
        tmp = np.array(tmp)
        phasing_grid, Z_grid =np.meshgrid(phasings,Z)
        im = ax_tmp.pcolormesh(np.rad2deg(phasing_grid),Z_grid,tmp.T,cmap = 'hot')
        im.set_clim(clim)
    pt.colorbar(im,ax = ax_cur.tolist(), orientation = 'horizontal')
ax[0,0].set_xlim([0,360])
ax[0,0].set_ylim([-0.5,0.5])
for i in ax[:,0]:i.set_ylabel('Z (m)')
for i in ax[-1,:]:i.set_xlabel('Phasing (deg)')
fig.canvas.draw(); fig.show()

for type in ['plasma','vacuum']:
    #q95 = []
    tmp = np.zeros((len(Z),len(project_dict['sims'].keys())), dtype = float)
    for j, key in enumerate(project_dict['sims'].keys()):
        tmp[:,j] = np.real(results[key]['{}_{}'.format(type, 'upper')] + results[key]['{}_{}'.format(type, 'lower')])
        q95.append(project_dict['sims'][key]['Q95'])
    results[type] = +tmp
results['Q95'] = np.array(q95)
results['Z'] = Z
pickle.dump(results, file('output_josh_n3_sorted.pickle','w'))

import matplotlib.pyplot as pt
import numpy as np
import pickle
results = pickle.load(file('output_josh_n3_sorted.pickle','r'))
Z = np.linspace(-0.5,0.5,100)
R = Z * 0 + 0.98
fig, ax = pt.subplots()
#X, Y = np.meshgrid(Z,results['Q95'])
X, Y = np.meshgrid(results['Q95'],results['Z'])
mult = 1232.42963413606/1.9556788
im = ax.pcolormesh(X, Y, mult*(results['plasma']-results['vacuum']), cmap = 'RdBu')
cbar = pt.colorbar(im, ax = ax)
cbar.set_label('Re(Bz) G/kA')
lim = 0.6
im.set_clim([-lim,lim])
ax.set_xlim([2,5])
ax.set_xlabel('q95')
ax.set_ylabel('HFS Height (m)')
#ax.set_xlim([3.1,3.8])
ax.set_ylim([-0.5,0.5])
fig.canvas.draw(); fig.show()
loc = '/u/haskeysr/'
np.savetxt(loc+'josh_plasma_only.txt', mult*(results['plasma']-results['vacuum']))
np.savetxt(loc+'josh_total.txt', mult*(results['plasma']))
np.savetxt(loc+'josh_vacuum.txt', mult*(results['vacuum']))
np.savetxt(loc+'josh_q95.txt', mult*(results['Q95']))
np.savetxt(loc+'josh_Z.txt', mult*(results['Z']))

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
