#!/usr/bin/env Python
import pyMARS.results_class as results_class
import pickle,sys
import numpy as np
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs
import scipy.interpolate as interp
#import matplotlib.pyplot as pt
import time

project_name = '/u/haskeysr/mars/shot153585_03795_q95_scan_josh_q95_scan_n3/shot153585_03795_q95_scan_josh_q95_scan_n3_post_processing_PEST.pickle'
project_name = "/u/haskeysr/mars/shot158115_04702_n2_q95_scan/shot158115_04702_n2_q95_scan_post_processing_PEST.pickle"

pickle_file = open(project_name,'r')
project_dict = pickle.load(pickle_file)
pickle_file.close()
print 'opened project_dict %d items'%(len(project_dict.keys()))

start_time = time.time()
link_RMZM = 0
results = {}
Z = np.linspace(-0.5,0.5,150)
R = Z * 0 + 0.98
#for i in [1,2]:
tmp_a = project_dict['sims'].keys()

#for i in project_dict['sims'].keys():
for i in tmp_a[:3]:
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

pickle.dump(results, file('output_josh_n3_raw.pickle','w'))
results = pickle.load(file('output_josh_n3_raw.pickle','r'))

for type in ['plasma','vacuum']:
    q95 = []
    tmp = np.zeros((len(Z),len(project_dict['sims'].keys())), dtype = float)
    for j, key in enumerate(project_dict['sims'].keys()):

    #for j, key in enumerate(range(1,26)):
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
mult = 1
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
