#!/usr/bin/env Python
import pickle, sys
import numpy as num
from PythonMARS_funcs import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pt

project_name = sys.argv[1]

pickle_file = open('/u/haskeysr/mars/'+ project_name +'/7_post_mars_run.pickle','r')
project_dict = pickle.load(pickle_file)
pickle_file.close()

#dir = '/u/haskeysr/mars/project1/shot138344/tc_003/qmult0.60/exp0.63/marsrun/RUNrfa.p/'

#extract data:
Bn_Div_Li = []
q95 = []
Br_val_list = []
Bz_val_list = []
Bphi_val_list = []
R_val_list = []
Z_val_list = []
success = 0
fail = 0

for i in project_dict['sims'].keys():
    #get Bn/Li and q95
    q95.append(project_dict['sims'][i]['Q95'])
    Bn_Div_Li.append(project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI'])
    for type in ['plasma', 'vac']:
        try:
            if type == 'plasma':
                dir = project_dict['sims'][i]['dir_dict']['mars_plasma_dir']
            else:
                dir = project_dict['sims'][i]['dir_dict']['mars_vac_dir']

            r_array, z_array = post_mars_r_z(dir)
            r_array = r_array * project_dict['sims'][i]['R0EXP']
            z_array = z_array * project_dict['sims'][i]['R0EXP']
            Br = extractB(dir,'Br')
            Bz = extractB(dir,'Bz')
            Bphi = extractB(dir,'Bphi')

            if type == 'plasma':
                #project_dict['sims'][i]['plasma_response1'] = coil_responses2(r_array,z_array,Br,Bz,Bphi)
                #project_dict['sims'][i]['plasma_response2'] = coil_responses_single(r_array,z_array,Br,Bz,Bphi)
                #project_dict['sims'][i]['plasma_response3'] = coil_responses3(r_array,z_array,Br,Bz,Bphi)
                project_dict['sims'][i]['plasma_response4'] = coil_responses4(r_array,z_array,Br,Bz,Bphi)

            else:
                #project_dict['sims'][i]['vacuum_response1'] = coil_responses2(r_array,z_array,Br,Bz,Bphi)
                #project_dict['sims'][i]['vacuum_response2'] = coil_responses_single(r_array,z_array,Br,Bz,Bphi)
                #project_dict['sims'][i]['vacuum_response3'] = coil_responses3(r_array,z_array,Br,Bz,Bphi)
                project_dict['sims'][i]['vacuum_response4'] = coil_responses4(r_array,z_array,Br,Bz,Bphi)

#                print 'vac', project_dict['sims'][i]['vacuum_response']

            R=2.34 #2.34
            Z=0.0
            #Br_val, Bz_val, Bphi_val, R_val, Z_val = find_r_z(r_array, z_array, R, Z, Br, Bz, Bphi)

            #Br_val_list.append(Br_val) 
            #Bz_val_list.append(Bz_val)
            #Bphi_val_list.append(Bphi_val)
            #R_val_list.append(R_val)
            #Z_val_list.append(Z_val)

            success+=1
            #difference_percent =10
            print 'succeeded!!! %d' %(success)#, R,Z req: %.4f,%.4f; R,Z rec:%.4f,%.4f'%(success, R, Z, R_val, Z_val)
            ##project_dict['sims'][i]['RESULTS']=[r_array, z_array, Br, Bz, Bphi]
            file_log = open('/u/haskeysr/calc_log.log','a')
            file_log.write('success !!!  %d\n'%(success))
            file_log.close()

        except:
            print 'sorry failed.... %d'%(fail)
            fail +=1
            file_log = open('/u/haskeysr/calc_log.log','a')
            file_log.write('sorry failed.... %d\n'%(fail))
            file_log.close()


project_name = 'project1'
pickle_file = open('/u/haskeysr/mars/'+ project_name +'/9_coil_outputs_TESTnewInterp.pickle','w')
pickle.dump(project_dict, pickle_file)
pickle_file.close()

print 'sucess %d, fail %d'%(success,fail)


#Br_abs = num.abs(Br_val)
#Bz_abs = num.abs(Bz_val)
#Bphi_abs = num.abs(Bphi_val)

#list_data = [q95, Bn_Div_Li, R_val_list, Z_val_list, Br_val_list, Bz_val_list, Bphi_val_list]

#file_name = 'RESULTS_Plasma_r_%.2f_z_%.1f'%(R,Z)
#pickle.dump(list_data,open(file_name,'w'))

#print 'start pickle dump of main file'
#file_name = '9_interpret_results.pickle'
#pickle.dump(list_data,open(file_name,'w'))
#print 'finished'

'''
import scipy.interpolate as interpolate

#newfunc = interpolate.Rbf(q95,Bn_Div_Li,B1_abs,function='cubic')
newfunc = interpolate.Rbf(num.array(q95),num.array(Bn_Div_Li),num.array(B1_abs),function='cubic')
#ewfunc = interpolate.interp2d(q95,Bn_Div_Li,B1_abs,kind='cubic')

xnew,ynew = num.mgrid[2.15:3.7:100j, 0.36:2.7:100j]
newvals = newfunc(xnew,ynew)
fig = pt.figure()
ax = fig.add_subplot(111)
ax.imshow(newvals,extent=[2.15,3.7,0.36,2.7])
fig.savefig('temp13.png')



import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
# make up some randomly distributed data
npts = 200
x = Bn_Div_Li
y = q95
z = B1_abs
# define grid.
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.

CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('griddata test (%d points)' % npts)
plt.show()



fig = pt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.plot(q,p,'.')
ax.set_xlabel('qmult')
ax.set_ylabel('pmult')

ax2.plot(BnLI,q95,'.')
ax2.set_xlabel('B_N')
ax2.set_ylabel('q95')
fig.savefig('temp2.png')
'''
