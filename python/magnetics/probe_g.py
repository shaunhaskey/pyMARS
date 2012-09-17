'''
This will run probe_g and SH biot-savart calculation and compare their values across the
face of a pickup coil. Produces the plots that show large spikes around the conductors
SH : 14 Sept 2012
'''

import numpy as np
import os, time
import PythonMARS_funcs as pyMARS
import results_class as res
import RZfuncs
import biot_funcs
import magnetics_details as mag_details

start_working_dir = os.getcwd()

probe_name = 'ISL'
N = 6; n = 2; I = np.array([1.,-1.,0.,1,-1.,0.])
number_interp_points = 800
phi_location = 32.7#85.#32.7#85

print 'phi_location %.2fdeg'%(phi_location)
template_dir = '/u/haskeysr/mars/templates/PROBE_G_TEMPLATE/'
base_run_dir = '/u/haskeysr/PROBE_G_RUNS/'
project_name = 'test2/'
run_dir = base_run_dir + project_name
print run_dir
os.system('mkdir '+run_dir)
os.system('cp -r ' + template_dir +'*  ' + run_dir)

print 'go to new directory'
os.chdir(run_dir + 'PROBE_G')

probe_g_template = file('probe_g.in', 'r')
probe_g_template_txt = probe_g_template.read()
probe_g_template.close()

diiid = file('diiid.in', 'r')
diiid_txt = diiid.read()
diiid.close()

#a, b = coil_responses6(1,1,1,1,1,1,Navg=120,default=1)

probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
# probe type 1: poloidal field, 2: radial field
probe_type   = np.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
# Poloidal geometry
Rprobe = np.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
Zprobe = np.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
tprobe = np.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*np.pi/360  #DTOR # poloidal inclination
lprobe = np.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

k = probe.index(probe_name)
#Generate interpolation points
Rprobek, Zprobek = pyMARS.pickup_interp_points(Rprobe[k], Zprobe[k], lprobe[k], tprobe[k], probe_type[k], number_interp_points)

coil_currents = np.array([1., 1., -1., -1., 0., 0.,1., 1., -1., -1., 0., 0.])*1000.


n_I_coil = 15.
r_upper = [2.375, 2.164]; z_upper = [0.504, 1.012]
r_lower = [2.164 , 2.375]; z_lower = [-1.012, -0.504]

#phi_zero = range(0,360,60) #location of the I-coils OLD
#phi_range = 45. #phi width of the I-coils OLD
#coil_point_list = biot_funcs.I_coil_points(r_upper, z_upper, r_lower, z_lower, n_I_coil, phi_range, phi_zero) OLD

####################
##NEW
phi_zero = mag_details.coils.phi('I_coils_upper')
phi_range = mag_details.coils.width('I_coils_upper')
coil_point_list = []
from_surfmn = 0

if from_surfmn==1:
    coil_point_list = biot_funcs.surfmn_coil_points(ax)
else:
    for j, tmp in enumerate(phi_zero):
        #coil_points1 = generate_coil_points(r, z, n, tmp, phi_range)
        coil_points_up = biot_funcs.generate_coil_points2(r_upper, z_upper, n_I_coil, tmp, phi_range[j])
        coil_points_lower = biot_funcs.generate_coil_points2(r_lower, z_lower, n_I_coil, tmp, phi_range[j])
        tmp_array_up = np.array(coil_points_up)
        tmp_array_lower = np.array(coil_points_lower)
        #ax.scatter3D(tmp_array_up[:,0],tmp_array_up[:,1],tmp_array_up[:,2])
        #ax.scatter3D(tmp_array_lower[:,0],tmp_array_lower[:,1],tmp_array_lower[:,2])
        #ax.plot(tmp_array_up[:,0],tmp_array_up[:,1],tmp_array_up[:,2],'b-o')
        #ax.plot(tmp_array_lower[:,0],tmp_array_lower[:,1],tmp_array_lower[:,2], 'b-o')
        for i in range(0, tmp_array_up.shape[0]):
            pass
            #ax.text3D(tmp_array_up[i,0],tmp_array_up[i,1],tmp_array_up[i,2], str(i), fontsize=8)
            #ax.text3D(tmp_array_lower[i,0],tmp_array_lower[i,1],tmp_array_lower[i,2], str(i),fontsize=8)
        coil_point_list.append(coil_points_up)
        coil_point_list.append(coil_points_lower)
phi_loc_tmp= [phi_zero[0]];phi_width_tmp=[0.]; n_phi_width = 1
xyz_X_list, xyz_Y_list, xyz_Z_list, phi_array_list = biot_funcs.generate_pickupcoil_points([probe[k]], [probe_type[k]], [Rprobe[k]], [Zprobe[k]], [tprobe[k]], [lprobe[k]], phi_loc_tmp, phi_width_tmp, n_phi_width=n_phi_width, Navg=number_interp_points)

#Bx_sh, By_sh, Bz_sh = biot_funcs.biot_calc(coil_point_list, Rprobek, Rprobek*0, Zprobek, coil_currents, dist_thresh=0./100, multi_proc=1)
Bx_sh, By_sh, Bz_sh = biot_funcs.biot_calc(coil_point_list, np.array(xyz_X_list), np.array(xyz_Y_list), np.array(xyz_Z_list), coil_currents, dist_thresh=0./100, multi_proc=1)
####

#Bx_sh, By_sh, Bz_sh = biot_funcs.biot_calc(coil_point_list, Rprobek, Rprobek*0, Zprobek, coil_currents, dist_thresh=0./100)


#Generate the points string and modify the .in file
r_flattened = Rprobek.flatten()
z_flattened = Zprobek.flatten()
phi_flattened = z_flattened * 0 + phi_location
points_string = ''
print len(r_flattened)
for i in range(0,len(r_flattened)):
    points_string+='%.3f   %.3f   %.3f\n'%(r_flattened[i], phi_flattened[i], z_flattened[i])

changes = {'<<npts>>' : str(len(r_flattened)),
          '<<points>>' : points_string}

for tmp_key in changes.keys():
    probe_g_template_txt = probe_g_template_txt.replace(tmp_key, changes[tmp_key])
probe_g_template = file('probe_g.in', 'w')
probe_g_template.write(probe_g_template_txt)
probe_g_template.close()

diiid_changes = {'<<upper>>': '1000 -1000 0 1000 -1000 0',
                '<<lower>>': '1000 -1000 0 1000 -1000 0'}

for tmp_key in diiid_changes:
    diiid_txt = diiid_txt.replace(tmp_key, diiid_changes[tmp_key])

diiid = file('diiid.in', 'w')
diiid.write(diiid_txt)
diiid.close()



#run probe_g
os.system('./probe_g')


#Read the output file
results = np.loadtxt('probe_gb.out', skiprows=8)
B_R = results[:,4]
B_phi =results[:,3]
B_Z= results[:,5]
phi_out = results[:,0]
R_out = results[:,1]
Z_out = results[:,2]
include_MARS = 1
if include_MARS:
    print 'get the answer from MARS'
    I0EXP = RZfuncs.I0EXP_calc(N,n,I)

    base_dir = '/u/haskeysr/mars/grid_check10/qmult1.000/exp1.000/marsrun/'
    Nchi=513

    #plas_run = res.data(base_dir + 'RUNrfa.p', I0EXP = I0EXP, Nchi=Nchi)
    vac_run = res.data(base_dir + 'RUNrfa.vac', I0EXP = I0EXP, Nchi=Nchi)


    grid_r = vac_run.R*vac_run.R0EXP
    grid_z = vac_run.Z*vac_run.R0EXP

    Brprobek, Bzprobek = pyMARS.pickup_field_interpolation(grid_r, grid_z, vac_run.Br, vac_run.Bz, vac_run.Bphi, np.array(Rprobek), np.array(Zprobek))

    MARS_pickup_output = (np.average((np.sin(tprobe[k])*np.real(Bzprobek) + np.cos(tprobe[k])*np.real(Brprobek)) + 1j * (np.sin(tprobe[k])*np.imag(Bzprobek) +np.cos(tprobe[k])*np.imag(Brprobek))))
biot_savart_pickup_output = (np.average((np.sin(tprobe[k])*np.real(B_Z) + np.cos(tprobe[k])*np.real(B_R)) + 1j * (np.sin(tprobe[k])*np.imag(B_Z) +np.cos(tprobe[k])*np.imag(B_R))))
biot_savart_sh_pickup_output = (np.average((np.sin(tprobe[k])*np.real(Bz_sh) + np.cos(tprobe[k])*np.real(Bx_sh)) + 1j * (np.sin(tprobe[k])*np.imag(Bz_sh) +np.cos(tprobe[k])*np.imag(Bx_sh))))


print 'biot :', biot_savart_pickup_output*10000., ' MARS : ', MARS_pickup_output, ' Biot_sh :', biot_savart_sh_pickup_output
    
import matplotlib.pyplot as pt
fig = pt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.plot(Rprobek, np.real(Brprobek),  'b-,', label='B_R MARS')
ax.plot(Rprobek, -np.real(B_R*10000), 'k--,', label='B_R PROBE_G')
ax.plot(Rprobek, -np.real(Bx_sh.flatten()), 'y--,', label='B_R sh Biot-Savart')

ax2.plot(Zprobek, np.real(Bzprobek), 'b-,', label = 'B_Z MARS')
ax2.plot(Zprobek, -np.real(B_Z*10000), 'k--,', label = 'B_Z PROBE_G')
ax2.plot(Rprobek, np.real(Bz_sh.flatten()), 'y,', label='B_Z sh Biot-Savart')

ax.legend(loc='best')
ax2.legend(loc='best')
ax.grid(b=True);ax2.grid(b=True)
ax.set_ylim(-50,50)
ax.set_title(probe[k])
ax.set_xlabel('R (m)')
ax2.set_xlabel('Z (m)')
ax.set_ylabel('G/kA')
ax2.set_ylabel('G/kA')
ax2.set_ylim(-50,50)

fig.canvas.draw()
fig.show()


fig = pt.figure()
ax = fig.add_subplot(111)
x_axis = np.linspace(0,lprobe[k],number_interp_points)*100
ax.plot(x_axis, -10000.*(np.sin(tprobe[k])*np.real(B_Z) + np.cos(tprobe[k])*np.real(B_R)),'k,-',label='Biot-Savart')
ax.plot(x_axis, -1*(np.sin(tprobe[k])*np.real(Bz_sh.flatten()) + np.cos(tprobe[k])*np.real(Bx_sh.flatten())),'yo',label='Biot-Savart_SH')
ax.plot(x_axis, np.sin(tprobe[k])*np.real(Bzprobek) + np.cos(tprobe[k])*np.real(Brprobek),'rx',label='MARS-F')

#ax.set_title(probe[k])
ax.set_title('Midplane radial pickup')
ax.set_xlabel('Location along pickup (cm)')
ax.set_ylabel('Field strength normal to pickup (G/kA)')
ax.legend(loc='best')
ax.grid(b=True)
ax.set_ylim([-20, 20])
fig.canvas.draw()
fig.show()
os.chdir(start_working_dir)
