import numpy as num
import PythonMARS_funcs as pyMARS
import os, copy
from RZfuncs import *
import matplotlib.pyplot as pt
import mpl_toolkits.mplot3d.axes3d as p3
import biot_funcs
import magnetics_details as mag_details

directory = '/u/haskeysr/mars/grid_check10/qmult1.000/exp1.000/marsrun/RUNrfa.vac'
directory = '/home/srh112/Desktop/Test_Case/test_shot/marsrun/RUN_rfa_upper.p/'
Nchi = 513; N = 6; n = 2; I = num.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)

os.chdir(directory) 
chi = num.linspace(num.pi*-1,num.pi,Nchi)
chi.resize(1,len(chi))
phi = num.linspace(0,2.*num.pi,Nchi)
phi.resize(len(phi),1)

file_name = 'RMZM_F'
RM, ZM, Ns, Ns1, Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
Nm2 = Nm0
R, Z =  GetRZ(RM,ZM,Nm0,Nm2,chi,phi)
FEEDI = get_FEEDI('FEEDI')
BNORM = calc_BNORM(FEEDI, R0EXP, I0EXP = I0EXP)

R=R[0:250,:]*R0EXP; Z=Z[0:250,:]*R0EXP


###############################
print 'start calculation'
fig=pt.figure(); ax = p3.Axes3D(fig)

#upper I-coil array points
n_I_coil = 15.
r_upper = [2.375, 2.164]; z_upper = [0.504, 1.012]
r_lower = [2.164 , 2.375]; z_lower = [-1.012, -0.504]

phi_zero = mag_details.coils.phi('I_coils_upper')
#phi_zero = range(0,360,60) #location of the I-coils
phi_range = mag_details.coils.width('I_coils_upper')
simulation_phi = phi_zero[0]
#phi_range = 45. #phi width of the I-coils

coil_point_list = biot_funcs.I_coil_points(r_upper, z_upper, r_lower, z_lower, n_I_coil, phi_range, phi_zero, ax = ax)


ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

points = []
I = num.array([1., 1., -1., -1., 0., 0.,1., 1., -1., -1., 0., 0.])
coil_currents = I*1000.
cyl_phi = R*0 + (simulation_phi)/180.*num.pi
xyz_X = R*num.cos(cyl_phi)
xyz_Y = R*num.sin(cyl_phi)
xyz_Z = Z

for i in range(0, int(Ns1+20), 20):
    ax.plot(xyz_X[i,:], xyz_Y[i,:], xyz_Z[i,:], 'r-')
    for phi_tmp in range(90,360,90):
        cyl_phi = R*0 + (phi_tmp)/180.*num.pi
        xyz_X_tmp = R*num.cos(cyl_phi)
        xyz_Y_tmp = R*num.sin(cyl_phi)
        xyz_Z_tmp = Z
        ax.plot(xyz_X_tmp[i,:], xyz_Y_tmp[i,:], xyz_Z_tmp[i,:], 'b-')
ax.set_xlim([-2.5,2.5]); ax.set_ylim([-2.5,2.5]); ax.set_zlim([-2.5,2.5])
fig.canvas.draw(); fig.show()


Bx, By, Bz = biot_funcs.biot_calc(coil_point_list, xyz_X, xyz_Y, xyz_Z, coil_currents, multi_proc=4)

print 'start plot'

fig1 = pt.figure()
ax1 = fig1.add_subplot(111)
color_plot = ax1.pcolor(R, Z, num.sqrt(Bx**2+By**2+Bz**2),cmap = 'spectral',edgecolors=None)
#color_plot = ax1.pcolor(xyz_X, xyz_Y, Bx**2,cmap = 'spectral',edgecolors='blue')
ax1.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
pt.colorbar(color_plot, ax=ax1)
color_plot.set_clim([-20,20])
#color_plot.set_clim([-20,20])

ax1.set_xlim([1,2.5])
ax1.set_ylim([-1.5,1.5])

fig1.canvas.draw()
fig1.show()


probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
#probe type 1: poloidal field, 2: radial field
probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
# Poloidal geometry
Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

results =  pyMARS.coil_responses6(xyz_X,xyz_Z,Bx,Bz,By, probe, probe_type, Rprobe,Zprobe,tprobe,lprobe)
print results
for tmp in range(0,len(probe)):
    print probe[tmp], num.average(results[tmp]), num.angle(num.average(results[tmp]),deg=True), num.abs(num.average(results[tmp]))

