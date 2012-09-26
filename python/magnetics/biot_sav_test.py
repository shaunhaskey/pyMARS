
# Uses Biot Savart to calculate the field on a MARS cross-section. Or any other points you want.
# The coil model can come from the description of the I-coils in magnetics_details or
# from the SURFMN details. The biot-Sav results can be plotted next to MARS-F results if a MARS-F
# simulation directory is supplied.

# Additionally, the predicted output from a pickup coil can be calculated using this model - however,
# note that it doesn't take the effect of hte walls shielding into account.
# SH 6 Sept 2012


import numpy as np
import PythonMARS_funcs as pyMARS
import os, copy
from RZfuncs import *
import matplotlib.pyplot as pt
import mpl_toolkits.mplot3d.axes3d as p3
import biot_funcs
import results_class
import time
import magnetics_details as mag_details

fig=pt.figure(); ax = p3.Axes3D(fig)

start_working_dir = os.getcwd()

dump_data = 1
read_dumped_pickle = 1

directory = '/home/srh112/Documents/Work_Documents/My_Posters_Papers_Talks/n2expPaper/grid_vacuum_checks/grid_check10/qmult1.000/exp1.000/marsrun/RUNrfa.vac'
#directory = '/home/srh112/Desktop/Test_Case/test_shot/marsrun/RUN_rfa_upper.p/'
Nchi = 513; N = 6; n = 2; I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)

os.chdir(directory) 
chi = np.linspace(np.pi*-1,np.pi,Nchi)
chi.resize(1,len(chi))
phi = np.linspace(0,2.*np.pi,Nchi)
phi.resize(len(phi),1)

file_name = 'RMZM_F'
RM, ZM, Ns, Ns1, Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
print Ns, Ns1, Ns2
Nm2 = Nm0
R, Z =  GetRZ(RM,ZM,Nm0,Nm2,chi,phi)
FEEDI = get_FEEDI('FEEDI')
BNORM = calc_BNORM(FEEDI, R0EXP, I0EXP = I0EXP)
last_calc_surf = 250#int(Ns1+1)
last_calc_surf = 300
#last_calc_surf = int(Ns1+1)

R=R[0:last_calc_surf,:]*R0EXP; Z=Z[0:last_calc_surf,:]*R0EXP

###############################
print 'start calculation'

#upper I-coil array points
n_I_coil = 30
n_I_coil = 60
r_upper = [2.375, 2.164]; z_upper = [0.504, 1.012]
r_lower = [2.164 , 2.375]; z_lower = [-1.012, -0.504]

phi_zero = mag_details.coils.phi('I_coils_upper')
#phi_zero = phi_zero - phi_zero[0] - this is to move everything to zero... for the 

#phi_zero = range(0,360,60) #location of the I-coils
phi_range = mag_details.coils.width('I_coils_upper')
#simulation_phi = phi_zero[0]
#phi_range = 45. #phi width of the I-coils


#phi_zero = range(0,360,60) #location of the I-coils
#phi_range = phi_zero*0+45. #phi width of the I-coils

coil_point_list = []

#phi_cross_section = 0.#150.
phi_cross_section = phi_zero[0]#150.
#phi_cross_section = 0
from_surfmn = 0

if from_surfmn==1:
    coil_point_list = biot_funcs.surfmn_coil_points(ax)
else:
    for j, tmp in enumerate(phi_zero):
        #coil_points1 = generate_coil_points(r, z, n, tmp, phi_range)
        coil_points_up = biot_funcs.generate_coil_points2(r_upper, z_upper, n_I_coil, tmp, phi_range[j], ax)
        coil_points_lower = biot_funcs.generate_coil_points2(r_lower, z_lower, n_I_coil, tmp, phi_range[j], ax)
        tmp_array_up = np.array(coil_points_up)
        tmp_array_lower = np.array(coil_points_lower)
        #ax.scatter3D(tmp_array_up[:,0],tmp_array_up[:,1],tmp_array_up[:,2])
        #ax.scatter3D(tmp_array_lower[:,0],tmp_array_lower[:,1],tmp_array_lower[:,2])
        ax.plot(tmp_array_up[:,0],tmp_array_up[:,1],tmp_array_up[:,2],'b-o')
        ax.plot(tmp_array_lower[:,0],tmp_array_lower[:,1],tmp_array_lower[:,2], 'b-o')
        for i in range(0, tmp_array_up.shape[0]):
            pass
            #ax.text3D(tmp_array_up[i,0],tmp_array_up[i,1],tmp_array_up[i,2], str(i), fontsize=8)
            #ax.text3D(tmp_array_lower[i,0],tmp_array_lower[i,1],tmp_array_lower[i,2], str(i),fontsize=8)
        coil_point_list.append(coil_points_up)
        coil_point_list.append(coil_points_lower)

#points = []
coil_currents = np.array([1.,1.,-1.,-1.,0.,0., 1.,1.,-1.,-1.,0.,0.])*1000.
cyl_phi = R*0 + (phi_cross_section)/180.*np.pi
xyz_X = R*np.cos(cyl_phi)
xyz_Y = R*np.sin(cyl_phi)
xyz_Z = Z
#ax.plot(xyz_X[i,:], xyz_Y[i,:], xyz_Z[i,:], 'r-')

for i in range(0, min(int(Ns1+20), last_calc_surf), 20):
    ax.plot(xyz_X[i,:], xyz_Y[i,:], xyz_Z[i,:], 'r-')
    for phi_tmp in range(90,360,90):
        cyl_phi = R*0 + (phi_tmp)/180.*np.pi
        xyz_X_tmp = R*np.cos(cyl_phi)
        xyz_Y_tmp = R*np.sin(cyl_phi)
        xyz_Z_tmp = Z
        ax.plot(xyz_X_tmp[i,:], xyz_Y_tmp[i,:], xyz_Z_tmp[i,:], 'b-')



probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
#probe type 1: poloidal field, 2: radial field
probe_type   = np.array([     1,     1,     1,     0,     0,     0,     0, 1,0])

# Poloidal geometry
Rprobe = np.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
Zprobe = np.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
tprobe = np.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*np.pi/360  #DTOR # poloidal inclination
lprobe = np.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe


phi_loc = [phi_zero[0],phi_zero[0],phi_zero[0]] # [17.2,17.8,18.7]
phi_width = [60.3,61.5,59.7]

xyz_X_list, xyz_Y_list, xyz_Z_list, phi_array_list = biot_funcs.generate_pickupcoil_points(probe[4:7], probe_type[4:7], Rprobe[4:7], Zprobe[4:7], tprobe[4:7], lprobe[4:7], phi_loc, phi_width, n_phi_width=100, Navg=100)

for i in range(0,len(xyz_X_list)):
    ax.plot(xyz_X_list[i].flatten(), xyz_Y_list[i].flatten(), xyz_Z_list[i].flatten(), 'o')

ax.set_xlim([-2.5,2.5]); ax.set_ylim([-2.5,2.5]); ax.set_zlim([-2.5,2.5])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
fig.canvas.draw(); fig.show()

answers = []
for i in range(0,len(xyz_X_list)):
    Bx, By, Bz = biot_funcs.biot_calc(coil_point_list, xyz_X_list[i], xyz_Y_list[i], xyz_Z_list[i], coil_currents, multi_proc=3)
    
    #Br = np.sqrt(Bx**2 + By**2)
    k = i+3
    Br = Bx*np.cos(phi_array_list[i]) + By*np.sin(phi_array_list[i])
    coil_output = (np.sin(tprobe[k])*np.real(Bz) + np.cos(tprobe[k])*np.real(Br)) + 1j * (np.sin(tprobe[k])*np.imag(Bz) +np.cos(tprobe[k])*np.imag(Br))
    answers.append(np.average(np.real(coil_output)))
    fig1, ax1 = pt.subplots()
    color_plot = ax1.pcolor(phi_array_list[i], xyz_Z_list[i], np.real(coil_output))
    color_plot.set_clim([-20,20])
    ax1.set_title(probe[4+i] + ' {}'.format(answers[-1]))
    pt.colorbar(color_plot,ax = ax1)
    fig1.canvas.draw(); fig1.show()
    
print answers


#exit()
biot_start_time = time.time()
Bx, By, Bz = biot_funcs.biot_calc(coil_point_list, xyz_X, xyz_Y, xyz_Z, coil_currents, multi_proc=3)
print 'biot total time = ', time.time() - biot_start_time
print 'start plot'





'''

        pool_size = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=pool_size)
        #xyz_X_list = []; xyz_Y_list = []; xyz_Z_list = []; dist_thresh_list=[]
        #for i in range(0,len(coil_point_list)):
        #    xyz_X_list.append(xyz_X); xyz_Y_list.append(xyz_Y); xyz_Z_list.append(xyz_Z)
        #    dist_thresh_list.append(dist_thresh)
        print 'creating pool map'
        B_list = pool.map(basic_calculation_wrapper, itertools.izip(coil_point_list, itertools.repeat(xyz_X), 
                                                     itertools.repeat(xyz_Y), itertools.repeat(xyz_Z), 
                                                     coil_currents, itertools.repeat(dist_thresh)))
        print B_list
        print len(B_list)
        #B_list = pool.map(basic_calculation, coil_point_list, xyz_X_list, xyz_Y_list, xyz_Z_list, coil_currents, dist_thresh_list)
        #B_list = map(basic_calculation, coil_point_list, xyz_X_list, xyz_Y_list, xyz_Z_list, coil_currents, dist_thresh_list)
        print 'closing pool'
        pool.close() # no more tasks
        print 'waiting for pool to finish'
        pool.join()  # wrap up current tasks
        print 'pool finished'
        #basic_calculation(coil_points, xyz_X, xyz_Y, xyz_Z, coil_currents, dist_thresh)


'''



'''
fig1 = pt.figure()
ax1 = fig1.add_subplot(111)
color_plot = ax1.pcolor(R, Z, np.sqrt(Bx**2+By**2+Bz**2),cmap = 'spectral')#,edgecolors='blue')
ax1.plot(R[Ns1,:], Z[Ns1,:],'b-')

#color_plot = ax1.pcolor(xyz_X, xyz_Y, Bx**2,cmap = 'spectral',edgecolors='blue')
ax1.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
pt.colorbar(color_plot, ax=ax1)
color_plot.set_clim([0,40])
fig1.canvas.draw()
fig1.show()
'''


MARS_results = results_class.data(directory,Nchi=513, I0EXP=I0EXP, spline_B23=2)
results =  pyMARS.coil_responses6(np.sqrt(xyz_X**2+xyz_Y**2),xyz_Z, Bx, Bz, By, probe, probe_type, Rprobe,Zprobe,tprobe,lprobe)
print results
results2 = pyMARS.coil_responses6(MARS_results.R*MARS_results.R0EXP, MARS_results.Z*MARS_results.R0EXP, MARS_results.Br, MARS_results.Bz, MARS_results.Bphi, probe, probe_type, Rprobe,Zprobe,tprobe,lprobe)
for tmp in range(0,len(probe)):
    print probe[tmp], np.average(results[tmp]), np.angle(np.average(results[tmp]),deg=True), np.abs(np.average(results[tmp]))
    print probe[tmp], np.average(results2[tmp]), np.angle(np.average(results2[tmp]),deg=True), np.abs(np.average(results2[tmp]))


#now I want to compare the whole grid to the MARS grid
#Need to extract the various stuff


diff_Z = (np.abs(Bz) - np.abs(np.real(MARS_results.Bz[0:last_calc_surf,:])))/np.abs(np.real(Bz)) * 100
diff_R = (np.abs(Bx) - np.abs(np.real(MARS_results.Br[0:last_calc_surf,:])))/np.abs(Bx) * 100

'''
fig1 = pt.figure()
ax1 = fig1.add_subplot(111)
color_plot = ax1.pcolor(R, Z, diff_Z, cmap = 'hot')#,edgecolors='blue')
#color_plot = ax1.pcolor(xyz_X, xyz_Y, Bx**2,cmap = 'spectral',edgecolors='blue')
ax1.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax1.plot(R[Ns1,:], Z[Ns1,:],'b-')
pt.colorbar(color_plot, ax=ax1)
color_plot.set_clim([0,100])
ax1.set_title('Diff Z')
fig1.canvas.draw()
fig1.show()

fig1 = pt.figure()
ax1 = fig1.add_subplot(111)
color_plot = ax1.pcolor(R, Z, diff_R, cmap = 'hot')#,edgecolors='blue')
#color_plot = ax1.pcolor(xyz_X, xyz_Y, Bx**2,cmap = 'spectral',edgecolors='blue')
ax1.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax1.plot(R[Ns1,:], Z[Ns1,:],'b-')
pt.colorbar(color_plot, ax=ax1)
ax1.set_title('Diff R')
color_plot.set_clim([0,100])
fig1.canvas.draw()
fig1.show()
'''


def color_plot_fig(R,Z,plot_biot, plot_MARS, title=None, clim = [-15,15], tol = 0., inc_diff = 0, cmap='spectral'):
    plot_start_time = time.time()
    if inc_diff:
        fig1, ax = pt.subplots(ncols=3, sharey=1, sharex=1)
    else:
        fig1, ax = pt.subplots(ncols=2, sharey=1, sharex=1)
    color_plot = []
    color_plot.append(ax[0].pcolor(R, Z, np.real(plot_biot), cmap = cmap)) #,edgecolors='blue')
    color_plot.append(ax[1].pcolor(R, Z, np.real(plot_MARS), cmap = cmap))#,edgecolors='blue')
    diff = (np.real(plot_MARS)-np.real(plot_biot))
    ave = np.abs((np.real(plot_MARS)+np.real(plot_biot))/2.)
    ave[np.where(ave<tol)] = tol
    #should this one be included?
    #diff[np.where(np.abs(diff)<tol)] = tol
    if inc_diff:
        color_plot.append(ax[2].pcolor(R, Z, np.abs(diff)/ave*100., cmap = 'hot'))#,edgecolors='blue')
        ax[2].set_title('% difference - ' + str(title))
        pt.colorbar(color_plot[2], ax=ax[2])
        color_plot[2].set_clim([0,100])
    ax[0].set_title('Biot-Savart')
    ax[1].set_title('MARS-F')
    fig1.suptitle(title+' Vacuum Field G/kA',fontsize=15)
    for i, tmp_ax in enumerate(ax):
        tmp_ax.plot(R[Ns1,:], Z[Ns1,:],'b-')
        tmp_ax.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
        tmp_ax.set_aspect('equal')
        tmp_ax.set_xlabel('R (m)')
    ax[0].set_ylabel('Z (m)')
    pt.colorbar(color_plot[1], ax=ax[1])
    pt.colorbar(color_plot[0], ax=ax[0])
    color_plot[0].set_clim(clim)
    color_plot[1].set_clim(clim)
    ax[0].set_xlim([1,2.5])
    ax[0].set_ylim([-1.3, 1.2])
    
    #fig1.savefig('/home/srh112/test.png')
    fig1.canvas.draw(); fig1.show()
    print 'plot took (s): ', -plot_start_time+time.time()


color_plot_fig(R,Z,-Bz, MARS_results.Bz[0:last_calc_surf,:], title='Bz', clim=[-3,3], tol=0.5, inc_diff = 0)

last_calc_surf=Ns1+1
color_plot_fig(R[0:last_calc_surf,:],Z[0:last_calc_surf,:],np.sqrt(Bz[0:last_calc_surf,:]**2+By[0:last_calc_surf,:]**2+Bx[0:last_calc_surf,:]**2), np.sqrt(MARS_results.Bz[0:last_calc_surf,:]**2+MARS_results.Br[0:last_calc_surf,:]**2+MARS_results.Bphi[0:last_calc_surf,:]**2), title='|B|', clim=[0,9], tol=0.5, inc_diff = 0, cmap = 'jet')
#color_plot_fig(R,Z,-Bx, MARS_results.Br[0:last_calc_surf,:], title='Br', clim=[-5,5])
#color_plot_fig(R,Z,By, MARS_results.Bphi[0:last_calc_surf,:], title='Bphi')

os.chdir(start_working_dir)

'''

#Comparison between Br/Bx
fig1 = pt.figure()
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)
color_plot = ax1.pcolor(R, Z, -np.real(Bx), cmap = 'spectral')#,edgecolors='blue')
color_plot2 = ax2.pcolor(R, Z,np.real(MARS_results.Br[0:last_calc_surf,:]), cmap = 'spectral')#,edgecolors='blue')
color_plot3 = ax3.pcolor(R, Z, np.abs((np.real(MARS_results.Br[0:last_calc_surf,:])+np.real(Bx))/np.real(Bx)*100.), cmap = 'hot')#,edgecolors='blue')
ax1.set_title('Biot - Bx')
ax2.set_title('MARS - Br')
ax3.set_title('% difference - Br')
ax1.plot(R[Ns1,:], Z[Ns1,:],'b-')
ax2.plot(R[Ns1,:], Z[Ns1,:],'b-')
ax3.plot(R[Ns1,:], Z[Ns1,:],'b-')

ax1.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax2.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax3.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
pt.colorbar(color_plot, ax=ax1)
pt.colorbar(color_plot2, ax=ax2)
pt.colorbar(color_plot3, ax=ax3)
color_plot.set_clim([-15,15])
color_plot2.set_clim([-15,15])
color_plot3.set_clim([0,100])
fig1.canvas.draw()
fig1.show()


#Comparison between Bz/Bz
fig1 = pt.figure()
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)
color_plot = ax1.pcolor(R, Z, -np.real(Bz), cmap = 'spectral')#,edgecolors='blue')
color_plot2 = ax2.pcolor(R, Z, np.real(MARS_results.Bz[0:last_calc_surf,:]), cmap = 'spectral')#,edgecolors='blue')
color_plot3 = ax3.pcolor(R, Z, np.abs((np.real(MARS_results.Bz[0:last_calc_surf,:])+np.real(Bz))/np.real(Bz)*100.), cmap = 'hot')#,edgecolors='blue')
ax1.set_title('Biot - Bz')
ax2.set_title('MARS - Bz')
ax3.set_title('% difference - Bz')
ax1.plot(R[Ns1,:], Z[Ns1,:],'b-')
ax2.plot(R[Ns1,:], Z[Ns1,:],'b-')
ax3.plot(R[Ns1,:], Z[Ns1,:],'b-')

ax1.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax2.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax3.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
pt.colorbar(color_plot, ax=ax1)
pt.colorbar(color_plot2, ax=ax2)
pt.colorbar(color_plot3, ax=ax3)
color_plot.set_clim([-15,15])
color_plot2.set_clim([-15,15])
color_plot3.set_clim([0,100])
fig1.canvas.draw()
fig1.show()


#Comparison between Bphi/Bphi
fig1 = pt.figure()
ax1 = fig1.add_subplot(131)
ax2 = fig1.add_subplot(132)
ax3 = fig1.add_subplot(133)
color_plot = ax1.pcolor(R, Z, np.real(By), cmap = 'spectral')#,edgecolors='blue')
color_plot2 = ax2.pcolor(R, Z, np.real(MARS_results.Bphi[0:last_calc_surf,:]), cmap = 'spectral')#,edgecolors='blue')
color_plot3 = ax3.pcolor(R, Z, np.abs((np.real(MARS_results.Bphi[0:last_calc_surf,:])+np.real(By))/np.real(By)*100.), cmap = 'hot')#,edgecolors='blue')
ax1.set_title('Biot - By')
ax2.set_title('MARS - Bphi')
ax3.set_title('% difference - Bz')
ax1.plot(R[Ns1,:], Z[Ns1,:],'b-')
ax2.plot(R[Ns1,:], Z[Ns1,:],'b-')
ax3.plot(R[Ns1,:], Z[Ns1,:],'b-')

ax1.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax2.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
ax3.plot([2.164,2.374,2.164,2.374], [1.012,0.504,-1.012,-0.504],'x')
pt.colorbar(color_plot, ax=ax1)
pt.colorbar(color_plot2, ax=ax2)
pt.colorbar(color_plot3, ax=ax3)
color_plot.set_clim([-5,5])
color_plot2.set_clim([-5,5])
color_plot3.set_clim([0,100])
fig1.canvas.draw()
fig1.show()
'''
