from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as num
import matplotlib.pyplot as pt
import time
import PythonMARS_funcs as pyMARS

def coil_responses6(r_array,z_array,Br,Bz,Bphi, probe, probe_type, Rprobe,Zprobe,tprobe,lprobe):
    #probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
    # probe type 1: poloidal field, 2: radial field
    #probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
    # Poloidal geometry
    #Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
    #Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
    #tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
    #lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe
    Nprobe = len(probe)

    Navg = 20    # points along probe to interpolate
    Bprobem = []; Rprobek_total = []; Zprobek_total = []

    start_time = time.time()
    for k in range(0, Nprobe):
        #depending on poloidal/radial - what is really going on here? why is there a difference between the two cases?
        if probe_type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
        Rprobek_total.append(Rprobek)
        Zprobek_total.append(Zprobek)

    R_tot_array = num.array(Rprobek_total)
    Z_tot_array = num.array(Zprobek_total)

    #must be linear interpolation, otherwise there are sometimes problems
    Brprobek = num.resize(scipy_griddata((r_array.flatten(),z_array.flatten()), Br.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear'),R_tot_array.shape)
    Bzprobek = num.resize(scipy_griddata((r_array.flatten(),z_array.flatten()), Bz.flatten(), (R_tot_array.flatten(),Z_tot_array.flatten()),method='linear'),Z_tot_array.shape)
    Bprobem = []
    
    #print num.abs(Brprobek)
    #print num.abs(Bzprobek)
    #print num.mean(num.abs(Brprobek))

    interp_values = ((num.sin(tprobe[k])*num.real(Bzprobek[k,:]) + num.cos(tprobe[k])*num.real(Brprobek[k,:])) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek[k,:]) +num.cos(tprobe[k])*num.imag(Brprobek[k,:])))
    #calculate normal to coil and average over data points
    for k in range(0, Nprobe):
        Bprobem.append(num.average(interp_values))
        quantity = num.average(interp_values)
        print '%s, ave real: %.3f, ave imag: %.3f, output mag: %.3f, output phase: %.3fdeg '%(probe,quantity.real,quantity.imag, num.abs(quantity),num.angle(quantity,deg=True))
    print 'total time :', time.time() - start_time 
    return R_tot_array, Z_tot_array, interp_values, Bprobem



def extract_data(base_dir, I0EXP, ul=0):
    if ul==0:
        c = data(base_dir + 'RUNrfa.p', I0EXP = I0EXP)
        d = data(base_dir + 'RUNrfa.vac', I0EXP = I0EXP)
        return (c,d)
    else:
        a = data(base_dir + 'RUN_rfa_lower.p', I0EXP = I0EXP)
        b = data(base_dir + 'RUN_rfa_lower.vac', I0EXP = I0EXP)
        c = data(base_dir + 'RUN_rfa_upper.p', I0EXP = I0EXP)
        d = data(base_dir + 'RUN_rfa_upper.vac', I0EXP = I0EXP)
        return (a,b,c,d)


def combine_fields(input_data, attr_name, theta = 0, field_type='plas'):
    print 'combining property : ', attr_name
    if len(input_data)==2:
        if field_type=='plas':
            answer = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
        elif field_type == 'total':
            answer = getattr(input_data[0], attr_name)
        elif field_type == 'vac':
            answer = getattr(input_data[1], attr_name)
        print 'already combined, ', field_type
    elif len(input_data) == 4:
        if field_type=='plas':
            lower_data = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
            upper_data = getattr(input_data[2], attr_name) - getattr(input_data[3], attr_name)
        elif field_type=='total':
            lower_data = getattr(input_data[0], attr_name)
            upper_data = getattr(input_data[2], attr_name)
        elif field_type=='vac':
            lower_data = getattr(input_data[1], attr_name)
            upper_data = getattr(input_data[3], attr_name)
        answer = upper_data + lower_data*(num.cos(theta)+1j*num.sin(theta))
        print 'combine %s, theta : %.2f'%(field_type, theta)
    return answer

N = 6; n = 2; I = num.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)
print I0EXP, 1.0e+3 * 3./num.pi

#base_dir = '/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/'
#base_dir = '/home/srh112/code/pyMARS/shot146382_single2/qmult1.000/exp1.000/marsrun/'
base_dir = '/home/srh112/code/pyMARS/shot146382_single_ul/qmult1.000/exp1.000/marsrun/'
#base_dir = '/home/srh112/code/pyMARS/shot146382_single_tog/qmult1.000/exp1.000/marsrun/'
#base_dir = '/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/'


ul = 1; theta = 240./180*num.pi; plot_field = 'Bz'; field_type = 'plas'

run_data = extract_data(base_dir, I0EXP, ul=ul)
plot_quantity = combine_fields(run_data, plot_field, theta=theta, field_type=field_type)

title = plot_field
fig = pt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)


colour_plot = 0
if colour_plot==1:
    amp_plot = run_data[0].plot_Bn(num.abs(plot_quantity), ax1,start_surface = 0,end_surface = run_data[0].Ns1+30, skip=1, cmap='spectral',plot_coils_switch=1, plot_boundaries=1, wall_grid = run_data[0].NW)
    amp_plot2 = run_data[0].plot_Bn(num.angle(plot_quantity,deg=True), ax2,start_surface = 0,end_surface = run_data[0].Ns1+30, skip=1, cmap='hsv',plot_coils_switch=1, plot_boundaries = 1, wall_grid = run_data[0].NW)

    cbar1 = pt.colorbar(amp_plot,ax=ax1)
    cbar1 = pt.colorbar(amp_plot2,ax=ax2)
    amp_plot.set_clim([0,15])
    amp_plot2.set_clim([-180,180])
    ax1.set_xlabel('R (m)')
    ax2.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax2.set_ylabel('Z (m)')
    ax1.set_title(title+' Mag')
    ax2.set_title(title + ' Phase')
    fig.canvas.draw()
    fig.show()

probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad', 'MPI11M067','MPI2A067','MPI2B067', 'MPI66M067']
#probe type 1: poloidal field, 2: radial field
probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1, 0, 1, 1, 1, 1])
#Poloidal geometry
Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1., 0.973, 0.973, 0.972, 2.413,])
Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0., -0.004, 0.518, -0.518, 0.003])
tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0., 90.0, 89.8, 89.8, -89.9])*2*num.pi/360  #DTOR # poloidal inclination
lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05, 0.141, 0.140, 0.141, 0.141])  # Length of probe

Br = combine_fields(run_data, 'Br', theta = theta, field_type=field_type)
#Br = c.Br - d.Br
Bz = combine_fields(run_data, 'Bz', theta = theta, field_type=field_type)
#Bz = c.Bz - d.Bz
#Bphi = c.Bphi - d.Bphi
Bphi = combine_fields(run_data, 'Bphi', theta = theta, field_type=field_type)
grid_r = run_data[0].R*run_data[0].R0EXP
grid_z = run_data[0].Z*run_data[0].R0EXP



fig = pt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex = ax)

origin = num.mean(grid_r[0,:])
origin = run_data[0].R0EXP

for surface_num in [2,3]:
    surface = run_data[0].Ns1+run_data[0].NW - surface_num
    ax.plot(num.arctan2(grid_z[surface,:],grid_r[surface,:]-origin)*180/num.pi, num.abs(plot_quantity[surface,:]), '.', label = 'wall-'+str(surface_num))
    ax2.plot(num.arctan2(grid_z[surface,:],grid_r[surface,:]-origin)*180/num.pi, num.angle(plot_quantity[surface,:],deg=True), '.', label = 'wall-'+str(surface_num))
y_limits = num.array(ax.get_ylim())

josh_values_x = [0, 150, 180, -140, 0]
josh_values_y = [1.8, 0.9, 0.45, 0.9, 1.8]
josh_values_error = [0.2,0.2,0.2,0.2,0.2]
ax.errorbar(josh_values_x, josh_values_y, yerr=josh_values_error,fmt='*')
#angle = num.arctan2(0.4, 2.431-num.mean(grid_r[0,:]))*180./num.pi
ax2.set_xlabel(r'$\theta - deg - real space$')
ax2.set_ylabel('Phase (deg)')
ax.set_ylabel('mag G/kA')
ax.set_title(plot_field)

i_range = 1
i_range = [9,10,11,12]
for tmp in range(0,len(i_range)):
    i = i_range[tmp]
    R_tot_array, Z_tot_array, interp_values, Bprobem = coil_responses6(grid_r, grid_z, Br, Bz, Bphi, [probe[i]], [probe_type[i]], [Rprobe[i]],[Zprobe[i]],[tprobe[i]],[lprobe[i]])
    angles = num.arctan2(Z_tot_array, R_tot_array-origin)*180./num.pi
    ax.plot(angles.flatten(), num.abs(interp_values).flatten(),'o',label=probe[i])
    ax2.plot(angles.flatten(), num.angle(interp_values,deg=True).flatten(),'o',label=probe[i])
ax.legend()
fig.canvas.draw()
fig.show()
 
