'''
Generates plots of 'kink amplification' as a function of phasing
Will also create the files for an animation of plasma, vac, and total 
components in PEST co-ordinates

'''

import results_class
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt
import PythonMARS_funcs as pyMARS
from scipy.interpolate import griddata
import pickle
import matplotlib.cm as cm
#file_name = '/home/srh112/NAMP_datafiles/mars/shot146382_scan/shot146382_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot146394_3000_q95/shot146394_3000_q95_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/q95_scan_fine/shot146394_3000_q95_fine_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing/equal_spacing_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4/equal_spacing_n4_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4_V2/equal_spacing_n4_post_processing_PEST.pickle'
N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)
I0EXP = 1.0e+3*0.528 #PMZ n4 real


facn = 1.0; psi = 0.92
q_range = [2,6]; ylim = [0,1.4]
#phasing_range = [-180.,180.]
#phasing_range = [0.,360.]
phasing_range = [-90.,90.]
phase_machine_ntor = 1
make_animations = 0
include_discrete_comparison = 0
seperate_res_plot = 0
include_vert_lines = 0

plot_type = 'best_harmonic'
#plot_type = 'normalised'
#plot_type = 'normalised_average'
#plot_type = 'standard_average'

project_dict = pickle.load(file(file_name,'r'))
phasing = 0.
#phasing = np.arange(0.,360.,1)
print phasing
phasing = phasing/180.*np.pi
print phasing
q95_list = []; Bn_Li_list = []
phase_machine_ntor = 0
amps_vac_comp = []; amps_tot_comp = []; amps_plas_comp=[]; mk_list = []; pmult_list = []; qmult_list = []; serial_list = []
amps_plas_comp_upper = []; amps_plas_comp_lower = []
amps_vac_comp_upper = []; amps_vac_comp_lower = []
#upper_tot_res = []; lower_tot_res = []
#upper_vac_res = []; lower_vac_res = []
res_vac = []; res_tot = []; res_plas = []
res_vac_list_upper = []; res_vac_list_lower = []
res_plas_list_upper = []; res_plas_list_lower = []
for i in project_dict['sims'].keys():
    q95_list.append(project_dict['sims'][i]['Q95'])
    serial_list.append(i)
    Bn_Li_list.append(project_dict['sims'][i]['BETAN'])
    relevant_values_upper_tot = project_dict['sims'][i]['responses'][str(psi)]['total_kink_response_upper']
    relevant_values_lower_tot = project_dict['sims'][i]['responses'][str(psi)]['total_kink_response_lower']
    relevant_values_upper_vac = project_dict['sims'][i]['responses'][str(psi)]['vacuum_kink_response_upper']
    relevant_values_lower_vac = project_dict['sims'][i]['responses'][str(psi)]['vacuum_kink_response_lower']
    mk_list.append(project_dict['sims'][i]['responses'][str(psi)]['mk'])

    upper_tot_res = np.array(project_dict['sims'][i]['responses']['total_resonant_response_upper'])
    lower_tot_res = np.array(project_dict['sims'][i]['responses']['total_resonant_response_lower'])
    upper_vac_res = np.array(project_dict['sims'][i]['responses']['vacuum_resonant_response_upper'])
    lower_vac_res = np.array(project_dict['sims'][i]['responses']['vacuum_resonant_response_lower'])

    pmult_list.append(project_dict['sims'][i]['PMULT'])
    qmult_list.append(project_dict['sims'][i]['QMULT'])

    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))

    res_vac.append(np.sum(np.abs(upper_vac_res + lower_vac_res*phasor)))
    res_tot.append(np.sum(np.abs(upper_tot_res + lower_tot_res*phasor)))
    res_plas.append(np.sum(np.abs(upper_tot_res-upper_vac_res + (lower_tot_res-lower_vac_res)*phasor)))

    amps_vac_comp.append(relevant_values_upper_vac + relevant_values_lower_vac*phasor)
    amps_tot_comp.append(relevant_values_upper_tot + relevant_values_lower_tot*phasor)
    amps_plas_comp.append(relevant_values_upper_tot-relevant_values_upper_vac + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)

    #These do not have phasings applied to them yet
    amps_plas_comp_upper.append((relevant_values_upper_tot-relevant_values_upper_vac).tolist())
    amps_plas_comp_lower.append((relevant_values_lower_tot-relevant_values_lower_vac).tolist())
    amps_vac_comp_upper.append((relevant_values_upper_vac).tolist())
    amps_vac_comp_lower.append((relevant_values_lower_vac).tolist())

    res_vac_list_upper.append(upper_vac_res)
    res_vac_list_lower.append(lower_vac_res)
    res_plas_list_upper.append(upper_tot_res - upper_vac_res)
    res_plas_list_lower.append(lower_tot_res - lower_vac_res)



plot_quantity_vac=[];plot_quantity_plas=[];plot_quantity_tot=[];
plot_quantity_vac_phase=[];plot_quantity_plas_phase=[];plot_quantity_tot_phase=[];

plot_quantity = 'max'
max_loc_list = []; mode_list = []
upper_values_plasma = []; lower_values_plasma = []
upper_values_vac = []; lower_values_vac = []
for i in range(0,len(amps_vac_comp)):
    if plot_quantity == 'average':
        plot_quantity_vac.append(np.sum(np.abs(amps_vac_comp[i])**2)/len(amps_vac_comp[i]))
        plot_quantity_plas.append(np.sum(np.abs(amps_plas_comp[i])**2)/len(amps_vac_comp[i]))
        plot_quantity_tot.append(np.sum(np.abs(amps_tot_comp[i])**2)/len(amps_vac_comp[i]))
    elif plot_quantity == 'max':
        
        max_loc = np.argmax(np.abs(amps_plas_comp[i]))
        max_loc_list.append(max_loc)
        mode_list.append(mk_list[i][max_loc])
        plot_quantity_vac.append(np.abs(amps_vac_comp[i][max_loc]))
        plot_quantity_plas.append(np.abs(amps_plas_comp[i][max_loc]))
        plot_quantity_tot.append(np.abs(amps_tot_comp[i][max_loc]))
        plot_quantity_vac_phase.append(np.angle(amps_vac_comp[i][max_loc], deg = True))
        plot_quantity_plas_phase.append(np.angle(amps_plas_comp[i][max_loc], deg= True))
        plot_quantity_tot_phase.append(np.angle(amps_tot_comp[i][max_loc], deg = True))

        upper_values_plasma.append(amps_plas_comp_upper[i][max_loc])
        lower_values_plasma.append(amps_plas_comp_lower[i][max_loc])
        upper_values_vac.append(amps_vac_comp_upper[i][max_loc])
        lower_values_vac.append(amps_vac_comp_lower[i][max_loc])
        


xnew = np.linspace(2.,7.,200)
ynew = np.linspace(0.75,4.5,200)
xnew_grid, ynew_grid = np.meshgrid(xnew,ynew)
q95_Bn_array = np.zeros((len(q95_list),2),dtype=float)
q95_Bn_array[:,0] = q95_list[:]
q95_Bn_array[:,1] = Bn_Li_list[:]

q95_Bn_new = np.zeros((len(xnew),2),dtype=float)
q95_Bn_new[:,0] = xnew[:]
q95_Bn_new[:,1] = ynew[:]

plas_data = griddata(q95_Bn_array, plot_quantity_plas, (xnew_grid, ynew_grid),method = 'cubic')
vac_data = griddata(q95_Bn_array, plot_quantity_vac, (xnew_grid, ynew_grid), method = 'cubic')
tot_data = griddata(q95_Bn_array, plot_quantity_tot, (xnew_grid, ynew_grid), method = 'cubic')

plas_data_res = griddata(q95_Bn_array, res_plas, (xnew_grid, ynew_grid),method = 'cubic')
vac_data_res = griddata(q95_Bn_array, res_vac, (xnew_grid, ynew_grid), method = 'cubic')
tot_data_res = griddata(q95_Bn_array, res_tot, (xnew_grid, ynew_grid), method = 'cubic')

mode_data = griddata(q95_Bn_array, mode_list, (xnew_grid, ynew_grid), method = 'cubic')
plas_data_phase = griddata(q95_Bn_array, plot_quantity_plas_phase, (xnew_grid, ynew_grid), method = 'linear')
vac_data_phase = griddata(q95_Bn_array, plot_quantity_vac_phase, (xnew_grid, ynew_grid), method = 'linear')


for interp_meth in ['linear', 'cubic']:
    q95_single = np.linspace(3.,5.5,1000)
    Bn_Li_value = 1.83
    plas_data_single = griddata(q95_Bn_array, plot_quantity_plas, (q95_single, q95_single*0.+Bn_Li_value),method = interp_meth)
    vac_data_single = griddata(q95_Bn_array, plot_quantity_vac, (q95_single, q95_single*0.+Bn_Li_value), method = interp_meth)
    tot_data_single = griddata(q95_Bn_array, plot_quantity_tot, (q95_single, q95_single*0.+Bn_Li_value), method = interp_meth)
    mode_data_single = griddata(q95_Bn_array, mode_list, (q95_single, q95_single*0.+Bn_Li_value), method = interp_meth)

    fig,ax = pt.subplots()
    ax.plot(q95_single, plas_data_single, '.-', label='plas')
    ax.plot(q95_single, vac_data_single, '.-', label='vac')
    ax.plot(q95_single, tot_data_single, '.-', label='tot')
    ax.plot(q95_single, mode_data_single, '.-', label='m')
    ax.legend(loc='best')
    ax.set_title('Bn_Li:%.2f, %s interpolation, sqrt(psi)=%.2f'%(Bn_Li_value,interp_meth,psi))
    ax.set_xlabel('q95')
    ax.set_ylim([0,14])
    ax.set_ylabel('amplitude or mode number')
    fig.suptitle(file_name,fontsize=8)
    fig.canvas.draw(); fig.show()

fig,ax = pt.subplots(nrows = 3, sharex = 1, sharey = 1)
color_fig = ax[0].pcolor(xnew, ynew, np.ma.array(vac_data_res, mask=np.isnan(mode_data)))
color_fig.set_clim([0,7])
pt.colorbar(color_fig, ax = ax[0])
ax[0].set_title('vac')
color_fig = ax[1].pcolor(xnew, ynew, np.ma.array(plas_data_res, mask=np.isnan(mode_data)))
color_fig.set_clim([0,7])
pt.colorbar(color_fig, ax = ax[1])
ax[1].set_title('plas')
color_fig = ax[2].pcolor(xnew, ynew, np.ma.array(tot_data_res, mask=np.isnan(mode_data)))
color_fig.set_clim([0,7])
pt.colorbar(color_fig, ax = ax[2])
ax[2].set_title('tot')
fig.canvas.draw(); fig.show()


fig,ax = pt.subplots()
color_fig = ax.pcolor(xnew, ynew, tot_data)
if plot_quantity=='average':
    color_fig.set_clim([0,7])
elif plot_quantity=='max':
    color_fig.set_clim([0,7])





ax.set_title('total data')
fig.suptitle(file_name,fontsize=8)
fig.canvas.draw(); fig.show()

fig,ax = pt.subplots(nrows = 2,sharex = 1, sharey = 1)
plas_data_phase[plas_data_phase<=-40]+=360
color_fig = ax[0].pcolor(xnew, ynew, np.ma.array(plas_data_phase, mask=np.isnan(mode_data)))
color_fig.set_clim([-40,-40+360])
pt.colorbar(color_fig, ax=ax[0])
ax[0].plot(q95_list, Bn_Li_list,'k.')
ax[0].set_title('Plasma phase')
vac_data_phase[plas_data_phase<=-40]+=360
color_fig = ax[1].pcolor(xnew, ynew, np.ma.array(vac_data_phase, mask=np.isnan(mode_data)))
color_fig.set_clim([-40,-40+360])
ax[1].set_title('Vacuum phase')
ax[1].plot(q95_list, Bn_Li_list,'k.')
pt.colorbar(color_fig, ax=ax[1])
include_text = 0
if include_text:
    for ax_tmp in ax:
        for i in range(0,len(q95_list)):
            print 'ehllo'
            ax_tmp.text(q95_list[i], Bn_Li_list[i], str(serial_list[i]), fontsize = 7.5)
print_phases = 1
if print_phases:
    for i in range(0,len(q95_list)):
        print 'serial %d : phase %.2f deg'%(serial_list[i],plot_quantity_plas_phase[i])
    
fig.canvas.draw(); fig.show()


fig,ax = pt.subplots()
color_fig = ax.pcolor(xnew, ynew, np.ma.array(mode_data, mask=np.isnan(mode_data)))
color_fig.set_clim([5,15])

pt.colorbar(color_fig, ax=ax)
ax.plot(q95_list, Bn_Li_list,'k.')
ax.set_title('Max mode number, sqrt(psi)=%.2f'%(psi))
ax.set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
ax.set_xlabel(r'$q_{95}$', fontsize = 14)
fig.suptitle(file_name,fontsize=8)
fig.canvas.draw(); fig.show()


color_map = 'jet'
fig,ax = pt.subplots(nrows = 3,sharex = 1, sharey = 1)
color_fig_plas = ax[0].pcolor(xnew, ynew, np.ma.array(plas_data, mask=np.isnan(plas_data)),cmap=color_map)#, cmap = cmap)
#color_fig = ax[0].pcolor(xnew, ynew, plas_data)
pt.colorbar(color_fig_plas, ax = ax[0])
if plot_quantity=='average':
    color_fig_plas.set_clim([0,7])
elif plot_quantity=='max':
    #pass
    color_fig_plas.set_clim([0,2])
ax[0].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
ax[0].set_title('Plasma, sqrt(psi)=%.2f'%(psi))
ax[0].plot(q95_list, Bn_Li_list,'k.')
#fig.canvas.draw(); fig.show()

#fig,ax = pt.subplots()
#import matplotlib.cm as cm
#cmap = cm.jet
#cmap.set_bad('w',1.)

color_fig_vac = ax[1].pcolor(xnew, ynew, np.ma.array(vac_data, mask=np.isnan(vac_data)),cmap=color_map)#, cmap = cmap)
color_fig_relative = ax[2].pcolor(xnew, ynew, np.ma.array(plas_data/vac_data, mask=np.isnan(vac_data)),cmap=color_map)#, cmap = cmap)
color_fig_relative.set_clim([0,10])
pt.colorbar(color_fig_relative, ax = ax[2])
if plot_quantity=='average':
    color_fig_vac.set_clim([0,0.2])
elif plot_quantity=='max':
    color_fig_vac.set_clim([0,0.5])

pt.colorbar(color_fig_vac, ax = ax[1])
ax[1].set_title('Vacuum, sqrt(psi)=%.2f'%(psi))
ax[1].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
ax[2].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
ax[-1].set_xlabel(r'$q_{95}$', fontsize = 14)
ax[1].plot(q95_list, Bn_Li_list,'k.')
ax[2].plot(q95_list, Bn_Li_list,'k.')
ax[2].set_title('Plasma Amplification')
#ax[1].set_ylim([2,3])
fig.suptitle(file_name,fontsize=8)
fig.canvas.draw(); fig.show()

pmult_based_dict = {}

pmult_values = [0.9, 0.95, 1.0, 1.05, 1.1]
pmult_values = [1.0]
pmult_array = np.array(pmult_list)
fig, ax = pt.subplots(nrows = 3, sharex = 1)
for i in pmult_values:
    xaxis = np.ma.array(q95_list, mask = (pmult_array!=i))
    yaxis = np.ma.array(plot_quantity_plas, mask = (pmult_array!=i))
    pmult_axis = np.ma.array(pmult_array, mask = (pmult_array!=i))
    ax[0].plot(xaxis, yaxis, '.', label = 'pmult=%s'%(str(i)))
    yaxis = np.ma.array(mode_list, mask = (pmult_array!=i))
    ax[0].plot(xaxis, yaxis, 'o', label = 'm')
    yaxis = np.ma.array(plot_quantity_plas_phase, mask = (pmult_array!=i))
    ax[1].plot(xaxis, yaxis, '.', label = 'pmult=%s'%(str(i)))
    include_text_line = 1
    if include_text_line:
        for j in range(0, len(q95_list)):
            if pmult_array[j]==i:
                ax[0].text(q95_list[j], plot_quantity_plas[j], str(serial_list[j]),fontsize=8)
                ax[1].text(q95_list[j], plot_quantity_plas_phase[j], str(serial_list[j]),fontsize=8)
                

            
ax[1].grid(b=True)
ax[0].grid(b=True)
ax[0].set_title('PlasmaResponse, sqrt(psi)=%.2f'%(psi))
ax[0].set_ylabel('Amplitude', fontsize = 14)
ax[1].set_ylabel('Phase', fontsize = 14)
ax[1].set_xlabel(r'$q_{95}$', fontsize = 14)
fig.suptitle(file_name,fontsize=8)
leg = ax[0].legend(loc='best')
leg.get_frame().set_alpha(0.5)
fig.canvas.draw(); fig.show()


phasing_array = np.linspace(-180,180,360)
fig, ax = pt.subplots(nrows =2 , sharex = 1, sharey = 1)
pmult_values = [1.0]
pmult_array = np.array(pmult_list)
q95_array = np.array(q95_list)
rel_q95_vals = q95_array[pmult_array==pmult_values[0]]


rel_lower_vals_plasma = np.array(lower_values_plasma)[pmult_array==pmult_values[0]]
rel_upper_vals_plasma = np.array(upper_values_plasma)[pmult_array==pmult_values[0]]
rel_lower_vals_vac = np.array(lower_values_vac)[pmult_array==pmult_values[0]]
rel_upper_vals_vac = np.array(upper_values_vac)[pmult_array==pmult_values[0]]

Bn_Li_value = 2.2
q95_single = np.linspace(2.6,6,100)

rel_lower_vals_plasma = griddata(q95_Bn_array, np.array(lower_values_plasma), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
rel_upper_vals_plasma = griddata(q95_Bn_array, np.array(upper_values_plasma), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
rel_lower_vals_vac = griddata(q95_Bn_array, np.array(lower_values_vac), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
rel_upper_vals_vac = griddata(q95_Bn_array, np.array(upper_values_vac), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')



plot_array_plasma = np.ones((phasing_array.shape[0], rel_q95_vals.shape[0]),dtype=float)
plot_array_plasma = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)


plot_array_vac = np.ones((phasing_array.shape[0], rel_q95_vals.shape[0]),dtype=float)
plot_array_vac = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)


plot_array_vac_res = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
plot_array_plas_res = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)

for i, curr_phase in enumerate(phasing_array):
    phasor = (np.cos(curr_phase/180.*np.pi)+1j*np.sin(curr_phase/180.*np.pi))
    plot_array_plasma[i,:] = np.abs(rel_upper_vals_plasma + rel_lower_vals_plasma*phasor)
    plot_array_vac[i,:] = np.abs(rel_upper_vals_vac + rel_lower_vals_vac*phasor)



#color_plot = ax[0].pcolor(rel_q95_vals, phasing_array, plot_array_plasma, cmap='hot')
#color_plot2 = ax[1].pcolor(rel_q95_vals, phasing_array, plot_array_vac, cmap='hot')
color_plot = ax[0].pcolor(q95_single, phasing_array, plot_array_plasma, cmap='hot', rasterized=True)
color_plot2 = ax[1].pcolor(q95_single, phasing_array, plot_array_vac, cmap='hot', rasterized=True)
#color_plot.set_clim()
ax[0].plot(q95_single, phasing_array[np.argmax(plot_array_plasma,axis=0)],'k.')
ax[1].plot(q95_single, phasing_array[np.argmax(plot_array_vac,axis=0)],'k.')
ax[0].plot(q95_single, phasing_array[np.argmin(plot_array_plasma,axis=0)],'b.')
ax[1].plot(q95_single, phasing_array[np.argmin(plot_array_vac,axis=0)],'b.')

ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
ax[0].set_ylabel('Phasing (deg)')
ax[1].set_ylabel('Phasing (deg)')
ax[0].set_title('Kink Amplitude - Plasma')
ax[1].set_title('Kink Amplitude - Vacuum')
ax[0].set_xlim([np.min(q95_single),np.max(q95_single)])
ax[0].set_ylim([-180,180])
color_plot.set_clim([0, 2])
color_plot2.set_clim([0, 1])
cb = pt.colorbar(color_plot, ax = ax[0])
cb.ax.set_ylabel('Amp G/kA')
cb = pt.colorbar(color_plot2, ax = ax[1])
cb.ax.set_ylabel('Amp G/kA')
fig.canvas.draw(); fig.show()


res_vac_array_upper = np.array(res_vac_list_upper)
res_vac_array_lower = np.array(res_vac_list_lower)
res_plas_array_upper = np.array(res_plas_list_upper)
res_plas_array_lower = np.array(res_plas_list_lower)

for i, curr_phase in enumerate(phasing_array):
    print 'phase :', curr_phase
    phasor = (np.cos(curr_phase/180.*np.pi)+1j*np.sin(curr_phase/180.*np.pi))
    tmp_vac_list = []; tmp_plas_list = []
    for ii in range(0,len(res_vac_array_upper)):
        tmp_vac_list.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor)))
        tmp_plas_list.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor)))

    plot_array_vac_res[i,:] = griddata(q95_Bn_array, np.array(tmp_vac_list), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
    plot_array_plas_res[i,:] = griddata(q95_Bn_array, np.array(tmp_plas_list), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')

fig, ax = pt.subplots(); ax = [ax]#nrows = 2, sharex = 1, sharey = 1)
color_plot = ax[0].pcolor(q95_single, phasing_array, plot_array_vac_res, cmap='hot', rasterized=True)
color_plot.set_clim([0,10])
ax[0].set_xlim([2.6, 6])
ax[0].set_ylim([-180, 180])

pt.colorbar(color_plot, ax = ax[0])
#color_plot2 = ax[1].pcolor(q95_single, phasing_array, plot_array_plas_res, cmap='hot', rasterized=True)
#color_plot2.set_clim([0,10])
#pt.colorbar(color_plot2, ax = ax[1])
fig.canvas.draw(); fig.show()
            

# ax[1].grid(b=True)
# ax[0].grid(b=True)
# ax[0].set_title('PlasmaResponse, sqrt(psi)=%.2f'%(psi))
# ax[0].set_ylabel('Amplitude', fontsize = 14)
# ax[1].set_ylabel('Phase', fontsize = 14)
# ax[1].set_xlabel(r'$q_{95}$', fontsize = 14)
# fig.suptitle(file_name,fontsize=8)
# leg = ax[0].legend(loc='best')
# leg.get_frame().set_alpha(0.5)
# fig.canvas.draw(); fig.show()





'''
if plot_type == 'best_harmonic':
    plot_quantity_vac = np.abs(amps_vac_comp)[:,best_harmonic]
    plot_quantity_plas = np.abs(amps_plasma_comp)[:,best_harmonic]
    plot_quantity_tot = np.abs(amps_tot_comp)[:,best_harmonic]
elif plot_type == 'normalised':
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_vac_comp)**2,axis=1)), 'b-', label = 'Vacuum')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_plasma_comp)**2,axis=1)), 'r-', label = 'Plasma')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_tot_comp)**2,axis=1)), 'k-', label='Total')
elif plot_type == 'normalised_average':
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_vac_comp)**2,axis=1))/number_points, 'b-', label = 'Vacuum')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_plasma_comp)**2,axis=1))/number_points, 'r-', label = 'Plasma')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_tot_comp)**2,axis=1))/number_points, 'k-', label='Total')
elif plot_type == 'standard_average':
    ax[0].plot(phasings, np.sum(np.abs(amps_vac_comp),axis=1)/number_points, 'b-', label = 'Vacuum')
    ax[0].plot(phasings, np.sum(np.abs(amps_plasma_comp),axis=1)/number_points, 'r-', label = 'Plasma')
    ax[0].plot(phasings, np.sum(np.abs(amps_tot_comp),axis=1)/number_points, 'k-', label='Total')



#using a few different ones
#-sim
#base_dir = '/home/srh112/NAMP_datafiles/mars/shot146398_upper_lower/qmult1.000/exp1.000/marsrun/'
base_dir = '/home/srh112/NAMP_datafiles/mars/shot146394_upper_lower/qmult1.000/exp1.000/marsrun/'
upper_data_tot = results_class.data(base_dir + 'RUN_rfa_upper.p',I0EXP=I0EXP)
lower_data_tot = results_class.data(base_dir + 'RUN_rfa_lower.p', I0EXP=I0EXP)
upper_data_vac = results_class.data(base_dir + 'RUN_rfa_upper.vac',I0EXP=I0EXP)
lower_data_vac = results_class.data(base_dir + 'RUN_rfa_lower.vac', I0EXP=I0EXP)

upper_data_tot.get_PEST(facn = facn)
lower_data_tot.get_PEST(facn = facn)
upper_data_vac.get_PEST(facn = facn)
lower_data_vac.get_PEST(facn = facn)

mk_upper, ss_upper, relevant_values_upper_tot = upper_data_tot.kink_amp(psi, q_range, n = n)
mk_lower, ss_lower, relevant_values_lower_tot = lower_data_tot.kink_amp(psi, q_range, n = n)
mk_upper, ss_upper, relevant_values_upper_vac = upper_data_vac.kink_amp(psi, q_range, n = n)
mk_lower, ss_lower, relevant_values_lower_vac = lower_data_vac.kink_amp(psi, q_range, n = n)

a, upper_vac_res = upper_data_vac.resonant_strength()
a, lower_vac_res = lower_data_vac.resonant_strength()
a, upper_tot_res = upper_data_tot.resonant_strength()
a, lower_tot_res = lower_data_tot.resonant_strength()

number_points = len(relevant_values_lower_vac)
phasings = np.arange(phasing_range[0], phasing_range[1]+1,0.01)
#amps_tot = []; amps_vac = []; amps_plasma = []
amps_vac_comp = [];amps_tot_comp = [];amps_plasma_comp = []
if seperate_res_plot:
    fig, ax = pt.subplots(nrows = 2, sharex=1)
else:
    fig, ax = pt.subplots()
    ax = [ax]

for phasing in phasings:
    phasing = phasing/180.*np.pi
    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))

    amps_vac_comp.append(relevant_values_upper_vac + relevant_values_lower_vac*phasor)
    amps_tot_comp.append(relevant_values_upper_tot + relevant_values_lower_tot*phasor)
    amps_plasma_comp.append(relevant_values_upper_tot-relevant_values_upper_vac + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)
    
    #amps_vac.append(np.sum(np.abs(relevant_values_upper_vac + relevant_values_lower_vac*phasor))/number_points)
    #amps_tot.append(np.sum(np.abs(relevant_values_upper_tot + relevant_values_lower_tot*phasor))/number_points)
    #amps_plasma.append(np.sum(np.abs((relevant_values_upper_tot-relevant_values_upper_vac) + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor))/number_points)

tmp_loc = np.argmax(np.sum(np.abs(amps_tot_comp),axis=1)/number_points)
tmp_max_phasing = phasings[tmp_loc]
best_harmonic = np.argmax(np.abs(np.array(amps_tot_comp)[tmp_loc,:]))

print 'best_harmonic_loc: %d, m:%d, phasing machine:%.2f, phasing MARS:%.2f'%(best_harmonic, mk_upper[best_harmonic], tmp_max_phasing,-tmp_max_phasing*n)
#important_value = np.argmax((relevant_values_upper_tot-relevant_values_upper_vac) + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)

# for phasing in phasings:
#     phasing = phasing/180.*np.pi
#     if phase_machine_ntor:
#         phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
#     else:
#         phasor = (np.cos(phasing)+1j*np.sin(phasing))
#     amps_vac.append(np.sum(np.abs(relevant_values_upper_vac + relevant_values_lower_vac*phasor))/number_points)
#     amps_tot.append(np.sum(np.abs(relevant_values_upper_tot + relevant_values_lower_tot*phasor))/number_points)
#     amps_plasma.append(np.sum(np.abs((relevant_values_upper_tot-relevant_values_upper_vac) + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor))/number_points)

vac_qn = []
for phasing in phasings:
    phasing = phasing/180.*np.pi
    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
    vac_qn.append(np.abs(upper_vac_res + lower_vac_res*phasor))
    #amps_tot.append(np.sum(np.abs(relevant_values_upper_tot + relevant_values_lower_tot*phasor)))
    #amps_plasma.append(np.sum(np.abs((relevant_values_upper_tot-relevant_values_upper_vac) + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)))
vac_qn = np.array(vac_qn)
#plot_list = [0,1,2,3,4,5,6]
plot_list = [4]
for i, j  in enumerate(upper_data_tot.qn):
    if i in plot_list:
        if seperate_res_plot:
            ax[1].plot(phasings,vac_qn[:,i], color = 'gray', linestyle = '-', label = 'q=%.2f,m=%d'%(j, upper_data_tot.mq[i]))
        else:
            ax[0].plot(phasings,vac_qn[:,i], color = 'gray', linestyle = '-',  label = 'q=%.2f,m=%d'%(j, upper_data_tot.mq[i]))
        
if seperate_res_plot:
    ax[1].grid(); leg = ax[1].legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)


if plot_type == 'best_harmonic':
    ax[0].plot(phasings, np.abs(amps_vac_comp)[:,best_harmonic], 'b-', label = 'Vacuum')
    ax[0].plot(phasings, np.abs(amps_plasma_comp)[:,best_harmonic], 'r-', label = 'Plasma')
    ax[0].plot(phasings, np.abs(amps_tot_comp)[:,best_harmonic], 'k-', label='Total')
elif plot_type == 'normalised':
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_vac_comp)**2,axis=1)), 'b-', label = 'Vacuum')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_plasma_comp)**2,axis=1)), 'r-', label = 'Plasma')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_tot_comp)**2,axis=1)), 'k-', label='Total')
elif plot_type == 'normalised_average':
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_vac_comp)**2,axis=1))/number_points, 'b-', label = 'Vacuum')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_plasma_comp)**2,axis=1))/number_points, 'r-', label = 'Plasma')
    ax[0].plot(phasings, np.sqrt(np.sum(np.abs(amps_tot_comp)**2,axis=1))/number_points, 'k-', label='Total')
elif plot_type == 'standard_average':
    ax[0].plot(phasings, np.sum(np.abs(amps_vac_comp),axis=1)/number_points, 'b-', label = 'Vacuum')
    ax[0].plot(phasings, np.sum(np.abs(amps_plasma_comp),axis=1)/number_points, 'r-', label = 'Plasma')
    ax[0].plot(phasings, np.sum(np.abs(amps_tot_comp),axis=1)/number_points, 'k-', label='Total')

# if include_vert_lines:
#     max_loc = np.argmax(amps_tot); min_loc = np.argmin(amps_tot)
#     ax[0].vlines([phasings[max_loc],phasings[min_loc]],ylim[0],ylim[1])
#     max_loc = np.argmax(amps_vac); min_loc = np.argmin(amps_vac)
#     ax[0].vlines([phasings[max_loc],phasings[min_loc]],ylim[0],ylim[1])
#     max_loc = np.argmax(amps_plasma); min_loc = np.argmin(amps_plasma)
#     ax[0].vlines([phasings[max_loc], phasings[min_loc]],ylim[0],ylim[1])

if include_discrete_comparison:
    #single_answers
    single_phasings = range(0,360,60)
    #single_phasings = [0,120]
    single_data_vac_dict = {}
    single_data_tot_dict = {}

    for i in single_phasings:
        single_data_vac_dict[str(i)] = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_%ddeg/qmult1.000/exp1.000/marsrun/RUNrfa.vac'%(i),I0EXP=I0EXP)
        single_data_tot_dict[str(i)] = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_%ddeg/qmult1.000/exp1.000/marsrun/RUNrfa.p'%(i),I0EXP=I0EXP)

    kink_values_vac = []; kink_values_tot = [];kink_values_plas = []
    resonant_values_vac = []

    for i in single_phasings:
        single_data_vac_dict[str(i)].get_PEST(facn = facn)
        single_data_tot_dict[str(i)].get_PEST(facn = facn)
        mk_upper, ss_upper, relevant_tmp_vac = single_data_vac_dict[str(i)].kink_amp(psi, q_range, n = n)
        kink_values_vac.append(np.sum(np.abs(relevant_tmp_vac)))
        mk_upper, ss_upper, relevant_tmp_tot = single_data_tot_dict[str(i)].kink_amp(psi, q_range, n = n)
        kink_values_tot.append(np.sum(np.abs(relevant_tmp_tot)))
        kink_values_plas.append(np.sum(np.abs(relevant_tmp_tot-relevant_tmp_vac)))

        a, res_tmp_vac = single_data_vac_dict[str(i)].resonant_strength()
        #vac_qn.append(np.abs(upper_vac_res + lower_vac_res*phasor))
        resonant_values_vac.append(np.abs(res_tmp_vac))

    #for i in range(0, len(single_phasings)):
    #    if single_phasings[i]>phasing_range[1]:
    #        single_phasings[i] = single_phasings[i] - 360

    #plot_angles = np.array(single_angles)*n*-1
    if phase_machine_ntor:
        plot_angles = (np.array(single_phasings)*-1)/2.
    else:
        plot_angles = single_phasings

    for i in range(0, len(plot_angles)):
        while (plot_angles[i]>phasing_range[1]) or (plot_angles[i]<=phasing_range[0]):
            if plot_angles[i]>phasing_range[1]:
                plot_angles[i] = plot_angles[i] - 360
            elif plot_angles[i]<=phasing_range[0]:
                plot_angles[i] = plot_angles[i] + 360

    ax[0].plot(plot_angles,kink_values_tot, 'ks-')
    ax[0].plot(plot_angles,kink_values_vac, 'bs-')
    ax[0].plot(plot_angles,kink_values_plas, 'rs-')

    resonant_values_vac = np.array(resonant_values_vac)
    for i,j in enumerate(single_phasings):
        ax[1].plot(plot_angles[i]*np.ones(len(resonant_values_vac[i,:])), resonant_values_vac[i,:], 'ys')

ax[0].set_xlabel('Phasing (deg)')
ax[0].set_ylabel('Kink amplitude')

ax[0].set_xlim(phasing_range)
ax[0].set_ylim(ylim)
leg = ax[0].legend(loc='best')
leg.get_frame().set_alpha(0.5)
minor_ticks = range(-90,91,15)
major_ticks = range(-90,91,45)
for i in major_ticks: minor_ticks.remove(i)

ax[0].xaxis.set_ticks(major_ticks,minor=False)
ax[0].xaxis.set_ticks(minor_ticks, minor=True)
ax[0].grid(b=True, which='major', linestyle='-',axis='x')
ax[0].grid(b=True, which='major', linestyle=':',axis='y')
ax[0].grid(b=True, which='minor', linestyle=':')
fig.canvas.draw(); fig.show()




if make_animations:
    #Total phasing animation
    phasings = np.linspace(0,360,15)
    for phasing in phasings:
        fig_anim, ax_anim = pt.subplots()
        phasing = phasing/180.*np.pi
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
        BnPEST_new = upper_data_tot.BnPEST  + lower_data_tot.BnPEST*phasor
        color_ax = ax_anim.pcolor(upper_data_tot.mk.flatten(),upper_data_tot.ss.flatten(),np.abs(BnPEST_new),cmap='hot')
        pt.colorbar(color_ax,ax=ax_anim)
        ax_anim.plot(mk_upper, mk_upper * 0 + ss_upper,'bo')
        color_ax.set_clim([0,2])
        ax_anim.plot(upper_data_tot.mq,upper_data_tot.sq,'bo')
        ax_anim.plot(upper_data_tot.q_profile*n,upper_data_tot.q_profile_s,'b--') 
        ax_anim.set_xlim([-29,29])
        ax_anim.set_ylim([0,1])
        ax_anim.set_title('phasing : %d deg'%(phasing/np.pi*180.))
        fig_anim.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/tot_tmp_%d.png'%(phasing/np.pi*180.))
        pt.close()

    #Vacuum phasing animation
    phasings = np.linspace(0,360,15)
    for phasing in phasings:
        fig_anim, ax_anim = pt.subplots()
        phasing = phasing/180.*np.pi
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
        BnPEST_new = upper_data_vac.BnPEST  + lower_data_vac.BnPEST*phasor
        color_ax = ax_anim.pcolor(upper_data_vac.mk.flatten(),upper_data_vac.ss.flatten(),np.abs(BnPEST_new),cmap='hot')
        pt.colorbar(color_ax,ax=ax_anim)
        ax_anim.plot(mk_upper, mk_upper * 0 + ss_upper,'bo')
        color_ax.set_clim([0,2])
        ax_anim.plot(upper_data_vac.mq,upper_data_vac.sq,'bo')
        ax_anim.plot(upper_data_vac.q_profile*n,upper_data_vac.q_profile_s,'b--') 
        ax_anim.set_xlim([-29,29])
        ax_anim.set_ylim([0,1])
        ax_anim.set_title('phasing : %d deg'%(phasing/np.pi*180.))
        fig_anim.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/vac_tmp_%d.png'%(phasing/np.pi*180.))
        pt.close()

    #Plasma phasing animation
    for phasing in phasings:
        fig_anim, ax_anim = pt.subplots()
        phasing = phasing/180.*np.pi
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
        BnPEST_new = (upper_data_tot.BnPEST - upper_data_vac.BnPEST)  + (lower_data_tot.BnPEST-lower_data_vac.BnPEST)*phasor
        color_ax = ax_anim.pcolor(upper_data_vac.mk.flatten(),upper_data_vac.ss.flatten(),np.abs(BnPEST_new),cmap='hot')
        pt.colorbar(color_ax,ax=ax_anim)
        ax_anim.plot(mk_upper, mk_upper * 0 + ss_upper,'bo')
        color_ax.set_clim([0,2])
        ax_anim.plot(upper_data_vac.mq,upper_data_vac.sq,'bo')
        ax_anim.plot(upper_data_vac.q_profile*n,upper_data_vac.q_profile_s,'b--') 
        ax_anim.set_xlim([-29,29])
        ax_anim.set_ylim([0,1])
        ax_anim.set_title('phasing : %d deg'%(phasing/np.pi*180.))
        fig_anim.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/plas_tmp_%d.png'%(phasing/np.pi*180.))
        pt.close()


'''
