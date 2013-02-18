'''
Generates plots of 'kink amplification' as a function of phasing
Will also create the files for an animation of plasma, vac, and total 
components in PEST co-ordinates

'''

import results_class
#from RZfuncs import I0EXP_calc
import RZfuncs
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
file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacingV2/equal_spacingV2_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4/equal_spacing_n4_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4_V2/equal_spacing_n4_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing_n4_V2/equal_spacing_n4_post_processing_PEST.pickle'
N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
#I0EXP = I0EXP_calc(N,n,I)
#I0EXP = 1.0e+3*0.528 #PMZ n4 real
I0EXP = RZfuncs.I0EXP_calc_real(n,I)

Bn_Li_value = 1.5 
facn = 1.0; psi = 0.97
q_range = [2,6]; ylim = [0,1.4]
#phasing_range = [-180.,180.]
#phasing_range = [0.,360.]
phasing_range = [-90.,90.]
phase_machine_ntor = 1
make_animations = 0
include_discrete_comparison = 0
seperate_res_plot = 0
include_vert_lines = 0
beta_n_axis = 'beta_n'#'beta_n/li'
beta_n_axis = 'beta_n/li'
plot_type = 'best_harmonic'
#plot_type = 'normalised'
#plot_type = 'normalised_average'
#plot_type = 'standard_average'



def no_wall_limit(q95_list, beta_n_list):
    '''
    Returns the maximum item in beta_n_list for each unique item in q95_list
    Useful for returning the no wall limit
    SH 26/12/2012
    '''
    q95 = np.array(q95_list)
    bn = np.array(beta_n_list)
    q95_values = set(q95_list)
    xaxis = []; yaxis = []; yaxis2 = []
    for i in q95_values:
        xaxis.append(i)
        yaxis.append(np.max(bn[q95==i]))
        increment = yaxis[-1] - np.max(bn[(q95==i) & (bn!=yaxis[-1])])
        yaxis2.append(yaxis[-1]+increment)
    tmp1 = sorted(zip(xaxis,yaxis,yaxis2))
    xaxis = [tmp for (tmp,tmp2,tmp3) in tmp1]
    yaxis = [tmp2 for (tmp,tmp2,tmp3) in tmp1]
    yaxis2 = [tmp3 for (tmp,tmp2,tmp3) in tmp1]

    return xaxis, yaxis, yaxis2

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
divisor_list = []
for i in project_dict['sims'].keys():
    q95_list.append(project_dict['sims'][i]['Q95'])
    serial_list.append(i)
    if beta_n_axis=='beta_n':
        Bn_Li_list.append(project_dict['sims'][i]['BETAN'])
    elif beta_n_axis=='beta_n/li':
        Bn_Li_list.append(project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI'])

    relevant_values_upper_tot = project_dict['sims'][i]['responses'][str(psi)]['total_kink_response_upper']
    relevant_values_lower_tot = project_dict['sims'][i]['responses'][str(psi)]['total_kink_response_lower']
    relevant_values_upper_vac = project_dict['sims'][i]['responses'][str(psi)]['vacuum_kink_response_upper']
    relevant_values_lower_vac = project_dict['sims'][i]['responses'][str(psi)]['vacuum_kink_response_lower']
    mk_list.append(project_dict['sims'][i]['responses'][str(psi)]['mk'])

    upper_tot_res = np.array(project_dict['sims'][i]['responses']['total_resonant_response_upper'])
    lower_tot_res = np.array(project_dict['sims'][i]['responses']['total_resonant_response_lower'])
    upper_vac_res = np.array(project_dict['sims'][i]['responses']['vacuum_resonant_response_upper'])
    lower_vac_res = np.array(project_dict['sims'][i]['responses']['vacuum_resonant_response_lower'])
    divisor_list.append(len(np.array(project_dict['sims'][i]['responses']['total_resonant_response_upper'])))
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


def calculate_db_kink_fixed(mk_list, q_val_list, n, to_be_calculated,n_plus):
    '''
    Calculate db_kink based on a fixed harmonic
    '''
    answer = []
    for i in range(0,len(to_be_calculated)):
        fixed_loc = np.min([np.argmin(np.abs(mk_list[i] - q_val_list[i]*n)) + n_plus, len(to_be_calculated[i])-1])
        answer.append(to_be_calculated[i][fixed_loc])
    return answer

def calculate_db_kink(mk_list, q_val_list, n, reference, to_be_calculated):
    '''
    Calculate db_kink based on the maximum value
    '''
    answer = []; mode_list = []; max_loc_list = []
    #answer_phase = []
    for i in range(0,len(reference)):
        allowable_indices = np.array(mk_list[i])>(np.array(q_val_list[i])*(n+0))
        maximum_val = np.max(np.abs(reference[i])[allowable_indices])
        max_loc = np.argmin(np.abs(np.abs(reference[i]) - maximum_val))
        max_loc_list.append(max_loc)
        mode_list.append(mk_list[i][max_loc])
        answer.append(to_be_calculated[i][max_loc])
        #answer_phase.append(np.angle(to_be_calculated[i][max_loc], deg = True))
    return answer, mode_list, max_loc_list

def extract_dB_kink(tmp_dict, psi):
    '''
    extract dB_kink information from a standard pyMARS output dictionary
    '''
    amps_vac_comp_upper = []; amps_vac_comp_lower = []
    amps_plas_comp_upper = []; amps_plas_comp_lower = []
    amps_tot_comp_upper = []; amps_tot_comp_lower = []
    mk_list = [];  q_val_list = []; resonant_close = []

    for i in tmp_dict['sims'].keys():
        relevant_values_upper_tot = tmp_dict['sims'][i]['responses'][str(psi)]['total_kink_response_upper']
        relevant_values_lower_tot = tmp_dict['sims'][i]['responses'][str(psi)]['total_kink_response_lower']
        relevant_values_upper_vac = tmp_dict['sims'][i]['responses'][str(psi)]['vacuum_kink_response_upper']
        relevant_values_lower_vac = tmp_dict['sims'][i]['responses'][str(psi)]['vacuum_kink_response_lower']

        mk_list.append(tmp_dict['sims'][i]['responses'][str(psi)]['mk'])
        q_val_list.append(tmp_dict['sims'][i]['responses'][str(psi)]['q_val'])
        resonant_close.append(np.min(np.abs(tmp_dict['sims'][i]['responses']['resonant_response_sq']-psi)))

        amps_plas_comp_upper.append(relevant_values_upper_tot-relevant_values_upper_vac)
        amps_plas_comp_lower.append(relevant_values_lower_tot-relevant_values_lower_vac)
        amps_vac_comp_upper.append(relevant_values_upper_vac)
        amps_vac_comp_lower.append(relevant_values_lower_vac)
        amps_tot_comp_upper.append(relevant_values_upper_tot)
        amps_tot_comp_lower.append(relevant_values_lower_tot)

    return amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close



x_axis_NW, y_axis_NW, y_axis_NW2 = no_wall_limit(q95_list, Bn_Li_list)

plot_quantity_vac=[];plot_quantity_plas=[];plot_quantity_tot=[];
plot_quantity_vac_phase=[];plot_quantity_plas_phase=[];plot_quantity_tot_phase=[];

plot_quantity = 'max'
max_loc_list = []; mode_list = []
upper_values_plasma = []; lower_values_plasma = []
upper_values_vac = []; lower_values_vac = []
upper_values_vac_fixed = []; lower_values_vac_fixed = []
nq_plus_one = 1


print 'starting new_section'
new_way = 1
if new_way:
    amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close = extract_dB_kink(project_dict, psi)

    reference = amps_tot_comp

    plot_quantity_vac, mode_list, max_loc_list = calculate_db_kink(mk_list, q_val_list, n, reference, amps_vac_comp)
    plot_quantity_plas, mode_list, max_loc_list = calculate_db_kink(mk_list, q_val_list, n, reference, amps_plas_comp)
    plot_quantity_tot, mode_list, max_loc_list = calculate_db_kink(mk_list, q_val_list, n, reference, amps_tot_comp)
    plot_quantity_vac_phase = np.angle(plot_quantity_vac,deg=True).tolist()
    plot_quantity_plas_phase = np.angle(plot_quantity_plas,deg=True).tolist()
    plot_quantity_tot_phase = np.angle(plot_quantity_tot,deg=True).tolist()
    plot_quantity_vac = np.abs(plot_quantity_vac).tolist()
    plot_quantity_plas = np.abs(plot_quantity_plas).tolist()
    plot_quantity_tot = np.abs(plot_quantity_tot).tolist()
    upper_values_plasma, mode_list, max_loc_list = calculate_db_kink(mk_list, q_val_list, n, reference, amps_plas_comp_upper)
    lower_values_plasma, mode_list, max_loc_list = calculate_db_kink(mk_list, q_val_list, n, reference, amps_plas_comp_lower)
    upper_values_vac, mode_list, max_loc_list = calculate_db_kink(mk_list, q_val_list, n, reference, amps_vac_comp_upper)
    lower_values_vac, mode_list, max_loc_list = calculate_db_kink(mk_list, q_val_list, n, reference, amps_vac_comp_lower)
    upper_values_vac_fixed = calculate_db_kink_fixed(mk_list, q_val_list, n, amps_vac_comp_upper, 5)
    lower_values_vac_fixed = calculate_db_kink_fixed(mk_list, q_val_list, n, amps_vac_comp_lower, 5)
    print 'finish new section'
else:
    for i in range(0,len(amps_vac_comp)):
        if plot_quantity == 'average':
            plot_quantity_vac.append(np.sum(np.abs(amps_vac_comp[i])**2)/len(amps_vac_comp[i]))
            plot_quantity_plas.append(np.sum(np.abs(amps_plas_comp[i])**2)/len(amps_vac_comp[i]))
            plot_quantity_tot.append(np.sum(np.abs(amps_tot_comp[i])**2)/len(amps_vac_comp[i]))
        elif plot_quantity == 'max':
            #argmin(mk_list[i][max_loc])

            #mode_list.append(mk_list[i][max_loc])

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


            upper_values_vac_fixed.append(amps_vac_comp_upper[i][4])
            lower_values_vac_fixed.append(amps_vac_comp_lower[i][4])
        


xnew = np.linspace(2.,7.,200)
ynew = np.linspace(0.75,4.5,200)
xnew_grid, ynew_grid = np.meshgrid(xnew,ynew)
q95_Bn_array = np.zeros((len(q95_list),2),dtype=float)
q95_Bn_array[:,0] = q95_list[:]
q95_Bn_array[:,1] = Bn_Li_list[:]

q95_Bn_new = np.zeros((len(xnew),2),dtype=float)
q95_Bn_new[:,0] = xnew[:]
q95_Bn_new[:,1] = ynew[:]

y_axis_NW_interp = np.interp(xnew,x_axis_NW, y_axis_NW)
y_axis_NW_interp2 = np.interp(xnew,x_axis_NW, y_axis_NW2)
plas_data = griddata(q95_Bn_array, np.array(plot_quantity_plas), (xnew_grid, ynew_grid),method = 'linear')
vac_data = griddata(q95_Bn_array, np.array(plot_quantity_vac), (xnew_grid, ynew_grid), method = 'linear')
tot_data = griddata(q95_Bn_array, np.array(plot_quantity_tot), (xnew_grid, ynew_grid), method = 'linear')

plas_data_res = griddata(q95_Bn_array, np.array(res_plas), (xnew_grid, ynew_grid),method = 'linear')
vac_data_res = griddata(q95_Bn_array, np.array(res_vac), (xnew_grid, ynew_grid), method = 'linear')
tot_data_res = griddata(q95_Bn_array, np.array(res_tot), (xnew_grid, ynew_grid), method = 'linear')
vac_data_res_ave = griddata(q95_Bn_array, np.array(res_vac)/np.array(divisor_list), (xnew_grid, ynew_grid), method = 'linear')

mode_data = griddata(q95_Bn_array, mode_list, (xnew_grid, ynew_grid), method = 'cubic')
plas_data_phase = griddata(q95_Bn_array, plot_quantity_plas_phase, (xnew_grid, ynew_grid), method = 'linear')
vac_data_phase = griddata(q95_Bn_array, plot_quantity_vac_phase, (xnew_grid, ynew_grid), method = 'linear')

mask = np.isnan(plas_data)
for i in range(0,plas_data.shape[1]):
    mask[ynew>((y_axis_NW_interp[i]+y_axis_NW_interp2[i])/2),i]=True

#for i in range(0,len(xnew)):
#    plas_data_res[ynew>y_axis_NW_interp[i],i] = np.nan
#    plas_data[ynew>y_axis_NW_interp[i],i] = np.nan


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

# fig,ax = pt.subplots()
# color_fig = ax.pcolor(xnew, ynew, tot_data)
# if plot_quantity=='average':
#     color_fig.set_clim([0,7])
# elif plot_quantity=='max':
#     color_fig.set_clim([0,7])


# ax.set_title('total data')
# fig.suptitle(file_name,fontsize=8)
# fig.canvas.draw(); fig.show()

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
if beta_n_axis=='beta_n':
    ax.set_ylabel(r'$\beta_N$', fontsize = 14)
elif beta_n_axis=='beta_n/li':
    ax.set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
ax.set_xlabel(r'$q_{95}$', fontsize = 14)
fig.suptitle(file_name,fontsize=8)
fig.canvas.draw(); fig.show()

color_map = 'jet'


fig_JAW, ax_JAW = pt.subplots()
#color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(plas_data, mask=(np.isnan(plas_data) & (plas_data<y_axis_NW_interp)),cmap=color_map, rasterized=True)
#color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(plas_data, mask=(plas_data>np.tile(y_axis_NW_interp,(200,1)))),cmap=color_map, rasterized=True)
color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(plas_data, mask=mask), cmap=color_map, rasterized=True)
#color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, plas_data,cmap=color_map, rasterized=True)
cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW)
cbar.ax.set_ylabel('G/kA')
ax_JAW.set_xlabel(r'$q_{95}$', fontsize = 14)
if beta_n_axis == 'beta_n/li':
    ax_JAW.set_ylabel(r'$\beta_N/\ell_i$', fontsize = 14)
else:
    ax_JAW.set_ylabel(r'$\beta_N$', fontsize = 14)
ax_JAW.plot(q95_list, Bn_Li_list,'k.')

ax_JAW.plot(x_axis_NW, y_axis_NW,'k-')
ax_JAW.plot(xnew, y_axis_NW_interp,'k-')
ax_JAW.plot(x_axis_NW, y_axis_NW2,'k-')
ax_JAW.fill_between(x_axis_NW,  y_axis_NW, y_axis_NW2, facecolor='black', alpha=1)
ax_JAW.set_title(r'Plasma, $\psi_N=%.2f$'%(psi**2))
ax_JAW.set_xlim([2.5,6])
ax_JAW.set_ylim([0.75,4.5])
color_fig_plas_JAW.set_clim([0,3.0])
fig_JAW.canvas.draw(); fig_JAW.show()

#Vacuum single plot db_res
fig_JAW, ax_JAW = pt.subplots()
#color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(vac_data_res_ave, mask=np.isnan(plas_data)),cmap=color_map, rasterized=True)
color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(vac_data_res_ave, mask=mask),cmap=color_map, rasterized=True)
#color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(plas_data, mask=mask), cmap=color_map, rasterized=True)
cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW)
cbar.ax.set_ylabel(r'$\bar{\delta B}_{res}^{n=2}$ G/kA',fontsize=20)
ax_JAW.set_xlabel(r'$q_{95}$', fontsize = 20)
if beta_n_axis == 'beta_n/li':
    ax_JAW.set_ylabel(r'$\beta_N/\ell_i$', fontsize = 20)
else:
    ax_JAW.set_ylabel(r'$\beta_N$', fontsize = 20)
ax_JAW.plot(q95_list, Bn_Li_list,'k.')
ax_JAW.fill_between(xnew,  y_axis_NW_interp, y_axis_NW_interp2, facecolor='black', alpha=1)
#ax_JAW.set_title(r'Vacuum, $\psi_N=%.2f$'%(psi**2), fontsize = 18)
ax_JAW.set_xlim([2.5, 6.8])
ax_JAW.set_ylim([0.75,4.5])
#color_fig_plas_JAW.set_clim([0,9.0])
fig_JAW.canvas.draw(); fig_JAW.show()


fig_JAW, ax_JAW = pt.subplots(nrows=2, sharex =1, sharey=1)
#color_fig_plas_JAW = ax_JAW[0].pcolor(xnew, ynew, np.ma.array(vac_data_res_ave, mask=np.isnan(plas_data)),cmap=color_map, rasterized=True)
color_fig_plas_JAW = ax_JAW[0].pcolor(xnew, ynew, np.ma.array(vac_data_res_ave, mask=mask),cmap=color_map, rasterized=True)
cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW[0])
cbar.ax.set_ylabel(r'$\overline{\delta B}_{res}^{n=2}$ G/kA',fontsize=20)
#color_fig_plas_JAW = ax_JAW[1].pcolor(xnew, ynew, np.ma.array(vac_data_res, mask=np.isnan(plas_data)),cmap=color_map, rasterized=True)
color_fig_plas_JAW = ax_JAW[1].pcolor(xnew, ynew, np.ma.array(vac_data_res, mask=mask),cmap=color_map, rasterized=True)
color_fig_plas_JAW.set_clim([0,8])
cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW[1])
cbar.ax.set_ylabel(r'$\delta B_{res}^{n=2}$ G/kA',fontsize=20)

ax_JAW[1].set_xlabel(r'$q_{95}$', fontsize = 20)
if beta_n_axis=='beta_n':
    ax_JAW[0].set_ylabel(r'$\beta_N$', fontsize = 20)
    ax_JAW[1].set_ylabel(r'$\beta_N$', fontsize = 20)
elif beta_n_axis=='beta_n/li':
    ax_JAW[0].set_ylabel(r'$\beta_N / L_i$', fontsize = 20)
    ax_JAW[1].set_ylabel(r'$\beta_N / L_i$', fontsize = 20)
ax_JAW[0].plot(q95_list, Bn_Li_list,'k.')
ax_JAW[1].plot(q95_list, Bn_Li_list,'k.')
ax_JAW[0].fill_between(xnew,  y_axis_NW_interp, y_axis_NW_interp2, facecolor='black', alpha=1)
ax_JAW[1].fill_between(xnew,  y_axis_NW_interp, y_axis_NW_interp2, facecolor='black', alpha=1)
#ax_JAW[0].plot(x_axis_NW, y_axis_NW,'k-')
#ax_JAW[1].plot(x_axis_NW, y_axis_NW,'k-')
#ax_JAW[0].plot(x_axis_NW, np.array(y_axis_NW)*0.75,'b-')
#ax_JAW[1].plot(x_axis_NW, np.array(y_axis_NW)*0.75,'b-')
#ax_JAW.set_title(r'Plasma, $\psi_N=%.2f$'%(psi**2), fontsize = 18)
ax_JAW[0].set_xlim([2.5, 6.8])
ax_JAW[0].set_ylim([0.75,4.5])
#color_fig_plas_JAW.set_clim([0,9.0])
fig_JAW.canvas.draw(); fig_JAW.show()


fig_JAW, ax_JAW = pt.subplots()
#color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(plas_data, mask=np.isnan(plas_data)),cmap=color_map, rasterized=True)
color_fig_plas_JAW = ax_JAW.pcolor(xnew, ynew, np.ma.array(plas_data, mask=mask),cmap=color_map, rasterized=True)
#ax_JAW.contour(xnew, ynew, np.ma.array(plas_data, mask=np.isnan(plas_data)))
cbar = pt.colorbar(color_fig_plas_JAW, ax = ax_JAW)
cbar.ax.set_ylabel(r'$\delta B_{kink}^{n=2}$ G/kA',fontsize=20)
ax_JAW.set_xlabel(r'$q_{95}$', fontsize = 20)

if beta_n_axis == 'beta_n/li':
    ax_JAW.set_ylabel(r'$\beta_N/\ell_i$', fontsize = 20)
else:
    ax_JAW.set_ylabel(r'$\beta_N$', fontsize = 20)
ax_JAW.plot(q95_list, Bn_Li_list,'k.')
ax_JAW.fill_between(xnew,  y_axis_NW_interp, y_axis_NW_interp2, facecolor='black', alpha=1)
#ax_JAW.set_title(r'Plasma, $\psi_N=%.2f$'%(psi**2), fontsize = 18)
ax_JAW.set_xlim([2.5, 6.8])
ax_JAW.set_ylim([0.75,4.5])
color_fig_plas_JAW.set_clim([0,3.0])
fig_JAW.canvas.draw(); fig_JAW.show()



fig,ax = pt.subplots(nrows = 3,sharex = 1, sharey = 1)
color_fig_plas = ax[0].pcolor(xnew, ynew, np.ma.array(plas_data, mask=np.isnan(plas_data)),cmap=color_map)#, cmap = cmap)



#color_fig_plas = ax[0].pcolor(xnew, ynew, np.ma.array(plas_data, mask=np.isnan(plas_data)),cmap=color_map)#, cmap = cmap)

#color_fig = ax[0].pcolor(xnew, ynew, plas_data)
cbar = pt.colorbar(color_fig_plas, ax = ax[0])
cbar.ax.set_ylabel('G/kA')
if plot_quantity=='average':
    color_fig_plas.set_clim([0,7])
elif plot_quantity=='max':
    #pass
    color_fig_plas.set_clim([0,3.0])
if beta_n_axis=='beta_n':
    ax[0].set_ylabel(r'$\beta_N$', fontsize = 14)
elif beta_n_axis=='beta_n/li':
    ax[0].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)

#ax[0].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
ax[0].set_title(r'Plasma, $\psi_N=%.2f$'%(psi**2))
ax[0].plot(q95_list, Bn_Li_list,'k.')
#fig.canvas.draw(); fig.show()

#fig,ax = pt.subplots()
#import matplotlib.cm as cm
#cmap = cm.jet
#cmap.set_bad('w',1.)

color_fig_vac = ax[1].pcolor(xnew, ynew, np.ma.array(vac_data, mask=np.isnan(vac_data)),cmap=color_map)#, cmap = cmap)
color_fig_relative = ax[2].pcolor(xnew, ynew, np.ma.array(plas_data/vac_data, mask=np.isnan(vac_data)),cmap=color_map)#, cmap = cmap)
color_fig_relative.set_clim([0,10])
cbar = pt.colorbar(color_fig_relative, ax = ax[2])
#cbar.ax.set_ylabel('G/kA')
if plot_quantity=='average':
    color_fig_vac.set_clim([0,0.2])
elif plot_quantity=='max':
    color_fig_vac.set_clim([0,0.9])

cbar = pt.colorbar(color_fig_vac, ax = ax[1])
cbar.ax.set_ylabel('G/kA')
ax[1].set_title(r'Vacuum, $\psi_N=%.2f$'%(psi**2))


if beta_n_axis=='beta_n':
    ax[1].set_ylabel(r'$\beta_N$', fontsize = 14)
    ax[2].set_ylabel(r'$\beta_N$', fontsize = 14)
elif beta_n_axis=='beta_n/li':
    ax[1].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
    ax[2].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
#ax[1].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
#ax[2].set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
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


phasing_array = np.linspace(0,360,360)
fig, ax = pt.subplots(nrows =2 , sharex = 1, sharey = 1)
pmult_values = [1.0]
pmult_array = np.array(pmult_list)
q95_array = np.array(q95_list)
rel_q95_vals = q95_array[pmult_array==pmult_values[0]]


rel_lower_vals_plasma = np.array(lower_values_plasma)[pmult_array==pmult_values[0]]
rel_upper_vals_plasma = np.array(upper_values_plasma)[pmult_array==pmult_values[0]]
rel_lower_vals_vac = np.array(lower_values_vac)[pmult_array==pmult_values[0]]
rel_upper_vals_vac = np.array(upper_values_vac)[pmult_array==pmult_values[0]]

rel_lower_vals_vac_fixed = np.array(lower_values_vac_fixed)[pmult_array==pmult_values[0]]
rel_upper_vals_vac_fixed = np.array(upper_values_vac_fixed)[pmult_array==pmult_values[0]]

q95_single = np.linspace(2.6,6,100)

rel_lower_vals_plasma = griddata(q95_Bn_array, np.array(lower_values_plasma), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
rel_upper_vals_plasma = griddata(q95_Bn_array, np.array(upper_values_plasma), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
rel_lower_vals_vac = griddata(q95_Bn_array, np.array(lower_values_vac), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
rel_upper_vals_vac = griddata(q95_Bn_array, np.array(upper_values_vac), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')

rel_lower_vals_vac_fixed = griddata(q95_Bn_array, np.array(lower_values_vac_fixed), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
rel_upper_vals_vac_fixed = griddata(q95_Bn_array, np.array(upper_values_vac_fixed), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')



plot_array_plasma = np.ones((phasing_array.shape[0], rel_q95_vals.shape[0]),dtype=float)
plot_array_plasma = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)


plot_array_vac = np.ones((phasing_array.shape[0], rel_q95_vals.shape[0]),dtype=float)
plot_array_vac = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
plot_array_vac_fixed = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)


plot_array_vac_res = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
plot_array_plas_res = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
plot_array_vac_res2 = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)
plot_array_plas_res2 = np.ones((phasing_array.shape[0], q95_single.shape[0]),dtype=float)

for i, curr_phase in enumerate(phasing_array):
    phasor = (np.cos(curr_phase/180.*np.pi)+1j*np.sin(curr_phase/180.*np.pi))
    plot_array_plasma[i,:] = np.abs(rel_upper_vals_plasma + rel_lower_vals_plasma*phasor)
    plot_array_vac[i,:] = np.abs(rel_upper_vals_vac + rel_lower_vals_vac*phasor)
    plot_array_vac_fixed[i,:] = np.abs(rel_upper_vals_vac_fixed + rel_lower_vals_vac_fixed*phasor)



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
ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
ax[0].set_title('Kink Amplitude - Plasma')
ax[1].set_title('Kink Amplitude - Vacuum')
ax[0].set_xlim([np.min(q95_single),np.max(q95_single)])
ax[0].set_ylim([np.min(phasing_array),np.max(phasing_array)])
color_plot.set_clim([0, 2])
color_plot2.set_clim([0, 1])
cb = pt.colorbar(color_plot, ax = ax[0])
cb.ax.set_ylabel(r'$\delta B_{kink}^{n=2}$ G/kA',fontsize=20)
cb = pt.colorbar(color_plot2, ax = ax[1])
cb.ax.set_ylabel(r'$\delta B_{kink}^{n=2}$ G/kA',fontsize=20)
fig.canvas.draw(); fig.show()


fig, ax = pt.subplots(); ax=[ax]
color_plot = ax[0].pcolor(q95_single, phasing_array, plot_array_plasma, cmap='hot', rasterized=True)
ax[0].set_xlabel(r'$q_{95}$', fontsize=14)
ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
ax[0].set_title('Kink Amplitude - Plasma')
ax[0].set_xlim([np.min(q95_single),np.max(q95_single)])
ax[0].set_ylim([np.min(phasing_array),np.max(phasing_array)])
color_plot.set_clim([0, 2])
ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
tmp_xaxis = np.arange(1,10,0.1)
tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
ax[0].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
ax[0].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')
ax[0].set_xlim([2.6, 6])
cb = pt.colorbar(color_plot, ax = ax[0])
cb.ax.set_ylabel(r'$\delta B_{kink}^{n=2}$ G/kA',fontsize=20)
fig.canvas.draw(); fig.show()


fig, ax = pt.subplots(); ax=[ax]
color_plot = ax[0].pcolor(q95_single, phasing_array, plot_array_vac_fixed, cmap='hot', rasterized=True)
ax[0].set_xlabel(r'$q_{95}$', fontsize=14)
ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
ax[0].set_title('Kink Amplitude - Vacuum')
ax[0].set_xlim([np.min(q95_single),np.max(q95_single)])
ax[0].set_ylim([np.min(phasing_array),np.max(phasing_array)])
#color_plot.set_clim([0, 2])
ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
tmp_xaxis = np.arange(1,10,0.1)
tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
ax[0].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
ax[0].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')
ax[0].set_xlim([2.6, 6])
cb = pt.colorbar(color_plot, ax = ax[0])
cb.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+4,n=2}$ G/kA',fontsize=20)
fig.canvas.draw(); fig.show()



fig, ax = pt.subplots(nrows=2,sharex = 1, sharey=1)
color_plot = ax[0].pcolor(q95_single, phasing_array, plot_array_plasma, cmap='hot', rasterized=True)
cb = pt.colorbar(color_plot, ax = ax[0])
ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
color_plot.set_clim([0, 2])
ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
tmp_xaxis = np.arange(1,10,0.1)
tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
ax[0].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
ax[0].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')
cb.ax.set_ylabel(r'$\delta B_{kink}^{n=2}$ G/kA',fontsize=20)
color_plot = ax[1].pcolor(q95_single, phasing_array, plot_array_vac_fixed, cmap='hot', rasterized=True)
cb = pt.colorbar(color_plot, ax = ax[1])
ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 14)
#ax[1].set_title('Kink Amplitude - Vacuum')
ax[1].set_ylim([np.min(phasing_array),np.max(phasing_array)])
#color_plot.set_clim([0, 2])
ax[1].plot(np.arange(1,10), np.arange(1,10)*(-35.)+130+180,'b-')
tmp_xaxis = np.arange(1,10,0.1)
tmp_yaxis = np.arange(1,10,0.1)*(-35.)+130
ax[1].plot(tmp_xaxis[tmp_yaxis>0], tmp_yaxis[tmp_yaxis>0],'b-')
ax[1].plot(tmp_xaxis[tmp_yaxis<0], tmp_yaxis[tmp_yaxis<0]+360,'b-')
ax[1].set_xlim([2.6, 6])
cb.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+4,n=2}$ G/kA',fontsize=20)
fig.canvas.draw(); fig.show()



#res_vac_array_upper = np.array(res_vac_list_upper)
#res_vac_array_lower = np.array(res_vac_list_lower)
#res_plas_array_upper = np.array(res_plas_list_upper)
#res_plas_array_lower = np.array(res_plas_list_lower)

for i, curr_phase in enumerate(phasing_array):
    print 'phase :', curr_phase
    phasor = (np.cos(curr_phase/180.*np.pi)+1j*np.sin(curr_phase/180.*np.pi))
    tmp_vac_list = []; tmp_plas_list = []
    tmp_vac_list2 = []; tmp_plas_list2 = []
    for ii in range(0,len(res_vac_list_upper)):
        divisor = len(res_vac_list_upper[ii])
        tmp_vac_list2.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor))/divisor)
        tmp_plas_list2.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor))/divisor)
        tmp_vac_list.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor)))
        tmp_plas_list.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor)))

    plot_array_vac_res[i,:] = griddata(q95_Bn_array, np.array(tmp_vac_list), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
    plot_array_plas_res[i,:] = griddata(q95_Bn_array, np.array(tmp_plas_list), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
    plot_array_vac_res2[i,:] = griddata(q95_Bn_array, np.array(tmp_vac_list2), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')
    plot_array_plas_res2[i,:] = griddata(q95_Bn_array, np.array(tmp_plas_list2), (q95_single, q95_single*0.+Bn_Li_value),method = 'cubic')


max_phases = phasing_array[np.argmax(plot_array_vac_res,axis=0)]
max_phases[max_phases>max_phases[0]]-=360
poly_max_res = np.polyfit(q95_single,max_phases,1)
best_fit_max_res = np.polyval(poly_max_res, q95_single)
best_fit_max_res[best_fit_max_res<0]+=360
best_fit_max_res[best_fit_max_res>360]-=360

min_phases = phasing_array[np.argmin(plot_array_vac_res,axis=0)]
min_phases[min_phases>min_phases[0]]-=360
poly_min_res = np.polyfit(q95_single,min_phases,1)
best_fit_min_res = np.polyval(poly_min_res, q95_single)
best_fit_min_res[best_fit_min_res<0]+=360
best_fit_min_res[best_fit_min_res>360]-=360

print '############ best fit min res ################'
print poly_min_res
print '############ best fit max res  ################'
print poly_max_res
fig, ax = pt.subplots(nrows = 2, sharex = 1, sharey = 1); #ax = [ax]#nrows = 2, sharex = 1, sharey = 1)
color_plot = ax[0].pcolor(q95_single, phasing_array, plot_array_vac_res, cmap='hot', rasterized=True)
ax[0].contour(q95_single,phasing_array, plot_array_vac_res, colors='white')
color_plot2 = ax[1].pcolor(q95_single, phasing_array, plot_array_vac_res2, cmap='hot', rasterized=True)
ax[1].contour(q95_single,phasing_array, plot_array_vac_res2, colors='white')
color_plot.set_clim([0,10])
color_plot2.set_clim([0,0.75])
#ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+250,'b-')
#ax[0].plot(np.arange(1,10), np.arange(1,10)*(-35.)+250+180,'b-')
#ax[1].plot(np.arange(1,10), np.arange(1,10)*(-35.)+250,'b-')
#ax[1].plot(np.arange(1,10), np.arange(1,10)*(-35.)+250+180,'b-')
#ax[0].plot(q95_single, best_fit_max_res, 'b.')
#ax[0].plot(q95_single, best_fit_min_res, 'b.')
#ax[1].plot(q95_single, best_fit_max_res, 'b.')
#ax[1].plot(q95_single, best_fit_min_res, 'b.')
ax[0].set_xlim([2.6, 6])
ax[0].set_ylim([np.min(phasing_array), np.max(phasing_array)])
ax[1].set_xlabel(r'$q_{95}$', fontsize=20)
ax[0].set_title('n=%d, Pitch Resonant Forcing'%(n))
ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
cbar = pt.colorbar(color_plot, ax = ax[0])
cbar.ax.set_ylabel(r'$\delta B_{res}^{n=2}$ G/kA',fontsize = 20)
cbar = pt.colorbar(color_plot2, ax = ax[1])
cbar.ax.set_ylabel(r'$\overline{\delta B}_{res}^{n=2}$ G/kA', fontsize = 20)
fig.canvas.draw(); fig.show()


# fig, ax = pt.subplots()
# color_plot = ax.pcolor(q95_single, phasing_array, plot_array_vac_res, cmap='hot', rasterized=True)
# ax.contour(q95_single,phasing_array, plot_array_vac_res, colors='white')
# color_plot.set_clim([0,10])
# ax.set_ylim([np.min(phasing_array), np.max(phasing_array)])
# ax.set_title('n=%d, Pitch Resonant Forcing'%(n))
# ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
# ax.plot(np.arange(1,10), np.arange(1,10)*(-35.)+250,'b-')
# ax.plot(np.arange(1,10), np.arange(1,10)*(-35.)+250+180,'b-')
# cbar = pt.colorbar(color_plot, ax = ax)
# ax.set_xlim([2.6, 6])
# cbar.ax.set_ylabel(r'$\delta B_{res}$ G/kA',fontsize = 20)
# ax.set_xlabel(r'$q_{95}$', fontsize=20)
# fig.canvas.draw(); fig.show()
            

# ax[1].grid(b=True)
fig, ax = pt.subplots()
ax.plot(q95_list, Bn_Li_list,'k.')
if beta_n_axis=='beta_n':
    ax.set_ylabel(r'$\beta_N$', fontsize = 14)
    ax.set_ylabel(r'$\beta_N$', fontsize = 14)
elif beta_n_axis=='beta_n/li':
    ax.set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
    ax.set_ylabel(r'$\beta_N / L_i$', fontsize = 14)
ax.set_xlabel(r'$q_{95}$', fontsize=14)
fig.canvas.draw(); fig.show()
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
