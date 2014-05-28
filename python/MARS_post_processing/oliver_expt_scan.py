import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func
import copy

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

file_name = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan/shot_142614_rote_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp/shot_142614_rote_scan_w_damp_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp2/shot_142614_rote_scan_w_damp2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_scan_w_damp3/shot_142614_rote_scan_w_damp3_post_processing_PEST.pickle'
file_name='/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30/shot_142614_rote_res_scan_30x30_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_test/shot_142614_rote_res_scan_test_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_rote_res_scan_30x30_kpar1/shot_142614_rote_res_scan_30x30_kpar1_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/shot_142614_expt_scan_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq/shot_142614_expt_scan_const_eq_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eqV2/shot_142614_expt_scan_const_eqV2_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV1/shot_142614_expt_scan_NC_const_eqV1_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV2/shot_142614_expt_scan_NC_const_eqV2_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV3/shot_142614_expt_scan_NC_const_eqV3_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV4/shot_142614_expt_scan_NC_const_eqV4_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV5/shot_142614_expt_scan_NC_const_eqV5_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV6/shot_142614_expt_scan_NC_const_eqV6_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eqV3/shot_142614_expt_scan_const_eqV3_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq_eta_10-10/shot_142614_expt_scan_const_eq_eta_10-10_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_const_eq_eta_10-5/shot_142614_expt_scan_const_eq_eta_10-5_post_processing_PEST.pickle'


#Ideal cases
phasing = 0
n = 3
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [4,0]
sort_name = 'time_list'

fig, ax = pt.subplots(nrows = 6, ncols = 1, sharex = True)
ax = np.array(ax)[:,np.newaxis]

I_coil_ax, x_point_ax, rote_ax, dBres_plas_ax, dBres_tot_ax, probe_plas_ax, dBkink_plas_ax, q95_ax, eta_ax = [None]*9
#I_coil_ax, rote_ax, x_point_ax, probe_plas_ax, dBres_plas_ax, dBres_tot_ax, dBkink_plas_ax = [None]*len(ax)
#I_coil_ax, x_point_ax, rote_ax, dBres_plas_ax, dBres_tot_ax, probe_plas_ax, dBkink_plas_ax, q95_ax, eta_ax = ax
#I_coil_ax, rote_ax, x_point_ax, probe_plas_ax, dBres_plas_ax, dBres_tot_ax, dBkink_plas_ax,eta_ax = ax

const_rot = True
V_dict_const = {1:'1e-6',2:'5e-7',3:'1e-7',4:'7e-8',5:'3e-8',6:'1e-8'}
V_dict_non_const = {1:'1e-7',2:'7e-8',3:'3e-8',4:'1e-8',5:'5e-7',6:'1e-6'}
V_dict_const = {'1e-6':1,'5e-7':2,'1e-7':3,'7e-8':4,'3e-8':5,'1e-8':6}
V_dict_non_const = {'1e-7':1,'7e-8':2,'3e-8':3,'1e-8':4,'5e-7':5,'1e-6':6}

#for V in [1,2,3,4,5]:
count = 0
V_runs = [1,2,3,4,5,6]
eta_vals = ['1e-6', '5e-7', '1e-7', '7e-8', '3e-8', '1e-8']
eta_vals = ['5e-7', '1e-7', '7e-8']
eta_vals = ['1e-7']
eta_vals = ['1e-8']

ideal = False
spitz = True
#for V in V_runs:
gen_func.setup_publication_image(fig, height_prop = 1.5, single_col = True)
for eta in eta_vals:
    #gen_func.setup_publication_image(fig, height_prop = 1./1.618*, single_col = False)
    for const_rot, index, plot_style in zip([False, True], [0,0],[{'marker':'x'}, {'marker':'o'}]):
        if const_rot:
            V_dict = V_dict_const
        else:
            V_dict = V_dict_non_const
        cur_ax = ax[:,index]
        print cur_ax
        #I_coil_ax, rote_ax, x_point_ax, probe_plas_ax, dBres_tot_ax, dBkink_plas_ax = cur_ax
        I_coil_ax, rote_ax, probe_plas_ax, x_point_ax, dBres_tot_ax, dBkink_plas_ax = cur_ax
        V = V_dict[eta]
        if const_rot:
            if ideal:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_ideal/shot_142614_expt_scan_NC_const_eq_const_rot_prof_ideal_post_processing_PEST.pickle'
            elif spitz:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_spitz/shot_142614_expt_scan_NC_const_eq_const_rot_prof_spitz_post_processing_PEST.pickle'
            else:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V{}/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V{}_post_processing_PEST.pickle'.format(V,V)
        else:
            if ideal:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_ideal/shot_142614_expt_scan_NC_const_eq_ideal_post_processing_PEST.pickle'
            elif spitz:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_spitz/shot_142614_expt_scan_NC_const_eq_spitz_post_processing_PEST.pickle'
            else:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV{}/shot_142614_expt_scan_NC_const_eqV{}_post_processing_PEST.pickle'.format(V,V)
        #file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_V{}/shot_142614_expt_scan_NC_V{}_post_processing_PEST.pickle'.format(V,V)
        print file_name
        reference_dB_kink = 'plasma'
        a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)

        dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'sum')
        dBkink = dBres_dBkink.dBkink_calculations(a)
        probe = dBres_dBkink.magnetic_probe(a,' 66M')
        xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)

        tmp_a = np.array(dBres.single_phasing_individual_harms(phasing,field='plasma'))
        tmp_b = np.array(dBres.single_phasing_individual_harms(phasing,field='total'))
        tmp_c = np.array(dBres.single_phasing_individual_harms(phasing,field='vacuum'))


        if I_coil_ax!=None and count==0:
            icoil = np.loadtxt('iu30_142614.txt')
            I_coil_ax.plot(icoil[:,0], icoil[:,1]/1000, rasterized = True)
            I_coil_ax.set_ylabel('MP (kA)')
            multiplier = np.abs(np.interp(np.sort(a.raw_data['shot_time']), icoil[:,0], icoil[:,1])/1000)
        if x_point_ax!=None:
            xpoint.plot_single_phasing(phasing, 'shot_time', field = 'plasma',  ax = x_point_ax, plot_kwargs = plot_style, multiplier = multiplier)
            x_point_ax.set_ylabel('X-pt Disp (au)')
        if rote_ax!=None and count == 0:
            a.plot_parameters('shot_time', 'vtor0', ax = rote_ax, plot_kwargs = {'marker':'x'})
            rote_ax.set_ylabel('TorRot(rad/s)')
        if dBres_plas_ax!=None:
            dBres.plot_single_phasing(phasing, 'shot_time', field = 'plasma', plot_kwargs = plot_style, amplitude = True, ax = dBres_plas_ax, multiplier = multiplier)
            dBres_plas_ax.set_ylabel(r'$\delta B_{res}^{plas}$ (G)')
        if dBres_tot_ax!=None:
            dBres.plot_single_phasing(phasing, 'shot_time', field = 'total', plot_kwargs = plot_style, amplitude = True, ax = dBres_tot_ax, multiplier = multiplier)
            dBres_tot_ax.set_ylabel(r'$\delta B_{res}^{tot}$ (G)')
        if probe_plas_ax!=None:
            probe.plot_single_phasing(phasing, 'shot_time', field = 'plasma', plot_kwargs = plot_style, amplitude = True, ax = probe_plas_ax, multiplier = multiplier)
            #probe.plot_single_phasing(phasing, 'shot_time', field = 'total', plot_kwargs = plot_style, amplitude = True, ax = probe_plas_ax)
            #probe.plot_single_phasing(phasing, 'shot_time', field = 'vacuum', plot_kwargs = plot_style, amplitude = True, ax = probe_plas_ax)
            probe_expt = np.loadtxt('142614mpa.003', skiprows = 2)
            probe_expt_n1 = np.loadtxt('142614mpa.001', skiprows = 2)
            min_max = [np.argmin(np.abs(probe_expt[:,0] - i)) for i in [1400,2200]]
            probe_plas_ax.set_ylabel('Probe (G)')
            icoil_cur = 1 #kA
            if count == 0: 
                n = 20
                tmp_probe = moving_average(probe_expt[min_max[0]:min_max[1],1]/icoil_cur, n=n)
                tmp_probe_n1 = moving_average(probe_expt_n1[min_max[0]:min_max[1],1]/icoil_cur, n=n)
                tmp_time = moving_average(probe_expt[min_max[0]:min_max[1],0],n=n)
                probe_plas_ax.plot(tmp_time, tmp_probe_n1,'r-', rasterized = True, label='n=1')
                probe_plas_ax.plot(tmp_time, tmp_probe,'k-', rasterized = True, label='n=3')
                probe_plas_ax.legend(loc='upper left',fontsize=8)
                tmp_probe[tmp_time>1860]+=2
                #probe_plas_ax.plot(tmp_time, tmp_probe,'b-', rasterized = True)
        if dBkink_plas_ax!=None:
            dBkink.plot_single_phasing(phasing, 'shot_time', field = 'plasma', plot_kwargs = plot_style, amplitude = True, ax = dBkink_plas_ax, multiplier = multiplier)
            dBkink_plas_ax.set_ylabel(r'$\delta B_{\mathrm{RFA}}$ (G)')
        if q95_ax!=None:
            a.plot_parameters('shot_time', 'Q95', ax = q95_ax, plot_kwargs = {'marker':'x'})
            q95_ax.set_ylabel(r'$q_{95}$')
        if eta_ax!=None:
            a.plot_parameters('shot_time', 'ETA', ax = eta_ax, plot_kwargs = {'marker':'x'})
            eta_ax.set_ylabel(r'$\eta$')
        cur_ax[0].set_xlim([np.min(a.raw_data['shot_time']),np.max(a.raw_data['shot_time'])])

        if count==(len(V_runs)-1) and False:
            for i in cur_ax: 
                vline_times = [1600,1717,2134,1830]
                vline_labels = ['I-coil on', 'CIII image 2', 'CIII image 3', 'plas resp decay']
                vline_times = [1717,2134]
                vline_labels = ['CIII image 2', 'CIII image 3']
                for t_tmp, j in zip(vline_times, vline_labels):
                    i.axvline(t_tmp)
                    i.text(t_tmp,np.mean(i.get_ylim()),j,rotation = 90, verticalalignment='center')
        for i in cur_ax: i.grid(True)
        count+=1
for i in ax.flatten():gen_func.setup_axis_publication(i, n_yticks = 4)
for i in ax[-1,:]: i.set_xlabel('Time (ms)')
ax[-1,-1].set_xlim([1450,2200])
gen_func.setup_axis_publication(ax[-1,0], n_xticks = 5)
fig.tight_layout(pad = 0.1)
for end in ['svg','eps','pdf']:fig.savefig('comparison_oliver_data_allV{}.{}'.format(const_rot,end))
fig.canvas.draw(); fig.show()


#for V in V_runs:
for eta in eta_vals:
    fig_harms, ax_harms = pt.subplots(nrows = 2, sharex = True)
    #ax_harms = [ax_harms]
    gen_func.setup_publication_image(fig_harms, height_prop = 1./1.618 * 1.75, single_col = True)
    for const_rot, cur_ax in zip([False, True], ax_harms):
        if const_rot:
            V_dict = V_dict_const
        else:
            V_dict = V_dict_non_const
        V = V_dict[eta]
        if const_rot:
            if not ideal:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V{}/shot_142614_expt_scan_NC_const_eq_const_rot_prof_V{}_post_processing_PEST.pickle'.format(V,V)
            else:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_const_rot_prof_ideal/shot_142614_expt_scan_NC_const_eq_const_rot_prof_ideal_post_processing_PEST.pickle'
        else:
            if not ideal:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV{}/shot_142614_expt_scan_NC_const_eqV{}_post_processing_PEST.pickle'.format(V,V)
            else:
                file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eq_ideal/shot_142614_expt_scan_NC_const_eq_ideal_post_processing_PEST.pickle'
        #file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_V{}/shot_142614_expt_scan_NC_V{}_post_processing_PEST.pickle'.format(V,V)
        reference_dB_kink = 'plasma'
        a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)

        dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'sum')
        dBkink = dBres_dBkink.dBkink_calculations(a)
        probe = dBres_dBkink.magnetic_probe(a,' 66M')
        xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)

        tmp_a = np.array(dBres.single_phasing_individual_harms(phasing,field='plasma'))
        tmp_b = np.array(dBres.single_phasing_individual_harms(phasing,field='total'))
        tmp_c = np.array(dBres.single_phasing_individual_harms(phasing,field='vacuum'))
        min_time = 1735; max_time = 2175
        min_shot_time = np.min(a.raw_data['shot_time'])
        min_shot_time = min_time
        max_shot_time = np.max(a.raw_data['shot_time'])
        max_shot_time = max_time
        range_shot_time = max_shot_time - min_shot_time
        x_axis = dBres.raw_data['res_m_vals'][0]
        initial = 0
        for i in range(0,tmp_a.shape[0]):
            clr = (a.raw_data['shot_time'][i] - min_shot_time)/float(range_shot_time)
            clr = clr*0.9
            print clr, a.raw_data['shot_time'][i]
            if int(a.raw_data['shot_time'][i])>=min_time and int(a.raw_data['shot_time'][i])<=max_time:
                #ax_harms[0].plot(x_axis, np.abs(tmp_a[i,:]), color=str(clr), marker = 'o')
                #ax_harms[1].plot(x_axis, np.angle(tmp_a[i,:]), color=str(clr), marker = 'o')
                #ax_harms[0].text(tmp_a.shape[1]-1, np.abs(tmp_a[i,-1]), str(a.raw_data['shot_time'][i])+'plasma')
                cur_ax.plot(x_axis, np.abs(tmp_b[i,:]), color=str(clr), marker = 'x')
                #ax_harms[1].plot(x_axis, np.angle(tmp_b[i,:]), color=str(clr), marker = 'x')
                #ax_harms[0].text(tmp_a.shape[1]-1, np.abs(tmp_b[i,-1]), str(a.raw_data['shot_time'][i])+ 'total')
                if initial==0:
                    cur_ax.plot(x_axis, np.abs(tmp_c[i,:]), color='b', marker = '.')
                    #ax_harms[1].plot(x_axis, np.angle(tmp_c[i,:]), color='b', marker = '.')
                    #ax_harms[0].text(tmp_a.shape[1]-1, np.abs(tmp_b[i,-1]), str(a.raw_data['shot_time'][i]))
                initial += 1
    #ax_harms[0].set_title('{}-{}ms $\eta=${}'.format(min_time, max_time, eta))
    ax_harms[-1].set_xlabel('m')
    for i in ax_harms: i.set_ylabel('Resonant harm amp (G/kA)')
    #ax_harms[1].set_ylabel('Resonant harm phase (rad)')
    for i in ax_harms:i.grid(True)
    fig_harms.tight_layout(pad = 0.1)
    for end in ['svg','eps','pdf']:fig_harms.savefig('harms_{}_{}.{}'.format(eta, const_rot,end))
    fig_harms.canvas.draw(); fig_harms.show()

1/0

#fig, ax = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax = ax.flatten()
#fig2, ax2_orig = pt.subplots(ncols = 4, nrows = 2, sharex = True, sharey = True); ax2 = ax2_orig.flatten()
phasings_disp = [0,45,90,135,180,225,270,315]
phasings_disp = [0,180]
phasings_disp = [0]
fig, ax = pt.subplots(nrows = 7, sharex = True)
gen_func.setup_publication_image(fig, height_prop = 1./1.618*4, single_col = True)

for i in range(len(phasings_disp)):
    #a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = ax[i], ax_matrix = ax2[i], clim = [0, 0.015])
    #tmp, color_ax = a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = None, ax_matrix = ax2[i], clim = [0, 0.025])
    #tmp, color_ax = a.extract_organise_single_disp(phasings_disp[i], ax_line_plots = None, ax_matrix = None, clim = [0, 0.025])
    a.plot_values_vs_time(phasings_disp[i], ax = ax)
#dBres
tmp_vac_list, tmp_plas_list, tmp_tot_list, tmp_vac_list2, tmp_plas_list2,  tmp_tot_list2 = a.dB_res_single_phasing(i, phase_machine_ntor, n, a.res_vac_list_upper, a.res_vac_list_lower, a.res_plas_list_upper, a.res_plas_list_lower, a.res_tot_list_upper, a.res_tot_list_lower)
#tmp = np.sort([[t, res] for t, res in zip(a.time_list, tmp_plas_list)],axis = 0)
tmp = zip(a.time_list, tmp_plas_list)
tmp.sort()
#tmp = sorted(zip(a.time_list, tmp_plas_list), key = lambda sort_val:sort_val[0]) 
ax[2].plot([t for t, res in tmp], [res for t, res in tmp],'x-')
ax[2].set_ylabel('dBres plas')

#print a.time_list, tmp_tot_list
#tmp = np.sort([[t, res] for t, res in zip(a.time_list, tmp_tot_list)],axis = 0)
#tmp = sorted(zip(a.time_list, tmp_tot_list), key = lambda sort_val:sort_val[0]) 
tmp = zip(a.time_list, tmp_tot_list)
tmp.sort()
ax[3].plot([t for t, res in tmp], [res for t, res in tmp],'x-')
ax[3].set_ylabel('dBres tot')

a.plot_dB_res_ind_harmonics(0)

a.plot_probe_values_vs_time(0,' 66M',field='plas', ax = ax[4])

name_list = ['plot_array_plasma', 'plot_array_vac', 'plot_array_tot', 'plot_array_vac_fixed', 'q95_array', 'phasing_array', 'plot_array_plasma_fixed', 'plot_array_plasma_phase', 'plot_array_vac_phase', 'plot_array_vac_fixed_phase', 'plot_array_plasma_fixed_phase']
tmp1 = dBres_dBkink.dB_kink_phasing_dependence(a.q95_list_copy, a.lower_values_plasma, a.upper_values_plasma, a.lower_values_vac, a.upper_values_vac, a.lower_values_tot, a.upper_values_tot, a.lower_values_vac_fixed, a.upper_values_vac_fixed, phase_machine_ntor, a.upper_values_plas_fixed, a.lower_values_plas_fixed, n, phasing_array = [0])
#tmp = np.sort([[t, kink] for t, kink in zip(a.time_list, tmp1[0].flatten().tolist())],axis = 0)

tmp = zip(a.time_list, tmp1[0].flatten().tolist())
tmp.sort()

ax[5].plot([t for t, kink in tmp], [kink for t, kink in tmp],'x-')
ax[5].set_ylabel('dBkink')

tmp = np.sort([[t, q] for t, q in zip(a.time_list, a.q95_list)],axis = 0)
ax[6].plot([t for t, q in tmp], [q for t, q in tmp],'x-')
ax[6].set_ylabel('q')

for i in ax: 
    vline_times = [1600,1717,2134,1830]
    vline_labels = ['I-coil on', 'CIII image 2', 'CIII image 3', 'plas resp decay']
    for t_tmp, j in zip(vline_times, vline_labels):
        i.axvline(t_tmp)
        i.text(t_tmp,np.mean(i.get_ylim()),j,rotation = 90, verticalalignment='center')
    i.grid(True)

fig.tight_layout(pad = 0.01)
fig.savefig('comparison_oliver_data.pdf')
fig.canvas.draw(); fig.show()

#cbar = pt.colorbar(color_ax, ax = ax2.tolist())
#cbar.set_label('Displacement around x-point')
#fig.canvas.draw(); fig.show()
#fig2.savefig('res_rot_scan_displacement.pdf')
#fig2.canvas.draw(); fig2.show()
#a.eta_rote_matrix(phasing = 0, plot_type = 'plas')
