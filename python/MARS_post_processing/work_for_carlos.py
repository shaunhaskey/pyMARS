import numpy as np
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func

#Ideal cases
phasing = 0
n = 2
phase_machine_ntor = 0
s_surface = 0.94
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [4,0]
#reference_offset = [2,0]
sort_name = 'time_list'

file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_imp_grid/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04272_imp_grid/shot158115_04272_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_inc_MPID/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_imp_grid_0freq/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_imp_grid_0freq_B23_2/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan/shot158115_04780_scan_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan_high_res_low_rot/shot158115_04780_scan_high_res_low_rot_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan_ideal/shot158115_04780_scan_ideal_post_processing_PEST.pickle'


file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan/shot158115_04780_scan_post_processing_PEST.pickle'

def extract_useful(file_name, m = 10, probe_names = None):
    print file_name
    reference_dB_kink = 'plasma'
    a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
    dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean')
    dBkink = dBres_dBkink.dBkink_calculations(a)
    if probe_names == None:
        names = ['66M', 'MPID1A', 'MPID1B']
    else:
        names = [i.lstrip(' ').rstrip(' ') for i in probe_names]
    #m = 10
    min_loc = np.argmin(np.abs(np.array(dBres.raw_data['res_m_vals'][0])-m))
    n_simuls = len(dBres.raw_data['res_m_vals'])
    results = {'rfa':{},'res':{},'probe':{}}
    fields = ['vacuum','plasma','total']
    #Check to see if it is a scan
    if n_simuls>1:
        for field in fields:
            phases, rfa_vals = dBkink.phasing_scan(field = field,n_phases = 90)
            results['rfa'][field] = +rfa_vals.T

            vals = []
            for q in range(n_simuls):
                vals.append([dBres.single_phasing_individual_harms(i,field=field)[q][min_loc] for i in phases])
            #for i in phases:vals.append(dBres.single_phasing_individual_harms(i,field=field)[q][min_loc])
            results['res'][field] = np.array(vals)
        #names = [' 66M', ' MPID1A', ' MPID1B']
        for i in names:
            results['probe'][i] = {}
            probe = dBres_dBkink.magnetic_probe(a, i)
            for field in fields:
                phases, vals = probe.phasing_scan(field = field, n_phases = 90)
                results['probe'][i][field] = +vals.T
            #Need to implement this later on....
            print_results = False
            if print_results:
                tmp = a.project_dict['details']['pickup_coils']
                for q in range(n_simuls):
                    print '###### {} #####'.format(name)
                    print 'R={:.3f}, Z={:.3f}m, l_probe={:.3f}m, inc={:.3f}rad, pol={}, betaN/li={Bnli}'.format(*[tmp[i][ind] for i in ['Rprobe', 'Zprobe', 'lprobe','tprobe','probe_type']], Bnli = betaN_li[q]) 
                    for field in ['plasma','vacuum','total']:
                        for ul in ['upper','lower']:
                            print '{}_{}='.format(field, ul), probe.raw_data['{}_probe_{}'.format(field, ul)][q]

    else:
        fig, ax = pt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = True)
        for q in range(len(dBres.raw_data['res_m_vals'])):
            for field, style in zip(['vacuum','plasma','total'], ['o-','x-','.-']):
                phases, vals = dBkink.phasing_scan(field = field,n_phases = 90)
                ax[0].plot(phases, np.abs(vals), style, label = field)
                vals = []
                for i in phases:
                    vals.append(dBres.single_phasing_individual_harms(i,field=field)[q][min_loc])
                ax[1].plot(phases, np.abs(vals), style, label =field)
            ax[-1].set_xlabel('Upper-lower I-coil phasing (deg)')
            #ax[0].set_title(rfa_title)
            #ax[1].set_title(pitch_res_title)
            for i in ax: i.set_ylabel('Harmonic amplitude (G/kA)')
            for i in ax: i.legend(loc='best')
            for i in ax: i.grid()
            ax[0].set_xlim([0,360])
            fig.canvas.draw();fig.show()
    betaN_li = np.array(dBres.parent.raw_data['BETAN'])/ np.array(dBres.parent.raw_data['LI'])

    return results, phases, betaN_li

def max_amp_max_phase(inp_dict, name, field,phases, ax1 = None, ax2 = None, ax3 = None, x_ax = None, plot_kwargs = None, forced_phase = None, print_vals = False):
    data = inp_dict[name][field]
    if plot_kwargs == None: plot_kwargs = {}
    if forced_phase!=None: 
        max_locs = [np.argmin(np.abs(phases-forced_phase))]*(data.shape[0])
        #print 'using a foced phasing'
        #print max_locs, len(max_locs), data.shape
    else:
        max_locs = np.argmax(np.abs(data), axis = 1)
    x = range(data.shape[0])
    if ax1!=None: 
        y_ax = np.abs(data)[x,max_locs]
        ax1.plot(x_ax[np.argsort(x_ax)], y_ax[np.argsort(x_ax)] , **plot_kwargs)
        if print_vals:
            print '['+','.join(['{:.3f}'.format(i) for i in x_ax[np.argsort(x_ax)]])+']'
            print '['+','.join(['{:.3f}'.format(i) for i in y_ax[np.argsort(x_ax)]])+']'
        #print name
        #print x_ax
        #print data[x,max_locs]
    if ax2!=None: 
        y_ax = np.angle(data, deg = True)[x,max_locs]
        ax2.plot(x_ax[np.argsort(x_ax)], y_ax[np.argsort(x_ax)] , **plot_kwargs)
        #ax2.plot(x_ax, np.angle(data, deg = True)[x,max_locs], **plot_kwargs)
    if ax3!=None: 
        y_ax = phases[max_locs]
        ax3.plot(x_ax[np.argsort(x_ax)], y_ax[np.argsort(x_ax)] , **plot_kwargs)
        #ax3.plot(x_ax, , **plot_kwargs)

    return phases[max_locs], np.abs(data)[x,max_locs], np.angle(data, deg = True)[x,max_locs]


#max_phases, max_amps, max_angs = max_amp_max_phase(results1, 'res', 'plasma', phases)
#max_phases, max_amps, max_angs = max_amp_max_phase(results1, 'rfa', 'plasma', phases)


probe_names = ['66M', 'MPID1A', 'MPID1B']
probe_names = ['66M']#, 'MPID1A', 'MPID1B']
#probe_names = ['66M', 'Inner_pol']# ' MPID1A', ' MPID1B']
labels = ['ideal MHD', 'resistive MHD']
colour = ['b','r','k','y']
markers = ['x','o','.','s','d']
file_names = ['/home/srh112/NAMP_datafiles/mars/shot158115_04701_betaN_ramp_raffi_prl/shot158115_04701_betaN_ramp_raffi_prl_post_processing_PEST.pickle','/home/srh112/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_raffi_prl/shot158115_04780_betaN_ramp_raffi_prl_post_processing_PEST.pickle']
dirs = ['shot158115_04701_betaN_ramp_raffi_prl', 'shot158115_04780_betaN_ramp_raffi_prl', 'shot158115_04780_betaN_ramp_raffi_prl']

dirs = ['shot156746_02113_betaN_ramp_carlos_prl2', 'shot156746_03322_betaN_ramp_carlos_prl']
dirs = ['shot156746_02113_betaN_ramp_carlos_prl2', '156746_02113_betaNramp']
labels = ['2113', '3322']
dirs = ['shot156746_0{}_betaN_ramp_carlos_prlV2', 'shot156746_0{}_betaN_ramp_carlos_prlV2']
labels = ['2113']#, '3322']
labels = ['2113','2113-2']#, '3322']
dirs = ['shot156746_0{}_betaN_ramp_carlos_prlV2']#, 'shot156746_0{}_betaN_ramp_carlos_prlV2']
for i in range(len(dirs)):
    dirs[i]= dirs[i].format(labels[i])
#dirs = ['shot156746_02113_betaN_ramp_carlos_prl_n1','shot156746_02113_betaN_ramp_carlos_prl_n1_pos']
#dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2_n1', 'shot156746_02113_betaN_ramp_carlos_prlV2_n1_5Hz']
#dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2_n1', 'shot156746_02113_betaN_ramp_carlos_prlV2_n1_5Hz','shot156746_02113_betaN_ramp_carlos_prlV2_n1_20Hz_ideal_PVISC0']
#dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2_n1_5Hz','shot156746_02113_betaN_ramp_carlos_prlV2_n1_20Hz_ideal_PVISC0']
#dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2_n1_5Hz','shot156746_02113_betaN_ramp_carlos_prlV2_n1_5Hz_ideal_PVISC1']

dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2_n1',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n1_20Hz_ideal_PVISC0']
labels = ['']
dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2', 'shot156746_02113_betaN_ramp_carlos_prlV2_n2_5Hz_res_PVISC0']

#n=2 options
labels = ['res20PVISC1','res5PVISC0','ideal5PVISC1','ideal5PVISC0','res5PVISC1']
dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2', 'shot156746_02113_betaN_ramp_carlos_prlV2_n2_5Hz_res_PVISC0', 'shot156746_02113_betaN_ramp_carlos_prlV2_n2_5Hz_ideal_PVISC1','shot156746_02113_betaN_ramp_carlos_prlV2_n2_5Hz_ideal_PVISC0','shot156746_02113_betaN_ramp_carlos_prlV2_n2_5Hz_res_PVISC1']

labels = ['res20PVISC1','res20PVISC0','ideal20PVISC0','ideal20PVISC1']
dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n2_20Hz_res_PVISC0',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n2_20Hz_ideal_PVISC0',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n2_20Hz_ideal_PVISC1']

labels = ['res20PVISC1','res20PVISC0','ideal20PVISC0','ideal20PVISC1', 'rote20PVISC0']
dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n2_20Hz_res_PVISC0',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n2_20Hz_ideal_PVISC0',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n2_20Hz_ideal_PVISC1',
        'shot156746_02113_betaN_ramp_carlos_prlV2_n2_20Hz_rote_PVISC0']

#dirs = ['shot156746_02113_betaN_ramp_carlos_prlV2_n1','shot156746_02113_betaN_ramp_carlos_prlV2_n1_20Hz_ideal_PVISC0']
print dirs
#file_names = ['/home/srh112/NAMP_datafiles/mars/shot156746_02113_betaN_ramp_carlos_prl2']
file_names = []
base_dir = r'/home/srh112/NAMP_datafiles/mars/'
for dir_tmp in dirs:
    file_names.append('{}/{}/{}_post_processing_PEST.pickle'.format(base_dir, dir_tmp, dir_tmp))

file_names = ['/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan/shot158115_04780_scan_post_processing_PEST.pickle']

fig, ax = pt.subplots(nrows = 3, sharex = True) #ax = [ax]
for file_name, label, marker in zip(file_names, labels, markers):
    results, phases, betaN_li = extract_useful(file_name, m = 10, probe_names = probe_names)
    forced_phase = 0
    keys = probe_names#keys = ['66M', ' MPID1B']
    #keys = results1['probe'].keys()
    #marker = ['o']
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    for i, clr in zip(keys, colour):
        plot_kwargs = {'marker': marker, 'color': clr, 'label' :'{} - {}'.format(i, label)}
        #plot_kwargs = {'marker': 'o', 'color':clr, 'label' :'{}'.format(i.replace('_','\_'))}
        print i
        max_phases, max_amps, max_angs = max_amp_max_phase(results['probe'], i, 'plasma', phases, ax1 = ax[0], ax2 = ax2, ax3 = ax3, x_ax = betaN_li, plot_kwargs = plot_kwargs, forced_phase = forced_phase, print_vals = True)
        #lab = '{} - {}'.format(i,'ideal MHD')
        #max_phases, max_amps, max_angs = max_amp_max_phase(results['probe'], i, 'plasma', phases, ax1 = ax[0], ax2 = ax2, ax3 = ax3, x_ax = betaN_li2, plot_kwargs = plot_kwargs, forced_phase = forced_phase)
    #ax[0].set_xlim([0.5,3.177])
ax[0].set_xlim([2.,5])
ax[0].set_ylim([0,3.5])
ax[0].set_ylim([0,15.])
ax[-1].set_xlabel('BetaN/Li')
ax[0].set_ylabel('G/kA')
if ax[1]!=None: ax[1].set_ylabel('Signal Phase (deg)')
if ax[2]!=None: ax[2].set_ylabel('Optimum UL phasing (deg)')
for i in ax: ax[0].set_ylabel('G/kA')
leg = ax[0].legend(loc = 'best', fontsize = 8)
fig.canvas.draw(); fig.show()

flux = 10
for i in a.project_dict['sims'].keys():
    A = a.project_dict['sims'][i]['responses']['resonant_response_A_res']
    qn = a.project_dict['sims'][i]['responses']['resonant_response_qn']
    sq = a.project_dict['sims'][i]['responses']['resonant_response_sq']
    mq = a.project_dict['sims'][i]['responses']['resonant_response_mq']
    dqdpsi = a.project_dict['sims'][i]['responses']['resonant_response_dqdpsiN_res']

    Bmn = np.abs(a.project_dict['sims'][i]['responses']['total_resonant_response_upper'] + a.project_dict['sims'][i]['responses']['total_resonant_response_lower'])
    island_width = np.sqrt(16./(mq*flux) * qn/np.abs(dqdpsi) * A/(4.*np.pi**2) * Bmn)
    print island_width
1/0
fig, ax = pt.subplots(nrows = 3, sharex = True)
for i, clr in zip(['res','rfa'], colour):
    plot_kwargs = {'marker': 'o', 'color':clr}
    max_phases, max_amps, max_angs = max_amp_max_phase(results1, i, 'plasma', phases, ax1 = ax[0], ax2=ax[1], ax3 = ax[2], x_ax = betaN_li1, plot_kwargs = plot_kwargs, forced_phase = forced_phase)
    plot_kwargs = {'marker': 'x', 'color':clr}
    max_phases, max_amps, max_angs = max_amp_max_phase(results2, i, 'plasma', phases, ax1 = ax[0], ax2=ax[1], ax3 = ax[2], x_ax = betaN_li2, plot_kwargs = plot_kwargs, forced_phase = forced_phase)
ax[0].set_xlim([0.5,3.177])
fig.canvas.draw(); fig.show()


#For Raffi 12Sept2014
probe_names = ['66M', 'MPID1A']#, 'MPID1B']
#probe_names = ['66M', 'Inner_pol']# ' MPID1A', ' MPID1B']
labels = ['ideal MHD', 'resistive MHD']
colour = ['b','r','k']
markers = ['x','o','.']
labels = ['4780', '4702']
file_names = ['/home/srh112/NAMP_datafiles/mars/shot158115_04780_single_raffi2/shot158115_0{}_single_raffi2_post_processing_PEST.pickle','/home/srh112/NAMP_datafiles/mars/shot158115_04702_single_raffi2/shot158115_0{}_single_raffi2_post_processing_PEST.pickle']
file_names = ['/home/srh112/NAMP_datafiles/mars/shot158115_04780_single_raffi2_ideal/shot158115_0{}_single_raffi2_ideal_post_processing_PEST.pickle','/home/srh112/NAMP_datafiles/mars/shot158115_04702_single_raffi2_ideal/shot158115_0{}_single_raffi2_ideal_post_processing_PEST.pickle']
fig, ax = pt.subplots(nrows = len(probe_names), sharex = True); #ax = [ax]
for file_name, label, marker in zip(file_names, labels, markers):
    file_name = file_name.format(label)
    a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
    results = {'probe':{}}
    for i, ax_tmp in zip(probe_names, ax):
        results['probe'][i] = {}
        probe = dBres_dBkink.magnetic_probe(a, i)
        fields = ['vacuum','plasma','total']
        phases, vals = probe.phasing_scan(field = 'plasma', n_phases = 90)
        ax_tmp.plot(phases, np.abs(vals), label = label)
ax[0].legend()
fig.canvas.draw(); fig.show()

1/0

max_phases, max_amps_rfa, max_angs = max_amp_max_phase(results1, 'rfa', 'plasma', phases, ax1 = ax[0], ax2=ax[1], x_ax = betaN_li, plot_kwargs = plot_kwargs)
max_phases, max_amps_res, max_angs = max_amp_max_phase(results1, 'res', 'plasma', phases, ax1 = ax[0], ax2=ax[1], x_ax = betaN_li, plot_kwargs = plot_kwargs)
max_phases2, max_amps_66M, max_angs2 = max_amp_max_phase(results2['probe'], ' 66M', 'plasma', phases, ax1 = ax[0], ax2=ax[1], x_ax = betaN_li, plot_kwargs = plot_kwargs)
max_phases2, max_amps_MPID, max_angs2 = max_amp_max_phase(results2['probe'], ' MPID1A', 'plasma', phases, ax1 = ax[0], ax2=ax[1], x_ax = betaN_li, plot_kwargs = plot_kwargs)
print np.corrcoef(np.array([max_amps_rfa,max_amps_66M]))
print np.corrcoef(np.array([max_amps_rfa,max_amps_MPID]))
print np.corrcoef(np.array([max_amps_res,max_amps_66M]))
print np.corrcoef(np.array([max_amps_res,max_amps_MPID]))

1/0
results2 = extract_useful(file_name, m = 10)

#xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)

# tmp_a = np.array(dBres.single_phasing_individual_harms(phasing,field='plasma'))
# tmp_b = np.array(dBres.single_phasing_individual_harms(phasing,field='total'))
# tmp_c = np.array(dBres.single_phasing_individual_harms(phasing,field='vacuum'))

# phases_res, vals_res = dBres.phasing_scan()
# fig, ax = pt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)
# ax.plot(phases, np.abs(vals))
# phases, vals = dBkink.phasing_scan(field = 'plasma')
# ax.plot(phases, np.abs(vals))
# phases, vals = dBkink.phasing_scan(field = 'vacuum')
# ax.plot(phases, np.abs(vals))
# fig.canvas.draw();fig.show()


m = 10
min_loc = np.argmin(np.abs(np.array(dBres.raw_data['res_m_vals'][0])-m))
rfa_title = 'RFA : m = {}, $\psi_N$ = {:.3f}'.format(dBkink.raw_data['plasma_max_mode_list_upper'][0], s_surface)
pitch_res_title = 'Pitch Resonant : m = {}, q = {}, $\psi_N$ = {:.3f}'.format(dBres.raw_data['res_m_vals'][0][min_loc],  dBres.raw_data['res_q_vals'][0][min_loc], dBres.raw_data['res_s_vals'][0][min_loc]**2,)
n_simuls = len(dBres.raw_data['res_m_vals'])

if n_simuls>1:
    fig, ax = pt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = True)
    betaN_li = np.array(dBres.parent.raw_data['BETAN'])/ np.array(dBres.parent.raw_data['LI'])
    grad = np.max(betaN_li) - np.min(betaN_li)
    off = np.min(betaN_li)
    clr_list = ['{:.2f}'.format(0.9/grad * (betaN_li[i] - off)) for i in range(n_simuls)]
    max_phasings_rfa = {'vacuum':[],'plasma':[],'total':[]}
    max_phasings_res = {'vacuum':[],'plasma':[],'total':[]}
    max_amp_rfa = {'vacuum':[],'plasma':[],'total':[]}
    max_amp_res = {'vacuum':[],'plasma':[],'total':[]}
    for q in range(n_simuls):
        clr = clr_list[q]
        for field, style in zip(['vacuum','plasma','total'], ['o-','x-','.-']):
            phases, vals = dBkink.phasing_scan(field = field,n_phases = 90)
            lab = field if q==0 else None
            ax[0].plot(phases, np.abs(vals[:,q]), style, label = lab, color = clr)
            max_phasings_rfa[field].append(phases[np.argmax(np.abs(vals[:,q]))])
            max_amp_rfa[field].append(max(np.abs(vals[:,q])))
            vals = []
            for i in phases:
                vals.append(dBres.single_phasing_individual_harms(i,field=field)[q][min_loc])
            ax[1].plot(phases, np.abs(vals), style, label = lab, color = clr)
            max_phasings_res[field].append(phases[np.argmax(np.abs(vals))])
            max_amp_res[field].append(np.max(np.abs(vals)))
        ax[-1].set_xlabel('Upper-lower I-coil phasing (deg)')
        ax[0].set_title(rfa_title)
        ax[1].set_title(pitch_res_title)
        for i in ax: i.set_ylabel('Harmonic amplitude (G/kA)')
        for i in ax: 
            leg = i.legend(loc='best')
            leg.draw_frame(False)
        for i in ax: i.grid()
        ax[0].set_xlim([0,360])
    fig.canvas.draw();fig.show()
else:
    fig, ax = pt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = True)
    for q in range(len(dBres.raw_data['res_m_vals'])):
        for field, style in zip(['vacuum','plasma','total'], ['o-','x-','.-']):
            phases, vals = dBkink.phasing_scan(field = field,n_phases = 90)
            ax[0].plot(phases, np.abs(vals), style, label = field)
            vals = []
            for i in phases:
                vals.append(dBres.single_phasing_individual_harms(i,field=field)[q][min_loc])
            ax[1].plot(phases, np.abs(vals), style, label =field)
        ax[-1].set_xlabel('Upper-lower I-coil phasing (deg)')
        ax[0].set_title(rfa_title)
        ax[1].set_title(pitch_res_title)
        for i in ax: i.set_ylabel('Harmonic amplitude (G/kA)')
        for i in ax: i.legend(loc='best')
        for i in ax: i.grid()
        ax[0].set_xlim([0,360])
        fig.canvas.draw();fig.show()

fig, ax = pt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)
z = np.array(max_phasings_res['vacuum'])
z[z>200]-=360
ax[0].plot(betaN_li, z,'-bo', label = 'pitch-res vac')
z = np.array(max_phasings_res['total'])
z[z>200]-=360
ax[0].plot(betaN_li,z,'-xk', label = 'pitch-res total')
z = np.array(max_phasings_rfa['plasma'])
z[z>200]-=360
ax[0].plot(betaN_li, z,'-sr', label = 'rfa')
ax[0].set_xlim([0.7,3.5])
ax[0].set_xlabel('BetaN/li')
ax[0].set_ylabel('Maximising upper-lower phasing (deg)')
leg = ax[0].legend(loc = 'best')
leg.draw_frame(False)
#fig, ax = pt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)
z = np.array(max_amp_res['vacuum'])
#z[z>200]-=360
ax[1].plot(betaN_li, z,'-bo', label = 'pitch-res vac')
z = np.array(max_amp_res['total'])
#z[z>200]-=360
ax[1].plot(betaN_li,z,'-xk', label = 'pitch-res total')
z = np.array(max_amp_rfa['plasma'])
#z[z>200]-=360
ax[1].plot(betaN_li, z,'-sr', label = 'rfa')
ax[1].set_xlim([0.7,3.5])
ax[1].set_xlabel('BetaN/li')
ax[1].set_ylabel('Amplitude at max phasing')
leg = ax[1].legend(loc = 'best')
leg.draw_frame(False)
fig.canvas.draw();fig.show()

#Print out and plot the probe complex values
fig, ax = pt.subplots(nrows =3, sharex = True)
names = [' 66M', 'Inner_pol']
names = [' 66M', ' MPID1A', ' MPID1B']
names = [' 66M', ' MPID1A',]
marker_list = ['x','.','o']
print file_name
probe_max = {}
probe_phase = {}
for name,marker in zip(names,marker_list):
    probe_max[name] = {'vacuum':[],'plasma':[],'total':[]}
    probe_phase[name] = {'vacuum':[],'plasma':[],'total':[]}

    ind = a.project_dict['details']['pickup_coils']['probe'].index(name)
    probe = dBres_dBkink.magnetic_probe(a,name)
    print ''
    tmp = a.project_dict['details']['pickup_coils']
    for q in range(n_simuls):
        print '###### {} #####'.format(name)
        print 'R={:.3f}, Z={:.3f}m, l_probe={:.3f}m, inc={:.3f}rad, pol={}, betaN/li={Bnli}'.format(*[tmp[i][ind] for i in ['Rprobe', 'Zprobe', 'lprobe','tprobe','probe_type']], Bnli = betaN_li[q]) 
        for field in ['plasma','vacuum','total']:
            for ul in ['upper','lower']:
                print '{}_{}='.format(field, ul), probe.raw_data['{}_probe_{}'.format(field, ul)][q]
    for q in range(n_simuls):
        pu, pl= [probe.raw_data['plasma_probe_upper'][q], probe.raw_data['plasma_probe_lower'][q]]
        lab = name if q==0 else None
        tmp_phases = np.linspace(0,360,100)
        tmp_vals = np.abs(pu + pl * np.exp(1j*np.deg2rad(tmp_phases)))
        ax[0].plot(tmp_phases, tmp_vals,label=lab, color = clr_list[q], marker=marker)
        ax[1].plot(tmp_phases, np.rad2deg(np.angle(tmp_vals))%360,label=lab,color = clr_list[q], marker=marker)
        probe_max[name]['plasma'].append(max(np.abs(tmp_vals)))
        probe_phase[name]['plasma'].append(tmp_phases[np.argmax(np.abs(tmp_vals))])

fig20, ax20 = pt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)
for tmp_name in names:
    z = np.array(probe_phase[tmp_name]['plasma'])
    z[z>200]-=360
    ax20[0].plot(betaN_li, z,'-o', label = tmp_name)
    leg = ax20[0].legend(loc = 'best')
    leg.draw_frame(False)
    z = np.array(probe_max[tmp_name]['plasma'])
    ax20[1].plot(betaN_li, z,'-o', label = tmp_name)
    leg = ax[1].legend(loc = 'best')
    leg.draw_frame(False)
fig20.canvas.draw();fig20.show()


#Print out and plot the res metric values


print ''
for q in range(n_simuls):
    print '###### {} #####'.format('res_metric')
    pitch_res_title = 'Pitch Resonant : m = {}, q = {}, $\psi_N$ = {:.3f}, betaN/li={Bnli}'.format(dBres.raw_data['res_m_vals'][q][min_loc],  dBres.raw_data['res_q_vals'][q][min_loc], dBres.raw_data['res_s_vals'][q][min_loc]**2, Bnli = betaN_li[q])
    print pitch_res_title
    for field in ['plasma','vacuum','total']:
        for ul in ['upper','lower']:
            print '{}_{}='.format(field, ul), dBres.raw_data['{}_res_{}'.format(field, ul)][q][min_loc]

field = 'vacuum'

for q in range(n_simuls):
    u, l = dBres.raw_data['{}_res_{}'.format(field, 'upper')][q][min_loc], dBres.raw_data['{}_res_{}'.format(field, 'lower')][q][min_loc]
    lab = 'res vac' if q==0 else None
    ax[2].plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))), label = lab, color=clr_list[q], marker = 'x')



print ''
for q in range(n_simuls):
    print '###### {} #####'.format('rfa_metric')
    rfa_title = 'RFA : m = {}, $\psi_N$ = {:.3f}, betaN/li={Bnli}'.format(dBkink.raw_data['plasma_max_mode_list_upper'][q], s_surface, Bnli = betaN_li[q])
    print rfa_title
    #Print out and plot the rfa metric
    for field in ['plasma','vacuum','total']:
        for ul in ['upper','lower']:
            print '{}_{} ='.format(field, ul), dBkink.raw_data['{}_kink_harm_{}'.format(field, ul)][q]
    field = 'plasma'

for q in range(n_simuls):
    u, l = dBkink.raw_data['{}_kink_harm_{}'.format(field, 'upper')][q], dBkink.raw_data['{}_kink_harm_{}'.format(field, 'lower')][q]
    lab = 'rfa' if q==0 else None
    ax[2].plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))), label = lab, color=clr_list[q], marker = 'o')
ax[-1].set_xlim([0,360])
ax[0].set_ylabel('mod(B) G/kA')
ax[1].set_ylabel('arg(B) deg')
ax[-1].set_xlabel('phasing')
leg = ax[2].legend(loc = 'best')
leg.draw_frame(False)
leg = ax[0].legend(loc = 'best')
leg.draw_frame(False)
leg = ax[1].legend(loc = 'best')
leg.draw_frame(False)
fig.suptitle(file_name.replace('_','-'))
fig.canvas.draw()
fig.canvas.draw(); fig.show()

