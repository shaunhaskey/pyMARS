import numpy as np
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func
import matplotlib.colors as colors
import matplotlib.cm as cmx

#Sort this into more useful script
#Output RFA metric
#Output displacement metric
#Check l-mode
#Ideal cases
phasing = 0
n = 2
phase_machine_ntor = 0
s_surface = 0.94
fixed_harmonic = 4
reference_dB_kink = 'plasma'
reference_offset = [4,0]
#reference_offset = [2,0]
sort_name = 'time_list'
sort_name = 'betaN_li'

file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan/shot158115_04780_scan_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan_bak/shot158115_04780_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_scan/shot158115_04780_scan_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot158103_03796_betaN_ramp_carlos_hicol/shot158103_03796_betaN_ramp_carlos_hicol_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retest/shot158115_04780_betaN_ramp_retest_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retest2/shot158115_04780_betaN_ramp_retest2_post_processing_PEST.pickle'
#file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retestV2/shot158115_04780_betaN_ramp_retestV2_post_processing_PEST.pickle'


#These are the the set of simulations
file_name = '/home/srh112/NAMP_datafiles/mars/shot158103_03796_betaN_ramp_carlos_2/shot158103_03796_betaN_ramp_carlos_2_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/shot161198_03550_betaN_ramp_carlos_hicol2/shot161198_03550_betaN_ramp_carlos_hicol2_post_processing_PEST.pickle'

#file_name = '/home/srh112/NAMP_datafiles/mars/shot161205_03215_betaN_ramp_carlos_lmode2/shot161205_03215_betaN_ramp_carlos_lmode2_post_processing_PEST.pickle'

names = [' 66M', 'Inner_pol']
names = [' 66M', ' MPID1A', ' MPID1B']
names = ['66M', 'MPID1A',]
#names = [' 66M']
#names = ['MPID1A',]

fig = pt.figure()
nrows = 3
ncols = len(names)
ax_overall = []
#fig, ax = pt.subplots(nrows = 3, ncols = ncols, sharex = True)

for j in range(ncols):
    ax_tmp = []
    for i in range(nrows):
        if j==0:
            ax_tmp.append(fig.add_subplot(nrows,ncols,i*ncols + j + 1))
        else:
            ax_tmp.append(fig.add_subplot(nrows,ncols,i*ncols + j + 1, sharex = ax_overall[0][i], sharey = ax_overall[0][i]))
    ax_overall.append(ax_tmp)

#for i in range(1,ncols):
#    ax_tmp.append(fig.add_subplot(nrows,ncols,i*ncols + 1))

fig.canvas.draw()
marker_list = ['x','.','o']
print file_name
probe_max = {}
probe_phase = {}
probe_names = names
a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean')
dBkink = dBres_dBkink.dBkink_calculations(a)
n_simuls = len(dBres.raw_data['res_m_vals'])
betaN_li = np.array(dBres.parent.raw_data['BETAN'])/ np.array(dBres.parent.raw_data['LI'])
betaN_li_sort = np.argsort(betaN_li)
PMULT = np.array(dBres.parent.raw_data['PMULT'])
#QMULT = np.array(dBres.parent.raw_data['QMULT'])
grad = np.max(betaN_li) - np.min(betaN_li)
off = np.min(betaN_li)
clr_list = ['{:.2f}'.format(0.9/grad * (betaN_li[i] - off)) for i in range(n_simuls)]

fig25,ax25 = pt.subplots()
data_out = []
disp_txt = []
header = 'metric betaN/li PMULT '
for i in range(0,360,45):header+= '{} '.format(i) 
disp_txt.append(header.rstrip(' ')+'\n')
for i in range(0,360,45):
    x_point = dBres_dBkink.x_point_displacement_calcs(a, i)
    data_out.append(x_point.raw_data['plasma_dispx_'])
    #ax25.plot(x_point.parent.raw_data['BETAN'], x_point.raw_data['plasma_dispx_'],'o',color='{:.2f}'.format(float(i)/360.))
data_out = np.array(data_out)
for q in betaN_li_sort:# range(n_simuls):
#for i in range(data_out.shape[1]):
    cur_list = 'xptdisp {:.4f} {:.4f} '.format(x_point.parent.raw_data['BETAN'][q]/x_point.parent.raw_data['LI'][q],x_point.parent.raw_data['PMULT'][q])
    cur_list += ' '.join(['{:.4e}'.format(j) for j in data_out[:,q]])
    disp_txt.append(cur_list.rstrip(' ')+'\n')
    ax25.plot(range(0,360,45), data_out[:,q],color='{:.2f}'.format(x_point.parent.raw_data['BETAN'][q]/np.max(x_point.parent.raw_data['BETAN'])))
with file('{}_disp.txt'.format(file_name.split('/')[-1].rstrip('post_processing_PEST.pickle')),'w') as filehandle:
    filehandle.writelines(disp_txt)
fig25.canvas.draw();fig25.show()

cmap_cycle = gen_func.new_color_cycle(0,4)

text_output = []

field_list = ['plasma','vacuum','total']
ul_list = ['upper','lower']
extra_info = ['PMULT','BNLI']
for ident, (name,marker,ax) in enumerate(zip(names,marker_list,ax_overall)):
    ax[0].set_title(name)
    probe_max[name] = {'vacuum':[],'plasma':[],'total':[]}
    probe_phase[name] = {'vacuum':[],'plasma':[],'total':[]}

    ind = a.project_dict['details']['pickup_coils']['probe'].index(name)
    probe = dBres_dBkink.magnetic_probe(a,name)
    header, b1 = probe.output_values_string(extra_info=extra_info, sort_by='BNLI')
    if ident==0: 
        text_output.append(header)
    for tmp in b1: text_output.append(tmp)
    probe_max[name]['plasma'],probe_phase[name]['plasma'] = probe.plot_probe_outputs_vs_phase(color_by='BNLI', ax_amp = ax[0], ax_phase = ax[1], ax_complex = ax[2],min_val = 0, max_val  = 4)

    # print ''
    # tmp = a.project_dict['details']['pickup_coils']
    # if ident==0: header = 'quant '
    # #for q in range(n_simuls):
    # for q in betaN_li_sort:# range(n_simuls):
    #     print '###### {} #####'.format(name)
    #     print 'R={:.3f}, Z={:.3f}m, l_probe={:.3f}m, inc={:.3f}rad, pol={}, betaN/li={Bnli}'.format(*[tmp[i][ind] for i in ['Rprobe', 'Zprobe', 'lprobe','tprobe','probe_type']], Bnli = betaN_li[q]) 
    #     cur_line = '{} {:.4e} {:.3f} '.format(name, betaN_li[q], PMULT[q])
    #     header += 'betaN/li PMULT '
    #     for field in field_list:
    #         for ul in ul_list:
    #             print '{}_{}='.format(field, ul), probe.raw_data['{}_probe_{}'.format(field, ul)][q]
    #             tmp_data = probe.raw_data['{}_probe_{}'.format(field, ul)][q]
    #             cur_line += '{:.6e} {:.6e} '.format(float(np.real(tmp_data)), float(np.imag(tmp_data)))
    #             if q==0 and ident==0: 
    #                 header += '{}_{}_real {}_{}_imag '.format(field, ul, field, ul)
    #     if ident==0 and q==0: 
    #         text_output.append(header.rstrip(' '))
    #     print header
    #     text_output.append(cur_line.rstrip(' '))
    # for q in betaN_li_sort:# range(n_simuls):
    # #for q in range(n_simuls):
    #     colorVal = cmap_cycle(betaN_li[q])
    #     pu, pl= [probe.raw_data['plasma_probe_upper'][q], probe.raw_data['plasma_probe_lower'][q]]
    #     lab = name if q==0 else None
    #     tmp_phases = np.linspace(0,360,100)
    #     tmp_vals_abs = np.abs(pu + pl * np.exp(1j*np.deg2rad(tmp_phases)))
    #     #ax[0].plot(tmp_phases, tmp_vals_abs,label=lab, color = clr_list[q], marker=marker)
    #     ax[0].plot(tmp_phases, tmp_vals_abs,label=lab, color = colorVal, marker=marker)
    #     tmp_vals_ang = np.angle(pu + pl * np.exp(1j*np.deg2rad(tmp_phases)))
    #     tmp_vals2 = pu + pl * np.exp(1j*np.deg2rad(tmp_phases))
    #     #ax2[ident].plot(np.real(tmp_vals2), np.imag(tmp_vals2), color=clr_list[q])
    #     ax[2].plot(np.real(tmp_vals2), np.imag(tmp_vals2), color=colorVal)
    #     #ax[2].plot(np.real(tmp_vals2[0]), np.imag(tmp_vals2[0]), marker = 'o', color=clr_list[q])
    #     ax[2].plot(np.real(tmp_vals2[0]), np.imag(tmp_vals2[0]), marker = 'o', color=colorVal)
    #     #ax[1].plot(tmp_phases, np.rad2deg(tmp_vals_ang)%360,label=lab,color = clr_list[q], marker=marker)
    #     ax[1].plot(tmp_phases, np.rad2deg(tmp_vals_ang)%360,label=lab,color = colorVal, marker=marker)
    #     probe_max[name]['plasma'].append(max(np.abs(tmp_vals_abs)))
    #     probe_phase[name]['plasma'].append(tmp_phases[np.argmax(np.abs(tmp_vals_abs))])

leg = ax[1].legend(loc = 'best')
leg.draw_frame(False)
#ax[1].
max_val_list = []
polar_plots = [ax_tmp[-1] for ax_tmp in ax_overall]
for i in polar_plots:
    xlim = i.get_xlim()
    ylim = i.get_ylim()
    max_val_list.append(np.max(np.abs(np.array([xlim,ylim]))))
max_val = np.max(max_val_list)
for i in polar_plots:
    #i.set_aspect('equal', 'datalim')
    i.set_xlim([-max_val,max_val])
    i.set_ylim([-max_val,max_val])
    i.grid()
    i.set_xlabel('real')
    i.set_xlabel('imag')

fig.suptitle(file_name,fontsize=10)
ax_overall[0][0].set_ylabel('Magnitude G/kA')
ax_overall[0][1].set_ylabel('Phase deg')
ax_overall[0][2].set_ylabel('imag')
ax_overall[0][2].set_ylabel('real')
ax_overall[1][2].set_ylabel('real')
ax_overall[0][1].set_xlabel('Phasing deg')
ax_overall[1][1].set_xlabel('Phasing deg')

fig.canvas.draw();fig.show()

fig20, ax20 = pt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)
for tmp_name in names:
    z = np.array(probe_phase[tmp_name]['plasma'])
    z[z>200]-=360
    tmp_arg_sort = np.argsort(betaN_li)
    #ax20[0].plot(betaN_li[tmp_arg_sort], z[tmp_arg_sort],'-o', label = tmp_name)
    ax20[0].plot(betaN_li[tmp_arg_sort], z,'-o', label = tmp_name)
    leg = ax20[0].legend(loc = 'best')
    leg.draw_frame(False)
    z = np.array(probe_max[tmp_name]['plasma'])
    #ax20[1].plot(betaN_li[tmp_arg_sort], z[tmp_arg_sort],'-o', label = tmp_name)
    ax20[1].plot(betaN_li[tmp_arg_sort], z,'-o', label = tmp_name)
fig20.suptitle(file_name,fontsize=10)
fig20.canvas.draw();fig20.show()

#Print out and plot the res metric values
m_list = [10,8]
#m = 10

print ''
for m in m_list:
    header, b1 =  dBres.output_values_string(m, extra_info=extra_info, sort_by='BNLI')
    for tmp in b1: text_output.append(tmp)

header, b1 =  dBkink.output_values_string(extra_info=extra_info, sort_by='BNLI')
for tmp in b1: text_output.append(tmp)

    # min_loc = np.argmin(np.abs(np.array(dBres.raw_data['res_m_vals'][0])-m))
    # for q in betaN_li_sort:# range(n_simuls):
    # #for q in range(n_simuls):
    #     print '###### {} #####'.format('res_metric')
    #     pitch_res_title = 'Pitch Resonant : m = {}, q = {}, $\psi_N$ = {:.3f}, betaN/li={Bnli}'.format(dBres.raw_data['res_m_vals'][q][min_loc],  dBres.raw_data['res_q_vals'][q][min_loc], dBres.raw_data['res_s_vals'][q][min_loc]**2, Bnli = betaN_li[q])
    #     print pitch_res_title
    #     cur_line = 'm{}q{} {:.4e} {:.3f} '.format(int(dBres.raw_data['res_m_vals'][q][min_loc]),int(dBres.raw_data['res_q_vals'][q][min_loc]),betaN_li[q],PMULT[q])

    #     for field in field_list:
    #         for ul in ul_list:
    #             print '{}_{}='.format(field, ul), dBres.raw_data['{}_res_{}'.format(field, ul)][q][min_loc]
    #             tmp_data = dBres.raw_data['{}_res_{}'.format(field, ul)][q][min_loc]
    #             cur_line += '{:.6e} {:.6e} '.format(float(np.real(tmp_data)), float(np.imag(tmp_data)))
    #     text_output.append(cur_line.rstrip(' '))

#for i in range(len(text_output)):
#    text_output[i] = text_output[i] + '\n'

#Write the text output to a file
with file('{}.txt'.format(file_name.split('/')[-1].rstrip('post_processing_PEST.pickle')),'w') as filehandle:
    filehandle.writelines(text_output)

field = 'vacuum'
field = 'total'

#This is for plotting what happens with the vac metric
fig, ax = pt.subplots()
for q in range(n_simuls):
    u, l = dBres.raw_data['{}_res_{}'.format(field, 'upper')][q][min_loc], dBres.raw_data['{}_res_{}'.format(field, 'lower')][q][min_loc]
    lab = 'res vac' if q==0 else None
    ax.plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))), label = lab, color=clr_list[q], marker = 'x')
fig.canvas.draw();fig.show()

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

print ''
for q in range(n_simuls):
    print '###### {} #####'.format('rfa_metric')
    rfa_title = 'RFA : m = {}, $\psi_N$ = {:.3f}, betaN/li={Bnli}'.format(dBkink.raw_data['plasma_max_mode_list_upper'][q], s_surface, Bnli = betaN_li[q])
    print rfa_title
    #Print out and plot the rfa metric
    for field in ['plasma','vacuum','total']:
        for ul in ['upper','lower']:
            print '{}_{} ='.format(field, ul), dBkink.raw_data['{}_kink_fixed_harm_{}'.format(field, ul)][q]
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
