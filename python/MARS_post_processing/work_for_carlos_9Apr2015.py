import numpy as np
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
HOME = os.environ['HOME']

#Sort this into more useful script
#Output RFA metric
#Output displacement metric
#Check l-mode
#Ideal cases
phasing = 0; n = 2; phase_machine_ntor = 0; s_surface = 0.94; fixed_harmonic_offset = 3
fixed_harmonic = 11
reference_dB_kink = 'plasma'; reference_offset = [4,0]
sort_name = 'betaN_li'
file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_scan/shot158115_04780_scan_post_processing_PEST.pickle'
file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_scan_bak/shot158115_04780_scan_post_processing_PEST.pickle'
#file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_scan/shot158115_04780_scan_post_processing_PEST.pickle'
#file_name = HOME + '/NAMP_datafiles/mars/shot158103_03796_betaN_ramp_carlos_hicol/shot158103_03796_betaN_ramp_carlos_hicol_post_processing_PEST.pickle'
#file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retest/shot158115_04780_betaN_ramp_retest_post_processing_PEST.pickle'
#file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retest2/shot158115_04780_betaN_ramp_retest2_post_processing_PEST.pickle'
#file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retestV2/shot158115_04780_betaN_ramp_retestV2_post_processing_PEST.pickle'

#These are the the set of simulations
file_name = HOME + '/NAMP_datafiles/mars/shot158103_03796_betaN_ramp_carlos_2/shot158103_03796_betaN_ramp_carlos_2_post_processing_PEST.pickle'
file_name = HOME + '/NAMP_datafiles/mars/shot161198_03550_betaN_ramp_carlos_hicol2/shot161198_03550_betaN_ramp_carlos_hicol2_post_processing_PEST.pickle'
#file_name = HOME + '/NAMP_datafiles/mars/shot161205_03215_betaN_ramp_carlos_lmode2/shot161205_03215_betaN_ramp_carlos_lmode2_post_processing_PEST.pickle'
file_name = HOME + '/NAMP_datafiles/mars/shot158117_03885_betaN_ramp_carlos_q95/shot158117_03885_betaN_ramp_carlos_q95_post_processing_PEST.pickle'
file_name = HOME + '/NAMP_datafiles/mars/shot159346_02120_betaN_ramp_carlos_q95/shot159346_02120_betaN_ramp_carlos_q95_post_processing_PEST.pickle'
file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retest_carlos_prl_case/shot158115_04780_betaN_ramp_retest_carlos_prl_case_post_processing_PEST.pickle'

file_name = HOME + '/NAMP_datafiles/mars/shot158115_04780_betaN_ramp_retest_carlos_prl_caseV2/shot158115_04780_betaN_ramp_retest_carlos_prl_caseV2_post_processing_PEST.pickle'

file_name = HOME + '/NAMP_datafiles/mars/shot158103_03796_betaN_ramp_carlos_2_thetac0-005/shot158103_03796_betaN_ramp_carlos_2_thetac0-005_post_processing_PEST.pickle'
file_name = HOME + '/NAMP_datafiles/mars/shot158103_03796_betaN_ramp_carlos_2_thetac0-009/shot158103_03796_betaN_ramp_carlos_2_thetac0-009_post_processing_PEST.pickle'
file_name = HOME + '/NAMP_datafiles/mars/shot158103_03796_betaN_ramp_carlos_2/shot158103_03796_betaN_ramp_carlos_2_post_processing_PEST.pickle'

#Names of the probes that we are interested in
names = [' 66M', 'Inner_pol']
names = [' 66M', ' MPID1A', ' MPID1B']
names = ['66M', 'MPID1A',]

#Generate figure for the probe plots
fig, ax_overall = gen_func.make_figure_share_xy_row(3, len(names))
print file_name
min_val = 0; max_val = 4

probe_max = {}; probe_phase = {}; probe_names = names

a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic_offset, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean')
dBkink = dBres_dBkink.dBkink_calculations(a, fixed_harmonic = fixed_harmonic)


n_simuls = len(dBres.raw_data['res_m_vals'])
betaN_li = np.array(dBres.parent.raw_data['BETAN'])/ np.array(dBres.parent.raw_data['LI'])
betaN_li_sort = np.argsort(betaN_li)
PMULT = np.array(dBres.parent.raw_data['PMULT'])
#QMULT = np.array(dBres.parent.raw_data['QMULT'])
grad = np.max(betaN_li) - np.min(betaN_li)
off = np.min(betaN_li)
clr_list = ['{:.2f}'.format(0.9/grad * (betaN_li[i] - off)) for i in range(n_simuls)]


#Get the outputs for the probes, and plot the data
text_output = []
field_list = ['plasma','vacuum','total']
ul_list = ['upper','lower']
extra_info = ['PMULT','BNLI']
for ident, (name,ax) in enumerate(zip(names,ax_overall)):
    ax[0].set_title(name)
    probe_max[name] = {'vacuum':[],'plasma':[],'total':[]}
    probe_phase[name] = {'vacuum':[],'plasma':[],'total':[]}

    ind = a.project_dict['details']['pickup_coils']['probe'].index(name)
    probe = dBres_dBkink.magnetic_probe(a,name)
    header, b1 = probe.output_values_string(extra_info=extra_info, sort_by='BNLI')
    if ident==0: 
        text_output.append(header)
    for tmp in b1: text_output.append(tmp)
    probe_max[name]['plasma'],probe_phase[name]['plasma'] = probe.plot_probe_outputs_vs_phase(color_by='BNLI', ax_amp = ax[0], ax_phase = ax[1], ax_complex = ax[2],min_val = min_val, max_val  = max_val)

#cleaning up the plots
leg = ax[1].legend(loc = 'best')
leg.draw_frame(False)
#ax[1].
max_val_list = []
polar_plots = [ax_tmp[-1] for ax_tmp in ax_overall]
for i in polar_plots:
    xlim = i.get_xlim()
    ylim = i.get_ylim()
    max_val_list.append(np.max(np.abs(np.array([xlim,ylim]))))
max_val_tmp = np.max(max_val_list)
for i in polar_plots:
    #i.set_aspect('equal', 'datalim')
    i.set_xlim([-max_val_tmp,max_val_tmp])
    i.set_ylim([-max_val_tmp,max_val_tmp])
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

#Get the text output for the res metric values
m_list = [10,8]
for m in m_list:
    header, b1 =  dBres.output_values_string(m, extra_info=extra_info, sort_by='BNLI')
    for tmp in b1: text_output.append(tmp)

#Text output for the RFA metric with fixed harmonic
header, b1 =  dBkink.output_values_string(extra_info=extra_info, sort_by='BNLI', fixed_harm = True)
for tmp in b1: text_output.append(tmp)

#Write the text output to a file
with file('{}.txt'.format(file_name.split('/')[-1].replace('post_processing_PEST.pickle','')),'w') as filehandle:
    filehandle.writelines(text_output)

#Output the x-point stuff in a separate file, and produce a plot

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


fig_met, ax_met = pt.subplots(ncols = 3, sharex = True)
#min_val = None; max_val = None
field = 'total'; fixed_harm = True
dBkink.plot_rfa_outputs_vs_phase(color_by='BNLI', ax = ax_met[0], min_val = min_val, max_val  = max_val, marker = '.', field = field, fixed_harm = fixed_harm)
ax_met[0].set_title('RFA {} {}'.format(field, fixed_harm))
dBres.plot_res_outputs_vs_phase(10, color_by='BNLI', ax = ax_met[1], min_val = min_val, max_val  = max_val, marker = '.', field = field)
ax_met[1].set_title('res {}'.format(field))
#fig25,ax25 = pt.subplots()
data_out = []; disp_txt = []
header = 'quant ' + ' '.join(extra_info) + ' '

for i in range(0,360,45):header+= '{} '.format(i) 
disp_txt.append(header.rstrip(' ')+'\n')
for i in range(0,360,45):
    x_point = dBres_dBkink.x_point_displacement_calcs(a, i)
    data_out.append(x_point.raw_data['plasma_dispx_'])
    #ax25.plot(x_point.parent.raw_data['BETAN'], x_point.raw_data['plasma_dispx_'],'o',color='{:.2f}'.format(float(i)/360.))
data_out = np.array(data_out)
color_by = 'BNLI'
for q in betaN_li_sort:# range(n_simuls):
#for i in range(data_out.shape[1]):
    cmap_cycle = gen_func.new_color_cycle(min_val, max_val)
    colorVal = cmap_cycle(x_point.parent.raw_data[color_by][q])
    cur_line = 'xptdisp ' + ' '.join(['{:.4e}'.format(x_point.parent.raw_data[i][q]) for i in extra_info]) + ' '
    #cur_list = 'xptdisp {:.4f} {:.4f} '.format(x_point.parent.raw_data['BETAN'][q]/x_point.parent.raw_data['LI'][q],x_point.parent.raw_data['PMULT'][q])
    cur_line += ' '.join(['{:.4e}'.format(j) for j in data_out[:,q]])
    disp_txt.append(cur_line.rstrip(' ')+'\n')
    ax_met[2].plot(range(0,360,45), data_out[:,q],color=colorVal)
    ax_met[2].set_title('Displacement Metric')
with file('{}_disp.txt'.format(file_name.split('/')[-1].replace('post_processing_PEST.pickle','')),'w') as filehandle:
    filehandle.writelines(disp_txt)
#fig25.canvas.draw();fig25.show()
fig_met.suptitle(file_name,fontsize=10)
fig_met.canvas.draw();fig_met.show()

field = 'vacuum'
field = 'total'
