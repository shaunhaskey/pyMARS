import numpy as np
import matplotlib.pyplot as plt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
HOME = os.environ['HOME']


phasing = 0; n = 3; phase_machine_ntor = 0; s_surface = 0.97; fixed_harmonic_offset = 3
fixed_harmonic = 11
reference_dB_kink = 'plasma'; reference_offset = [4,0]
#min_val = 3; max_val = 7
min_val = None; max_val = None

#sort_name = 'betaN_li'
#sort_name = 'Q95'
sort_name = 'Q95'
project_name = 'shot157308_04200_q95_scan_raffi_n3_jan2016'
#project_name = 'shot161409_05300_q95_scan_raffi_n3_jan2016'
file_name = HOME + '/NAMP_datafiles/mars/{}/{}_post_processing_PEST.pickle'.format(project_name, project_name)
print file_name

#Names of the probes that we are interested in
#names = [' 66M', 'Inner_pol']
#names = [' 66M', ' MPID1A', ' MPID1B']
names = ['66M', 'MPID1A',]

#Generate figure for the probe plots
#fig, ax_overall = gen_func.make_figure_share_xy_row(3, len(names))
print file_name

probe_max = {}; probe_phase = {}; probe_names = names

a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic_offset, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
s_bounds = [0.9,1]
dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean', s_bounds=s_bounds)
dBkink = dBres_dBkink.dBkink_calculations(a, fixed_harmonic = fixed_harmonic)
probe_66M = dBres_dBkink.magnetic_probe(a, '66M')
probe_MPID1A = dBres_dBkink.magnetic_probe(a, 'MPID1A')


fig, ax2 = plt.subplots(nrows = 2, ncols =2, sharex = True, sharey = True)
ax_flat = ax2.flatten()
for ax,field in zip(ax_flat,['total','plasma','vacuum']):
    ax.set_title('dBkink {}'.format(field))
    im = dBkink.plot_phasing_scan('Q95',field=field,ax=ax,clim=[0,2], plot_ridge=True, n_contours=7)
    #im.set_clim([0,2.0])
    cbar = plt.colorbar(im, ax=ax)

for i in ax2[:,0]:i.set_ylabel('Phasing')
for i in ax2[-1,:]:i.set_xlabel('Q95')
fig.canvas.draw();fig.show()

    
field = 'total'
field = 'vacuum'
field = 'plasma'

clim_mult = 1.
fig_dBres, ax_dBres = plt.subplots(nrows = 3, sharex = True, sharey = True)
for field,clim,ax_cur in zip(['total','vacuum','plasma'], [[0,1.5*3/15], [0,1.5*1.5], [0,1.5*1.5]], ax_dBres):
    im_res = dBres.plot_phasing_scan('Q95',field=field,ax=ax_cur, plot_ridge = True, n_contours = 7)
    im_res.set_clim(clim)
    ax_cur.set_title('dBres {} s_bounds:{}'.format(field, s_bounds))
    cbar_res = plt.colorbar(im_res, ax=ax_cur)
for i in ax_dBres:i.set_ylabel('Phasing (deg)')
ax_dBres[-1].set_xlabel('q95')
fig_dBres.suptitle(project_name)
fig_dBres.canvas.draw();fig_dBres.show()

fig, ax = plt.subplots(nrows = 2, ncols =2, sharex = True, sharey = True)
ax_flat = ax.flatten()
im_res = dBres.plot_phasing_scan('Q95',field=field,ax=ax_flat[0])
if field=='total':
    clim = [0,1.5*3/25]
elif field=='plasma':
    clim = [0,1.5*3]
elif field=='vacuum':
    clim = [0,1.5*3]

im_res.set_clim()
cbar_res = plt.colorbar(im_res, ax=ax_flat[0])
ax_flat[0].set_title('dBres {} s_bounds:{}'.format(field, s_bounds))
im_kink = dBkink.plot_phasing_scan('Q95',field=field,ax=ax_flat[1])
im_kink.set_clim([0,2.5])
cbar_kink = plt.colorbar(im_kink, ax=ax_flat[1])
ax_flat[1].set_title('dBkink {}'.format(field))

im_66M = probe_66M.plot_phasing_scan('Q95',field=field,ax=ax_flat[2])
im_66M.set_clim([0,2.5])
cbar_kink = plt.colorbar(im_66M, ax=ax_flat[2])
ax_flat[2].set_title('66M {}'.format(field))

im_MPID1A = probe_MPID1A.plot_phasing_scan('Q95',field=field,ax=ax_flat[3])
im_MPID1A.set_clim([0,.5])
cbar_kink = plt.colorbar(im_MPID1A, ax=ax_flat[3])
ax_flat[3].set_title('MPID1A {}'.format(field))


for i in ax[:,0]:i.set_ylabel('Phasing')
for i in ax[-1,:]:i.set_xlabel('Q95')
fig.canvas.draw();fig.show()



1/0
n_simuls = len(dBres.raw_data['res_m_vals'])
betaN_li = np.array(dBres.parent.raw_data['BETAN'])/ np.array(dBres.parent.raw_data['LI'])
q95 = np.array(dBres.parent.raw_data['Q95'])
shot_time = np.array(dBres.parent.raw_data['shot_time'])
betaN_li_sort = np.argsort(betaN_li)
q95_sort = np.argsort(q95)
shot_time_sort = np.argsort(shot_time)

sort_quant = q95
sort_quant = betaN_li
sort_quant = shot_time
grad = np.max(sort_quant) - np.min(sort_quant)
off = np.min(sort_quant)
clr_list = ['{:.2f}'.format(0.9/grad * (q95[i] - off)) for i in range(n_simuls)]

#Get the outputs for the probes, and plot the data
text_output = []
field_list = ['plasma','vacuum','total']
ul_list = ['upper','lower']
extra_info = ['PMULT','BNLI','Q95','QMAX','shot_time']
sort_by = sort_name
# Go through each of the probes
for ident, (name,ax) in enumerate(zip(names,ax_overall)):
    ax[0].set_title(name)
    probe_max[name] = {'vacuum':[],'plasma':[],'total':[]}
    probe_phase[name] = {'vacuum':[],'plasma':[],'total':[]}

    ind = a.project_dict['details']['pickup_coils']['probe'].index(name)
    probe = dBres_dBkink.magnetic_probe(a, name)
    header, b1 = probe.output_values_string(extra_info=extra_info, sort_by=sort_by)
    if ident==0:
        text_output.append(header)
    for tmp in b1: text_output.append(tmp)
    probe_max[name]['plasma'],probe_phase[name]['plasma'] = probe.plot_probe_outputs_vs_phase(color_by=sort_by, ax_amp = ax[0], ax_phase = ax[1], ax_complex = ax[2], min_val = min_val, max_val  = max_val)

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
    print 'BLAH', m
    header, b1 =  dBres.output_values_string(m, extra_info=extra_info, sort_by=sort_by)
    #print b1
    for tmp in b1: text_output.append(tmp)

#Text output for the RFA metric with fixed harmonic
header, b1 =  dBkink.output_values_string(extra_info=extra_info, sort_by=sort_by, fixed_harm = True)
for tmp in b1: text_output.append(tmp)

#Write the text output to a file
with file('{}.txt'.format(file_name.split('/')[-1].replace('post_processing_PEST.pickle','')),'w') as filehandle:
    filehandle.writelines(text_output)

#Output the x-point stuff in a separate file, and produce a plot

fig20, ax20 = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)
for tmp_name in names:
    z = np.array(probe_phase[tmp_name]['plasma'])
    z[z>200]-=360
    tmp_arg_sort = np.argsort(sort_quant)
    #ax20[0].plot(betaN_li[tmp_arg_sort], z[tmp_arg_sort],'-o', label = tmp_name)
    #ax20[0].plot(betaN_li[tmp_arg_sort], z,'-o', label = tmp_name)
    ax20[0].plot(sort_quant[tmp_arg_sort], z,'-o', label = tmp_name)
    leg = ax20[0].legend(loc = 'best')
    leg.draw_frame(False)
    z = np.array(probe_max[tmp_name]['plasma'])
    #ax20[1].plot(betaN_li[tmp_arg_sort], z[tmp_arg_sort],'-o', label = tmp_name)
    #ax20[1].plot(betaN_li[tmp_arg_sort], z,'-o', label = tmp_name)
    ax20[1].plot(sort_quant[tmp_arg_sort], z,'o', label = tmp_name)
fig20.suptitle(file_name,fontsize=10)
fig20.canvas.draw();fig20.show()

fig_met, ax_met = plt.subplots(ncols = 3, sharex = True)
#min_val = None; max_val = None
field = 'total'; fixed_harm = True
dBkink.plot_rfa_outputs_vs_phase(color_by=sort_by, ax = ax_met[0], min_val = min_val, max_val  = max_val, marker = '.', field = field, fixed_harm = fixed_harm)
ax_met[0].set_title('RFA {} {}'.format(field, fixed_harm))
dBres.plot_res_outputs_vs_phase(10, color_by=sort_by, ax = ax_met[1], min_val = min_val, max_val  = max_val, marker = '.', field = field)
ax_met[1].set_title('res {}'.format(field))
#fig25,ax25 = plt.subplots()
data_out = []; disp_txt = []
header = 'quant ' + ' '.join(extra_info) + ' '

for i in range(0,360,45):header+= '{} '.format(i) 
disp_txt.append(header.rstrip(' ')+'\n')
for i in range(0,360,45):
    x_point = dBres_dBkink.x_point_displacement_calcs(a, i)
    data_out.append(x_point.raw_data['plasma_dispx_'])
    #ax25.plot(x_point.parent.raw_data['BETAN'], x_point.raw_data['plasma_dispx_'],'o',color='{:.2f}'.format(float(i)/360.))
data_out = np.array(data_out)
color_by = sort_by
for q in betaN_li_sort:# range(n_simuls):
#for i in range(data_out.shape[1]):

    cmap_cycle = gen_func.new_color_cycle(np.min(sort_quant), np.max(sort_quant))
    colorVal = cmap_cycle(x_point.parent.raw_data[color_by][q])
    cur_line = 'xptdisp ' + ' '.join(['{:.4e}'.format(x_point.parent.raw_data[i][q]) for i in extra_info]) + ' '
    #cur_list = 'xptdisp {:.4f} {:.4f} '.format(x_point.parent.raw_data['BETAN'][q]/x_point.parent.raw_data['LI'][q],x_point.parent.raw_data['PMULT'][q])
    cur_line += ' '.join(['{:.4e}'.format(j) for j in data_out[:,q]])
    disp_txt.append(cur_line.rstrip(' ')+'\n')
    ax_met[2].plot(range(0,360,45), data_out[:,q],color=colorVal)
    ax_met[2].set_title('Displacement Metric')
with file('{}_disp.txt'.format(file_name.split('/')[-1].replace('post_processing_PEST.pickle','')),'w') as filehandle:
    filehandle.writelines(disp_txt)
fig_met.suptitle(file_name,fontsize=10)
fig_met.canvas.draw();fig_met.show()
