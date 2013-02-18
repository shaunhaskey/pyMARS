import pickle
import numpy as np
import matplotlib.pyplot as pt
import results_class
import RZfuncs

def plot_data(ax, psi, data_object, data_object2 = None, label = '', plot_txt_max = 1):
    mk = data_object.mk
    ss = data_object.ss
    ss_plas_edge = np.argmin(np.abs(ss-1.0))
    if data_object2 == None:
        BnPEST = data_object.BnPEST
    else:
        BnPEST = data_object.BnPEST + data_object2.BnPEST

    qn, sq, q, s, mq = RZfuncs.return_q_profile(mk,file_name='PROFEQ.OUT', n=2)
    #tmp_SURFMN_loc = np.argmin(np.abs(ydat[0,:]-tmp_psi))
    tmp_MARSF_loc = np.argmin(np.abs(ss[:ss_plas_edge] - psi))
    q_val = q.flatten()[tmp_MARSF_loc]
    print mk.shape, np.abs(BnPEST[tmp_MARSF_loc, : ]).shape
    ax.plot(mk.flatten(), np.abs(BnPEST[tmp_MARSF_loc, :]), 'x-', label=label+',q=%.1f'%(q_val))
    tmp = np.argmax(np.abs(BnPEST[tmp_MARSF_loc, :]).flatten())
    if plot_txt_max:
        ax.text(mk.flatten()[tmp],np.abs(BnPEST[tmp_MARSF_loc, tmp]), label, fontsize=8)

psi = np.sqrt(0.95)
fig, ax = pt.subplots(nrows = 2, sharex = 1)
target_q95 = 3.5
I0EXP = 826
facn = 1.
file_name = '/home/srh112/NAMP_datafiles/mars/equal_spacing/equal_spacing_post_processing_PEST.pickle'
file_name = '/u/haskeysr/mars/equal_spacing/equal_spacing_post_processing_PEST.pickle'
project_dict = pickle.load(file(file_name,'r'))
key_list = np.array(project_dict['sims'].keys())
q95_list = []; Bn_list = []

for i in project_dict['sims'].keys():
    q95_list.append(project_dict['sims'][i]['Q95'])
    Bn_list.append(project_dict['sims'][i]['BETAN'])

q95_array = np.array(q95_list)
Bn_array = np.array(Bn_list)

tmp_loc = np.argmin(np.abs(q95_array - target_q95))
target_q95 = q95_array[tmp_loc]
relevant_keys = key_list[q95_array==target_q95]

tmp =  1
for j in range(0,len(relevant_keys),2):
    i = relevant_keys[j]
    print project_dict['sims'][i]['Q95'], project_dict['sims'][i]['BETAN']
    dir1 = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
    dir2 = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']
    dir3 = project_dict['sims'][i]['dir_dict']['mars_upper_vacuum_dir']
    dir4 = project_dict['sims'][i]['dir_dict']['mars_lower_vacuum_dir']

    data_object_upper = results_class.data(dir1, I0EXP=I0EXP)
    data_object_upper.get_PEST(facn = facn)
    data_object_lower = results_class.data(dir2, I0EXP=I0EXP)
    data_object_lower.get_PEST(facn = facn)
    plot_data(ax[0], psi, data_object_upper, data_object2 = data_object_lower, label = '%.1f'%(project_dict['sims'][i]['BETAN']))

    data_object_upper = results_class.data(dir3, I0EXP=I0EXP)
    data_object_upper.get_PEST(facn = facn)
    data_object_lower = results_class.data(dir4, I0EXP=I0EXP)
    data_object_lower.get_PEST(facn = facn)
    plot_data(ax[1], psi, data_object_upper, data_object2 = data_object_lower, label = '%.1f'%(project_dict['sims'][i]['BETAN']), plot_txt_max = 0)
    tmp+=1


betaN_list = []
for j in range(0,len(relevant_keys)):
    i = relevant_keys[j]
    betaN_list.append(project_dict['sims'][i]['BETAN'])

print '################ max betaN'
print betaN_list
print np.max(betaN_list)

ax[0].legend(loc='best',prop={'size':8})
ax[0].set_title(r'Plasma + Vacuum response n = 2, $\Psi_N=%.2f$'%(psi**2))
ax[1].set_xlabel('m')
ax[0].set_ylabel('Amplitude')
ax[1].set_ylabel('Amplitude')
fig.canvas.draw(); fig.show()


'''
dir_list = ['/home/srh112/NAMP_datafiles/mars/146382_thetac_003/qmult1.000/exp1.000/marsrun/RUNrfa.vac', '/home/srh112/NAMP_datafiles/mars/146382_thetac_006/qmult1.000/exp1.000/marsrun/RUNrfa.vac', '/home/srh112/NAMP_datafiles/mars/146382_thetac_010/qmult1.000/exp1.000/marsrun/RUNrfa.vac', '/home/srh112/NAMP_datafiles/mars/146382_thetac_020/qmult1.000/exp1.000/marsrun/RUNrfa.vac']

for i in dir_list:
    data_object = results_class.data(i, I0EXP=I0EXP)
    data_object.get_PEST(facn = facn)
    plot_data(ax, data_object, psi,label = '%d'%(tmp))
    tmp+=1
ax.legend(loc='best')

fig.canvas.draw(); fig.show()
'''
