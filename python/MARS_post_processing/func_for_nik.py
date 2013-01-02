import pickle
import numpy as np
import matplotlib.pyplot as pt
import results_class


def plot_data(ax, psi, data_object, data_object2 = None, label = ''):
    mk = data_object.mk
    ss = data_object.ss
    ss_plas_edge = np.argmin(np.abs(ss-1.0))
    if data_object2 == None:
        BnPEST = data_object.BnPEST
    else:
        BnPEST = data_object.BnPEST + data_object2.BnPEST

    #tmp_SURFMN_loc = np.argmin(np.abs(ydat[0,:]-tmp_psi))
    tmp_MARSF_loc = np.argmin(np.abs(ss[:ss_plas_edge] - psi))
    print mk.shape, np.abs(BnPEST[tmp_MARSF_loc, : ]).shape
    ax.plot(mk.flatten(), np.abs(BnPEST[tmp_MARSF_loc, :]), 'x-', label=label)


psi = 0.92
fig, ax = pt.subplots()
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
for i in relevant_keys:
    print project_dict['sims'][i]['Q95'], project_dict['sims'][i]['BETAN']
    dir1 = project_dict['sims'][i]['dir_dict']['mars_upper_plasma_dir']
    dir2 = project_dict['sims'][i]['dir_dict']['mars_lower_plasma_dir']
    data_object_upper = results_class.data(dir1, I0EXP=I0EXP)
    data_object_upper.get_PEST(facn = facn)
    data_object_lower = results_class.data(dir2, I0EXP=I0EXP)
    data_object_lower.get_PEST(facn = facn)

    plot_data(ax, psi, data_object_upper, data_object2 = data_object_lower, label = '%d'%(project_dict['sims'][i]['BETAN']))
    plot_data(ax, data_object, psi,label = '%d'%(project_dict['sims'][i]['BETAN']))
    tmp+=1
ax.legend(loc='best')

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
