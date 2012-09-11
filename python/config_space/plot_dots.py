import pickle
import matplotlib.pyplot as pt

name = '9_project1_new_eq_COIL_lower_post_setup_new.pickle'
name = '4_project1_new_eq_post_fxrun_low_beta.pickle'
project_dict = pickle.load(open(name))

q95_list=[]
Bn_list = []
Bn_Div_Li_list = []
Li_list = []
for jjj in project_dict['sims'].keys():
    q95_list.append(project_dict['sims'][jjj]['Q95'])
    Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
    Bn_list.append(project_dict['sims'][jjj]['BETAN'])
    Li_list.append(project_dict['sims'][jjj]['LI'])

fig = pt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(q95_list,Bn_list,'.')
ax1.set_xlabel('q95')
ax1.set_ylabel('Bn')

ax2.plot(q95_list,Bn_Div_Li_list,'.')
ax2.set_xlabel('q95')
ax2.set_ylabel('Bn/Li')



name = '9_project1_new_eq_COIL_lower_post_setup_new.pickle'
project_dict = pickle.load(open(name))

q95_list=[]
Bn_list = []
Bn_Div_Li_list = []
Li_list = []
for jjj in project_dict['sims'].keys():
    q95_list.append(project_dict['sims'][jjj]['Q95'])
    Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
    Bn_list.append(project_dict['sims'][jjj]['BETAN'])
    Li_list.append(project_dict['sims'][jjj]['LI'])

ax1.plot(q95_list,Bn_list,'rx')
ax1.set_xlabel('q95')
ax1.set_ylabel('Bn')

ax2.plot(q95_list,Bn_Div_Li_list,'rx')
ax2.set_xlabel('q95')
ax2.set_ylabel('Bn/Li')

fig.canvas.draw()
fig.show()
