import pickle, time
import numpy as np
import matplotlib.pyplot as pt
#from Post_Proc_Funcs import *

name = 'test_output2.pickle'
name = ['resonant_phasing_0.pickle','resonant_phasing_0-8.pickle','resonant_phasing_0-9.pickle','resonant_phasing_0-95.pickle']
name = ['0','0-8','0-9','0-95']
#fig = pt.figure()
base_dir = '/home/srh112/NAMP_datafiles/'

fig, ax = pt.subplots(nrows = 4, sharex =1 , sharey = 1)
ax = ax.flatten()
for name_loc in range(0,len(name)):
    file_name = base_dir + 'resonant_phasing_'+name[name_loc]+'.pickle'
    project_dictionary = pickle.load(open(file_name))

    phasing_list = np.arange(0,-360,-10)
    phasing_list = []; res_amp = []; q95_plot_list = []


    q95_list = project_dictionary.keys()
    q95_list.sort()
    for iii in range(0, len(q95_list)):
        current = q95_list[iii]
        current_list = project_dictionary[current]
        for i in range(0, len(current_list)):
            phasing_list.append(current_list[i][0])
            res_amp.append(current_list[i][1])
            q95_plot_list.append(current)

    phasing_array = np.array(phasing_list)
    res_array = np.array(res_amp)
    new_size = len(project_dictionary.keys()),len(current_list)
    q95_plot_array = np.array(q95_plot_list)
    q95_plot_array.resize(new_size)
    #phasing_grid, q95_grid = np.meshgrid(phasing_array,q95_list)
    phasing_array.resize(new_size)
    res_array.resize(new_size)


    print name_loc
    #ax = fig.add_subplot(2,2,name_loc+1)
    #mk,ss,BnPEST=increase_grid(self.mk.flatten(),self.ss.flatten(),abs(self.BnPEST),number=200)
    #print q95_grid.shape
    #print phasing_array.shape
    #print res_array.shape
    color_ax = ax[name_loc].pcolor(q95_plot_array,phasing_array,res_array,cmap='hot')
    #pt.colorbar(color_ax, ax = ax[name_loc])
    for i in range(0,-360,-60):
        ax[name_loc].plot([2,7],[i,i],'b-')
    ax[name_loc].set_xlabel('q95')
    ax[name_loc].set_ylabel('I-coil phasing')
    ax[name_loc].set_title(r'Resonant forcing, n=2, Beta_n=1.5, from $\sqrt{\Psi_p}$='+name[name_loc], fontsize = 12)
    ax[name_loc].set_xlim([2,6.8])
    ax[name_loc].set_ylim([0, -350])
    if name[name_loc]=='0':
        single_fig, single_ax = pt.subplots()
        #mk,ss,BnPEST=increase_grid(self.mk.flatten(),self.ss.flatten(),abs(self.BnPEST),number=200)
        #print q95_grid.shape
        #print phasing_array.shape
        #print res_array.shape
        color_ax_single = single_ax.pcolor(q95_plot_array, phasing_array, res_array,cmap='hot')
        tmp_xaxis = []; tmp_yaxis_min = []; tmp_yaxis_max = []
        for tmp1 in range(0,phasing_array.shape[0]):
            max_loc = np.argmax(np.abs(res_array[tmp1,:]))
            min_loc = np.argmin(np.abs(res_array[tmp1,:]))
            tmp_xaxis.append(q95_plot_array[tmp1,0])
            tmp_yaxis_min.append(phasing_array[tmp1, min_loc])
            tmp_yaxis_max.append(phasing_array[tmp1, max_loc])
            
        single_ax.plot(tmp_xaxis, tmp_yaxis_min, 'bo')
        single_ax.plot(tmp_xaxis, tmp_yaxis_max, 'bx')
        for i in range(0,-360,-60):
            single_ax.plot([2,7],[i,i],'b-')
        pt.colorbar(color_ax_single, ax = single_ax)
        single_ax.set_xlabel(r'$q_{95}$',fontsize=14)
        single_ax.set_ylabel('Phasing (deg)')
        single_ax.set_title(r'Resonant forcing, n=2, Beta_n=1.5, from $\sqrt{\Psi_p}$='+name[name_loc], fontsize = 12)
        single_ax.set_xlim([2,6.8])
        single_ax.set_ylim([0, -350])
        single_fig.canvas.draw(); single_fig.show()

fig.canvas.draw()
fig.show()
