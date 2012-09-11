import pickle, time
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata
import matplotlib.cm as cm
from Post_Proc_Funcs import *

name = 'test_output2.pickle'
name = ['resonant_phasing_0.pickle','resonant_phasing_0-8.pickle','resonant_phasing_0-9.pickle','resonant_phasing_0-95.pickle']
name = ['0','0-8','0-9','0-95']
fig = pt.figure()


for name_loc in range(0,len(name)):
    file_name = 'resonant_phasing_'+name[name_loc]+'.pickle'
    project_dictionary = pickle.load(open(file_name))

    q95_list = []
    phasing_list = []
    res_amp = []

    phasing_list = num.arange(0,-360,-10)
    q95_list = num.arange(2,7,0.2)
    phasing_grid, q95_grid = num.meshgrid(phasing_list,q95_list)
    q95_list = []
    phasing_list = []
    res_amp = []

    q95_list = project_dictionary.keys()
    q95_list.sort()

    for iii in range(0, len(q95_list)):
        current = q95_list[iii]
        current_list = project_dictionary[current]
        for i in range(0, len(current_list)):
            #q95_list.append(current)
            phasing_list.append(current_list[i][0])
            res_amp.append(current_list[i][1])

    #q95_array=num.array(q95_list)
    phasing_array = num.array(phasing_list)
    res_array = num.array(res_amp)

    new_size = len(project_dictionary.keys()),len(current_list)
    #q95_array.resize(new_size)
    phasing_array.resize(new_size)
    res_array.resize(new_size)

    print name_loc
    ax = fig.add_subplot(2,2,name_loc+1)
    #mk,ss,BnPEST=increase_grid(self.mk.flatten(),self.ss.flatten(),abs(self.BnPEST),number=200)
    print q95_grid.shape
    print phasing_array.shape
    print res_array.shape
    color_ax = ax.pcolor(q95_grid,phasing_array,res_array,cmap='hot')
    for i in range(0,-360,-60):
        ax.plot([2,7],[i,i],'b-')
    pt.colorbar(color_ax)
    ax.set_xlabel('q95')
    ax.set_ylabel('I-coil phasing')
    ax.set_title(r'Resonant forcing, n=2, Beta_n=1.5, from $\sqrt{\Psi_p}$='+name[name_loc], fontsize = 12)
    ax.set_xlim([2,6.8])
    ax.set_ylim([0, -350])


fig.canvas.draw()
fig.show()
