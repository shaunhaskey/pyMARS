import pickle, time
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata
import matplotlib.cm as cm
from Post_Proc_Funcs import *


def return_resonant_values(project_dict):
    q95_list = [];Bn_Div_Li_list = [];resonant_list = [];passes = 0;fails = 0
    #iii = 1
    for jjj in project_dict['sims'].keys():
        try:
            if ((project_dict['sims'][jjj]['QMULT'] != 0.81) or (project_dict['sims'][jjj]['PMULT'] !=0.825)):
                resonant_list.append(project_dict['sims'][jjj]['resonant'][0][1])
                q95_list.append(project_dict['sims'][jjj]['Q95'])
                Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
                passes+=1
                #print probe[iii], coil1[-1], 'pmult', project_dict['sims'][jjj]['PMULT'], 'q95:', q95_list[-1], 'Bn/Li', Bn_Div_Li_list[-1]
        except:
            fails+=1
            del project_dict['sims'][jjj]
    print 'pass : %d, fails : %d'%(passes, fails)

    q95_array = num.array(q95_list)
    print q95_array.shape
    Bn_Div_Li_array = num.array(Bn_Div_Li_list)
    print Bn_Div_Li_array.shape
    resonant_array = num.array(resonant_list)
    print resonant_array.shape

    return project_dict, q95_array, Bn_Div_Li_array, resonant_array

def return_grid_data(q95_values, Bn_Div_Li_values,q95_array, Bn_Div_Li_array, resonant_array):
    xnew = num.linspace(q95_values[0], q95_values[1], 100)
    ynew = num.linspace(Bn_Div_Li_values[0], Bn_Div_Li_values[1],100)
    resonant_grid_data = griddata(q95_array, Bn_Div_Li_array, resonant_array, xnew, ynew, interp = 'linear')
    return resonant_grid_data

name = 'test_output.pickle'
project_dict = pickle.load(open(name))
print len(project_dict['sims'].keys())
project_dict, q95_array, Bn_Div_Li_array, resonant_array = return_resonant_values(project_dict)

q95_values =[2.,7.]
Bn_Div_Li_values = [0.75,3.]
#Bn_Div_Li_values = [0,3.]

resonant_grid_data = return_grid_data(q95_values, Bn_Div_Li_values, q95_array, Bn_Div_Li_array, resonant_array)

fig = pt.figure()
ax = fig.add_subplot(111)
list_images=[]
list_images.append(ax.imshow(resonant_grid_data,extent=[q95_values[0], q95_values[1], Bn_Div_Li_values[0], Bn_Div_Li_values[1]], cmap = cm.jet, origin='lower'))
#list_images[-1].set_clim(clim_list[iii])
cbar = fig.colorbar(list_images[-1])
cbar.set_label('|B| G x pol_flux /kA')
ax.plot(q95_array, Bn_Div_Li_array,'k,')
ax.set_ylabel(r'$\beta_N/L_i$',fontsize=20)
ax.set_xlabel('q95')
ax.set_title('Resonant forcing, 0deg I-coil phasing\nn=2',fontsize=14)
ax.set_xlim(q95_values)#([2.1,6.8])
ax.set_ylim(Bn_Div_Li_values)#([0.75,3])
fig.canvas.draw()
fig.show()

fig = pt.figure()
ax = fig.add_subplot(111)
q95_grid, Bn_grid = num.meshgrid(num.r_[q95_values[0]:q95_values[1]:100j],num.r_[Bn_Div_Li_values[0]:Bn_Div_Li_values[1]:100j])
color_ax = ax.pcolor(q95_grid,Bn_grid,resonant_grid_data,cmap=cm.jet)
ax.plot(q95_array, Bn_Div_Li_array,'k,')
ax.set_xlim(q95_values)#([2.1,6.8])
ax.set_ylim(Bn_Div_Li_values)#([0.75,3])
ax.set_title('Resonant forcing, 0deg I-coil phasing\nn=2',fontsize=14)
ax.set_ylabel(r'$\beta_N/L_i$',fontsize=20)
ax.set_xlabel('q95')

cbar = pt.colorbar(color_ax)
cbar.set_label('Wb / kA')
fig.canvas.draw()
fig.show()

#res_grid_data, interp_data_angle = return_grid_data(q95_values, Bn_Div_Li_values,q95_array, Bn_Div_Li_array, coil1_angle_array, coil1_abs_array)

'''
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

fig = pt.figure()
ax = fig.add_subplot(111)
#mk,ss,BnPEST=increase_grid(self.mk.flatten(),self.ss.flatten(),abs(self.BnPEST),number=200)

color_ax = ax.pcolor(q95_grid,phasing_array,res_array,cmap='hot')
pt.colorbar(color_ax)
ax.set_xlabel('q95')
ax.set_ylabel('I-coil phasing')
ax.set_title('Resonant forcing vs q95 and I-coil phasing\nn=2, Beta_n = 1.5')
ax.set_xlim([2,6.8])
ax.set_ylim([0, -350])
fig.canvas.draw()
fig.show()
'''
