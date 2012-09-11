import pickle, time
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata
import matplotlib.cm as cm
from Post_Proc_Funcs import *

name = '9_project1_new_eq_FEEDI_0_coil_outputs.pickle'
name = '9_project1_new_eq_FEEDI_-240_coil_outputs.pickle'
name = '9_project1_new_eq_FEEDI_-300_coil_outputs.pickle'
name = '9_project1_new_eq_FEEDI_-180_coil_outputs.pickle'
ROTE_value_list = [0, -60, -120, -180, -240, -300]
ROTE_value_list = [0]
ROTE_value_list = [-300]

#name = '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
#name = '9_benchmark2_coil_outputs.pickle'
#name = '9_project_n_3_coil_outputs.pickle'


probe  = [ 'dBp_upper - 67A', 'dBp_mid - 66M', 'dBp_lower - 67B', 'dBr_ext - ESL', 'dBr_mid - ISL','dBr_upper - UISL','dBr_lower  - LISL']
probe2  = [ 'dBp_u', 'dBp_m', 'dBp_l', 'dBr_ext', 'dBr_m','dBr_u','dBr_l']

probe_list = [0,1,2,3,4,5,6]
probe_list = [1]
clim_list = [[0,1.5],[0,0.8],[0,1.5],[0,0.7],[0,2],[0,1.5],[0,2]]
#clim_list = [[0,1.5],[0,0.8],[0,1.5],[0,0.7],[0,2],[0,1.5],[0,2]]
q95_slice_check = 0
Bn_slice_check = 0
calc = 'plasma_response4'
calc2 = 'vacuum_response4'
list_images = []

start = 1

for ROTE_value in ROTE_value_list:
    name = '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
    name = '9_project1_new_eq_FEEDI_-300_ICOIL_FREQ_0_coil_outputs.pickle'
    name = '9_project_n_3_COIL_upper_post_setup.pickle'
    name = '9_project_n_3_COIL_lower_post_setup.pickle'
#    name = '9_project_n_3_coil_outputs.pickle'
    project_dict = pickle.load(open(name))


    #Search for the base case equilibria
    if start == 1:
        initial_eq_serial = return_initial_eq(project_dict)
    start = 0

    for iii in probe_list:#range(0,len(probe)):
        num_count = 1
        project_dict, q95_array, Bn_Div_Li_array, coil1, coil1_abs_array, coil1_angle_array = return_probe_values(project_dict, calc, calc2, iii)

        q95_values =[2.,7.]
        Bn_Div_Li_values = [0.75,3.]

        B1grid_data, interp_data_angle = return_grid_data(q95_values, Bn_Div_Li_values,q95_array, Bn_Div_Li_array, coil1_angle_array, coil1_abs_array)

        #Print check the errors in the reconstruction
        #B1_err = []

    #    for i in range(0,len(q95_array)):
    #        if q95_array[i]<q95max and q95_array[i]>q95min:
    #            if Bn_Div_Li_array[i]<Bn_Div_Li_max and Bn_Div_Li_array[i]>Bn_Div_Li_min:
    #                B1_err.append((newfuncB1(q95_array[i],Bn_Div_Li_array[i])-num.abs(coil1_abs_array[i]))/num.abs(coil1_abs_array[i])*100)

    #    list_new_data = [newvalsB1]
        list_new_data = [B1grid_data]


        fig = pt.figure()
        ax = fig.add_subplot(211)
        list_images.append(ax.imshow(list_new_data[0],extent=[q95_values[0], q95_values[1], Bn_Div_Li_values[0], Bn_Div_Li_values[1]], cmap = cm.jet, origin='lower'))
        list_images[-1].set_clim(clim_list[iii])
        pt.contour(list_new_data[0], [0,0.1, 0.2,0.5,0.8,1.1, 1.5, 2.2],colors = 'k',origin='lower', extent=[q95_values[0], q95_values[1], Bn_Div_Li_values[0], Bn_Div_Li_values[1]])
        cbar = fig.colorbar(list_images[-1])
        cbar.set_label('|B| G/kA')
        ax.plot(q95_array, Bn_Div_Li_array,'k,')

            #        q_list = [3.17,3+3./6,4]
        q_list = [3,4.5,6.5]
        color_list3 = ['blue','black','yellow','red']
        #for arrow_temp in range(0,len(q_list)):
        #    ax.arrow(q_list[arrow_temp],0.8, 0, 2, width=0.05,fc=color_list3[arrow_temp], head_width = 0.2, head_length = 0.2)

            #ax.plot(3.6,2.0,'ko',markersize = 10,markerfacecolor='white')
            #ax.plot(project_dict['sims'][initial_eq_serial]['Q95'],project_dict['sims'][initial_eq_serial]['BETAN']/project_dict['sims'][initial_eq_serial]['LI'], 'k*', markersize = 10, markerfacecolor='white')
        ax.set_ylabel(r'$\beta_N/L_i$',fontsize=20)
        ax.set_title('n=2 ' + str(ROTE_value) + 'deg I-coil phasing ' + probe2[iii],fontsize=14)
        ax.set_title('n=3 upper only ' + probe2[iii],fontsize=16)
        ax.set_title('n=3 lower only ' + probe2[iii],fontsize=16)
        ax.set_xlim(q95_values)#([2.1,6.8])
        ax.set_ylim(Bn_Div_Li_values)#([0.75,3])

        ax2 = fig.add_subplot(212)
        list_images.append(ax2.imshow(interp_data_angle, extent=[q95_values[0], q95_values[1], Bn_Div_Li_values[0], Bn_Div_Li_values[1]], origin='lower'))
        list_images[-1].set_clim([0, 360])
        cbar = fig.colorbar(list_images[-1])
        cbar.set_label('deg')
        ax2.plot(q95_array, Bn_Div_Li_array,'k,')
        #for arrow_temp in range(0,len(q_list)):
        #    ax2.arrow(q_list[arrow_temp],0.8, 0, 2, width=0.05,fc=color_list3[arrow_temp], head_width = 0.2, head_length = 0.2)

            #ax2.plot(3.6,2.0,'ko',markersize = 10, markerfacecolor='white')
        ax2.set_xlabel('q95',fontsize = 16)
        ax2.set_ylabel(r'$\beta_N/L_i$',fontsize=20)
            #        ax2.set_title(str(ROTE_value) + ' ' + probe[iii])
            #        ax2.set_title('phase (deg)', fontsize = 12)
        ax2.set_xlim(q95_values)#([2.1,6.8])
        ax2.set_ylim(Bn_Div_Li_values)#([0.75,3])
        fig.canvas.draw()
        fig.show()

            #        fig = pt.figure()
            #        ax = fig.add_subplot(111)
        q95_range = 0.1
        pt.savefig('%03d.png'%(num_count))
        num_count+=1
    #Bn Slice
    Bn_slice_check = 1
    if Bn_slice_check == 1 :
        Bn_List = [1.2, 2., 2.8]
        plot_Bn_slice(Bn_List, q95_array, Bn_Div_Li_array, coil1_abs_array, coil1_angle_array, ROTE_value, probe)


    #Q95 Slice
    q95_slice_check = 1
    if q95_slice_check ==1:
        plot_q95_slice(q_list, project_dict, ROTE_value, probe, calc, calc2)


'''
        fig = pt.figure()
        ax = fig.add_subplot(211)
        ax.plot(q95_array, Bn_Div_Li_array,'k,')
#        ax.set_xlabel('q95')
        ax.plot(project_dict['sims'][initial_eq_serial]['Q95'],project_dict['sims'][initial_eq_serial]['BETAN']/project_dict['sims'][initial_eq_serial]['LI'], 'kx', markersize = 10, markerfacecolor='white')
        ax.set_ylabel('Bn/Li')
        ax.set_xlabel('q95')
#        ax.set_title(str(ROTE_value) + 'deg ' + probe[iii])
        ax.set_xlim(q95_values)#([2.1,6.8])
        ax.set_ylim(Bn_Div_Li_values)#([0.75,3])

        fig.canvas.draw()
        fig.show()
'''
