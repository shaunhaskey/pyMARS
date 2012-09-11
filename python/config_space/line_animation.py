'''
plots the q95 vs beta_n space, including an arrow, and a moving vertical
line so that you can include an animation that explains what is happening in the PEST plots
SH : 11Sept2012
'''

import pickle, time
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata
import matplotlib.cm as cm

name = '9_project1_new_eq_FEEDI_0_coil_outputs.pickle'
name = '9_project1_new_eq_FEEDI_-240_coil_outputs.pickle'
name = '9_project1_new_eq_FEEDI_-300_coil_outputs.pickle'
name = '/home/srh112/NAMP_datafiles/project1_new_eq/9_project1_new_eq_FEEDI_-180_coil_outputs.pickle'
ROTE_value_list = [0, -60, -120, -180, -240, -300]
ROTE_value_list = [0]
#ROTE_value_list = [-300]

#name = '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
#name = '9_benchmark2_coil_outputs.pickle'
#name = '9_project_n_3_coil_outputs.pickle'


probe  = [ 'dBp_upper - 67A', 'dBp_mid - 66M', 'dBp_lower - 67B', 'dBr_ext - ESL', 'dBr_mid - ISL','dBr_upper - UISL','dBr_lower  - LISL']
probe2  = [ 'dBp_u', 'dBp_m', 'dBp_l', 'dBr_ext', 'dBr_m','dBr_u','dBr_l']
#probe_list = [0,1,2,3,4,5,6]
probe_list = [0,1,2,3,4,5,6]
probe_list = [1]
clim_list = [[0,1.5],[0,2],[0,1.5],[0,0.7],[0,2],[0,1.5],[0,2]]
#clim_list = [[0,1.5],[0,0.8],[0,1.5],[0,0.7],[0,2],[0,1.5],[0,2]]
q95_slice_check = 0
Bn_slice_check = 0
calc = 'plasma_response4'
calc2 = 'vacuum_response4'
list_images = []

start = 1
min_distance = 1000
for ROTE_value in ROTE_value_list:
    name = '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
    #name = '9_project1_new_eq_FEEDI_-300_ROTE_1.0_coil_outputs.pickle'

    #name = '9_project_n_3_coil_outputs.pickle'
#    name = '9_project_n_4_FEEDI_0_coil_outputs.pickle'
    name = '/home/srh112/NAMP_datafiles/project_n_3/9_project_n_3_coil_outputs.pickle'
    project_dict = pickle.load(open(name))

    #Search for the base case equilibria
    if start == 1:
        for jjj in project_dict['sims'].keys():
            qmult_curr = project_dict['sims'][jjj]['QMULT']
            pmult_curr = project_dict['sims'][jjj]['PMULT']
            distance = (qmult_curr - 1.)**2 + (pmult_curr -1.)**2
            if distance < min_distance:
                min_distance = distance *1.
                initial_eq_serial = jjj *1
        print 'serial : %d, qmult %.2f, pmult %.2f, q95: %.2f, Bn/Li: %.2f'%(initial_eq_serial, project_dict['sims'][initial_eq_serial]['QMULT'],project_dict['sims'][initial_eq_serial]['PMULT'],project_dict['sims'][initial_eq_serial]['Q95'],project_dict['sims'][initial_eq_serial]['BETAN']/project_dict['sims'][initial_eq_serial]['LI'])

        start = 0
    for iii in probe_list:#range(0,len(probe)):
        q95_list = []
        Bn_Div_Li_list = []
        coil1 = []
        passes = 0
        fails = 0
    #    iii = 1
        for jjj in project_dict['sims'].keys():
            try:
                if ((project_dict['sims'][jjj]['QMULT'] != 0.81) or (project_dict['sims'][jjj]['PMULT'] !=0.825)):
                    coil1.append(project_dict['sims'][jjj][calc][iii]-project_dict['sims'][jjj][calc2][iii]) # Plasma only
                    #coil1.append(project_dict['sims'][jjj][calc][iii]) # Plasma + Vacuum

                    q95_list.append(project_dict['sims'][jjj]['Q95'])
                    Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
                    passes+=1
                    print probe[iii], coil1[-1], 'pmult', project_dict['sims'][jjj]['PMULT'], 'q95:', q95_list[-1], 'Bn/Li', Bn_Div_Li_list[-1]
            except:
                fails+=1
                del project_dict['sims'][jjj]
        print 'pass : %d, fails : %d'%(passes, fails)

        q95_array = num.array(q95_list)
        Bn_Div_Li_array = num.array(Bn_Div_Li_list)
        coil1_abs_array = num.abs(num.array(coil1))
        coil1_angle_array = num.angle(num.array(coil1))*360./2./num.pi

     #   newfuncB1 = interpolate.Rbf(q95_array,Bn_Div_Li_array, coil1_abs_array,function='linear')


        #newfuncB1 = interpolate.interp2d(q95_array,Bn_Div_Li_array, num.abs(B1_val),kind='linear')
        q95_values =[2.,7.]
        Bn_Div_Li_values = [0.75,3.]
#        q95_values =[2.5,4.5] #Matt benchmark
#        Bn_Div_Li_values = [1,3.] #Matt benchmark

#        q95min = 2.#2.5#2.
#        q95max = 7.#4.5#7.#6#3.7
#        Bn_Div_Li_min = 0.75#1.#0.5 # 0.36
#        Bn_Div_Li_max = 3.#2.5 #2.7 

        #xnew, ynew = num.mgrid[q95min:q95max:100j, Bn_Div_Li_min:Bn_Div_Li_max:100j]
        #newvalsB1 = newfuncB1(xnew,ynew)
        xnew = num.linspace(q95_values[0], q95_values[1], 100)
        ynew = num.linspace(Bn_Div_Li_values[0], Bn_Div_Li_values[1],100)

        for i in range(0,len(coil1_angle_array)):
            if coil1_angle_array[i]<0:
                coil1_angle_array[i] += 360.

        B1grid_data = griddata(q95_array, Bn_Div_Li_array, coil1_abs_array, xnew, ynew, interp = 'nn')
        interp_data_angle = griddata(q95_array, Bn_Div_Li_array, coil1_angle_array, xnew, ynew, interp = 'nn')


        B1_err = []


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
        q_list = [4]
        color_list3 = ['blue','black','yellow','red']
        for arrow_temp in range(0,len(q_list)):
            ax.arrow(q_list[arrow_temp],0.8, 0, 2, width=0.05,fc=color_list3[arrow_temp], head_width = 0.2, head_length = 0.2)

        #ax.plot(3.6,2.0,'ko',markersize = 10,markerfacecolor='white')
        #ax.plot(project_dict['sims'][initial_eq_serial]['Q95'],project_dict['sims'][initial_eq_serial]['BETAN']/project_dict['sims'][initial_eq_serial]['LI'], 'k*', markersize = 10, markerfacecolor='white')
        ax.set_ylabel('Bn/Li')
        ax.set_title('n=2 ' + str(ROTE_value) + 'deg I-coil phasing ' + probe2[iii],fontsize=14)
        ax.set_xlim(q95_values)#([2.1,6.8])
        ax.set_ylim(Bn_Div_Li_values)#([0.75,3])

        ax2 = fig.add_subplot(212)
        list_images.append(ax2.imshow(interp_data_angle, extent=[q95_values[0], q95_values[1], Bn_Div_Li_values[0], Bn_Div_Li_values[1]], origin='lower'))
        list_images[-1].set_clim([0, 360])
        cbar = fig.colorbar(list_images[-1])
        cbar.set_label('deg')
        ax2.plot(q95_array, Bn_Div_Li_array,'k,')
        for arrow_temp in range(0,len(q_list)):
            ax2.arrow(q_list[arrow_temp],0.8, 0, 2, width=0.05,fc=color_list3[arrow_temp], head_width = 0.2, head_length = 0.2)

        #ax2.plot(3.6,2.0,'ko',markersize = 10, markerfacecolor='white')
        ax2.set_xlabel('q95')
        ax2.set_ylabel('Bn/Li')
#        ax2.set_title(str(ROTE_value) + ' ' + probe[iii])
#        ax2.set_title('phase (deg)', fontsize = 12)
        ax2.set_xlim(q95_values)#([2.1,6.8])
        ax2.set_ylim(Bn_Div_Li_values)#([0.75,3])
        fig.canvas.draw()
        fig.show()

        h_line_list = num.arange(1.,3.,0.1)
        for i in range(0,len(h_line_list)):
            line = ax.axhline(y=h_line_list[i],color = 'k')
            line2 = ax2.axhline(y=h_line_list[i],color = 'k')
            ax.figure.canvas.draw()
            filename = str('%03d' % i) + '.png'
            pt.savefig(filename, dpi=100)
            line.remove()
            line2.remove()
            ax.figure.canvas.draw()

#        fig = pt.figure()
#        ax = fig.add_subplot(111)
