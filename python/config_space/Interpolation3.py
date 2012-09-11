'''
Possible predecessor to the func that plots the poloidal pickup output as a func of bn and q95
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
name = '9_project1_new_eq_FEEDI_-180_coil_outputs.pickle'
ROTE_value_list = [0, -60, -120, -180, -240, -300]
#ROTE_value_list = [0]
ROTE_value_list = [0]

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
    name = '9_project_n_4_FEEDI_0_coil_outputs.pickle'
    name = '9_project1_new_eq_FEEDI_-300_ICOIL_FREQ_0_coil_outputs.pickle'
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

        B1grid_data = griddata(q95_array, Bn_Div_Li_array, coil1_abs_array, xnew, ynew, interp = 'linear')
        interp_data_angle = griddata(q95_array, Bn_Div_Li_array, coil1_angle_array, xnew, ynew, interp = 'linear')


        B1_err = []


        #Print check the errors in the reconstruction

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

        q_list = [3,4.5,6.5]
        color_list3 = ['blue','black','yellow','red']
        for arrow_temp in range(0,len(q_list)):
            ax.arrow(q_list[arrow_temp],0.8, 0, 2, width=0.05,fc=color_list3[arrow_temp], head_width = 0.2, head_length = 0.2)

        #ax.plot(3.6,2.0,'ko',markersize = 10,markerfacecolor='white')
        #ax.plot(project_dict['sims'][initial_eq_serial]['Q95'],project_dict['sims'][initial_eq_serial]['BETAN']/project_dict['sims'][initial_eq_serial]['LI'], 'k*', markersize = 10, markerfacecolor='white')
        ax.set_ylabel('Bn/Li')
        ax.set_title('n=4 ' +str(ROTE_value) + 'deg I-coil phasing ' + probe2[iii],fontsize=14)
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

#        fig = pt.figure()
#        ax = fig.add_subplot(111)
        q95_range = 0.1

        #for q95_value in [2.5,3,4,5,6,6.5,6.8]:
        #for q95_value in [2.5,3,3.65,5,6,6.5,6.8]:
        '''
        for q95_value in [4., 5.,6.]:

            #q95_value = 4
            Bn_Div_Li_list = []
            q95_list = []
            coil_filt = []
            iii = 1
            qmult = []
            pmult = []
            for jjj in project_dict['sims'].keys():

                if  (q95_value - q95_range) < project_dict['sims'][jjj]['Q95'] < (q95_value + q95_range):
                    Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
                    q95_list.append(project_dict['sims'][jjj]['Q95'])
                    qmult.append(project_dict['sims'][jjj]['QMULT'])
                    pmult.append(project_dict['sims'][jjj]['PMULT'])
                    coil_filt.append(num.angle(project_dict['sims'][jjj][calc][iii]-project_dict['sims'][jjj][calc2][iii]))
                    #coil_filt.append(num.abs(project_dict['sims'][i][calc][iii]-project_dict['sims'][i][calc2][iii]))
                    print pmult[-1], qmult[-1], coil_filt[-1]

            fig = pt.figure()
            ax = fig.add_subplot(111)
            ax.plot(Bn_Div_Li_list, coil_filt,'o-', label=str(q95_value))
            fig.canvas.draw()
            fig.show()
            #print str(q95_value), len(coil_filt)
            #print str(q95_value), len(coil_filt)
        '''

    #Bn Slice
    Bn_slice_check = 1
    if Bn_slice_check == 1 :
        fig = pt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        Bn_List = [1.2, 2., 2.8]
        color_list = ['bo-','ko-','yo-']
        color_list2 = ['bx-','kx-','yx-']
        for iii in range(0,len(Bn_List)):
            Bn_Div_Li_value = Bn_List[iii]
            q95_range = num.arange(2.5,6.6,0.1)
            Bn_Div_Li = num.ones(len(q95_range),dtype=float)*Bn_Div_Li_value
            interp_abs_line = griddata(q95_array, Bn_Div_Li_array, coil1_abs_array, q95_range,Bn_Div_Li, interp = 'linear')[0]
            interp_angle_line = griddata(q95_array, Bn_Div_Li_array, coil1_angle_array, q95_range, Bn_Div_Li, interp = 'linear')[0]

            ax.plot(q95_range, interp_abs_line,color_list[iii],label='Bn/Li='+str(Bn_Div_Li_value))
            ax2.plot(q95_range, interp_angle_line,color_list[iii],label='Bn/Li='+str(Bn_Div_Li_value))

        leg = ax.legend(loc=1, fancybox = True)
        leg.get_frame().set_alpha(0.5)
        leg = ax2.legend(loc=1, fancybox = True)
        leg.get_frame().set_alpha(0.5)
        ax.grid()
        ax2.grid()
        ax2.set_ylim([0,360])
        ax.set_xlim([2,7])
        ax2.set_xlim([2,7])
        ax2.set_xlabel('q95')
        ax.set_title('n=4 '+str(ROTE_value) + 'deg ' + probe[iii])
        ax.set_ylabel('abs(output)')
        ax.set_ylabel('Magnitude (G/kA)')
        ax2.set_ylabel('Phase (deg)')
        fig.canvas.draw()
        fig.show()



    #Q95 Slice
    q95_slice_check = 1
    if q95_slice_check ==1:
        fig = pt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        q95_range = 0.1

        #for q95_value in [2.5,3,4,5,6,6.5,6.8]:
        #for q95_value in [2.5,3,3.65,5,6,6.5,6.8]:
        #for q95_value in [3.65, 6.28]:
#        q_list = [3.166, 3.5,3+5./6]
        for q95_i in range(0,len(q_list)):
            q95_value = q_list[q95_i]
            color_list = ['bo-','ko-','yo-','ro-']
            color_list2 = ['bo-','ko-','yo-','ro-']
            #q95_value = 4
            Bn_Div_Li_list = []
            q95_list = []
            coil_filt = []
            iii = 1
            qmult = []
            pmult = []
            fails = 0
            for jjj in project_dict['sims'].keys():
                try:
                    if  (q95_value - q95_range) < project_dict['sims'][jjj]['Q95'] < (q95_value + q95_range):
                        Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN'])#/project_dict['sims'][jjj]['LI'])
                        q95_list.append(project_dict['sims'][jjj]['Q95'])
                        qmult.append(project_dict['sims'][jjj]['QMULT'])
                        pmult.append(project_dict['sims'][jjj]['PMULT'])
                        coil_filt.append(project_dict['sims'][jjj][calc][iii]-project_dict['sims'][jjj][calc2][iii])
                        #coil_filt.append(num.abs(project_dict['sims'][i][calc][iii]-project_dict['sims'][i][calc2][iii]))
                        print 'hello', pmult[-1], qmult[-1], abs(coil_filt[-1]), jjj
                except:
                    print 'ERROR encountered'
                    fails +=1
                    print 'fails: %d'%(fails)
            angle_list = num.angle(coil_filt)
            for angle_i in range(0,len(angle_list)):
                if angle_list[angle_i]<0:
                    angle_list[angle_i]+= 2.*num.pi
            ax.plot(Bn_Div_Li_list, num.abs(coil_filt),color_list[q95_i], label='q95='+str(q95_value))
            ax2.plot(Bn_Div_Li_list, angle_list*180./num.pi, color_list2[q95_i], label='q95='+str(q95_value))
            #print str(q95_value), len(coil_filt)
            #print str(q95_value), len(coil_filt)

        #    ax.plot(Bn_Div_Li_list, q95_list, 'o')
        leg = ax.legend(loc=1)
        leg.get_frame().set_alpha(0.5)
        leg = ax2.legend(loc=1)
        leg.get_frame().set_alpha(0.5)
        ax.grid()
        ax2.grid()
        ax2.set_ylim([0,360])
        ax.set_ylim([0,2.5])
        ax.set_xlim([0.75,3])
        ax2.set_xlim([0.75,3])
        ax2.set_xlabel('Bn')
        ax.set_title('n=4 '+ str(ROTE_value) + 'deg ' + probe[iii])
        ax.set_xlabel('Bn')
        ax.set_ylabel('Magnitude (G/kA)')
        ax2.set_ylabel('Phase (deg)')
        fig.canvas.draw()
        fig.show()


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
