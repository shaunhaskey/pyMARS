import pickle, time
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate
from matplotlib.mlab import griddata
import matplotlib.cm as cm
from scipy.interpolate import griddata as griddata_scipy

def return_initial_eq(project_dict):
    min_distance = 1000
    for jjj in project_dict['sims'].keys():
        qmult_curr = project_dict['sims'][jjj]['QMULT']
        pmult_curr = project_dict['sims'][jjj]['PMULT']
        distance = (qmult_curr - 1.)**2 + (pmult_curr -1.)**2
        if distance < min_distance:
            min_distance = distance *1.
            initial_eq_serial = jjj *1
    print 'serial : %d, qmult %.2f, pmult %.2f, q95: %.2f, Bn/Li: %.2f'%(initial_eq_serial, project_dict['sims'][initial_eq_serial]['QMULT'],project_dict['sims'][initial_eq_serial]['PMULT'],project_dict['sims'][initial_eq_serial]['Q95'],project_dict['sims'][initial_eq_serial]['BETAN']/project_dict['sims'][initial_eq_serial]['LI'])
    return initial_eq_serial

def return_probe_values(project_dict, calc, calc2, iii):
    q95_list = []; Bn_Div_Li_list = [];coil1 = [];passes = 0;fails = 0
    Bn_list = []; Li_list = [];serial_list=[]
    for jjj in project_dict['sims'].keys():
        try:
            if ((project_dict['sims'][jjj]['QMULT'] != 0.81) or (project_dict['sims'][jjj]['PMULT'] !=0.825)):
                if calc2 == None:
                    coil1.append(project_dict['sims'][jjj][calc][iii]) # total response
                else:
                    coil1.append(project_dict['sims'][jjj][calc][iii]-project_dict['sims'][jjj][calc2][iii]) # Plasma only
                q95_list.append(project_dict['sims'][jjj]['Q95'])
                Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
                Li_list.append(project_dict['sims'][jjj]['LI'])
                Bn_list.append(project_dict['sims'][jjj]['BETAN'])
                serial_list.append(jjj)
                passes+=1
                #print probe[iii], coil1[-1], 'pmult', project_dict['sims'][jjj]['PMULT'], 'q95:', q95_list[-1], 'Bn/Li', Bn_Div_Li_list[-1]
        except:
            fails+=1
            del project_dict['sims'][jjj]
            print 'fail'
    print 'pass : %d, fails : %d'%(passes, fails)
    q95_array = num.array(q95_list)
    Bn_Div_Li_array = num.array(Bn_Div_Li_list)
    Bn_array = num.array(Bn_list)
    Li_array = num.array(Li_list)
    coil1_array = num.array(coil1)
    coil1_abs_array = num.abs(num.array(coil1))
    coil1_angle_array = num.angle(num.array(coil1),deg=True)#*180./num.pi
    serial_array = num.array(serial_list)
    return project_dict, q95_array, Bn_Div_Li_array, Bn_array, Li_array,coil1, coil1_abs_array, coil1_angle_array, serial_list



def return_dict_values(project_dict, key_name,serial_list):
    results_list = []
    for jjj in serial_list:
        results_list.append(project_dict['sims'][jjj][key_name])
    return results_list




def return_grid_data(q95_values, Bn_Div_Li_values,q95_array, Bn_Div_Li_array, coil1_angle_array, coil1_abs_array, xnew=None,ynew=None, interpolation='linear',deg_min=0, points = [100,100]):
    if xnew==None:
        xnew = num.linspace(q95_values[0], q95_values[1], points[0])
        ynew = num.linspace(Bn_Div_Li_values[0], Bn_Div_Li_values[1],points[1])

    for i in range(0,len(coil1_angle_array)):
        while coil1_angle_array[i]<deg_min or coil1_angle_array[i]>(deg_min+360):
            if coil1_angle_array[i]<deg_min:
                coil1_angle_array[i] += 360.
            if coil1_angle_array[i]>(deg_min+360):
                coil1_angle_array[i] -= 360.

    B1grid_data = griddata(q95_array, Bn_Div_Li_array, coil1_abs_array, xnew, ynew, interp = interpolation)
    interp_data_angle = griddata(q95_array, Bn_Div_Li_array, coil1_angle_array, xnew, ynew, interp = interpolation)
    return B1grid_data, interp_data_angle


def return_grid_data_new(x_limits, y_limits, x_array, y_array, z_array, xnew=None,ynew=None, interpolation='linear', points = [100,100]):
    if xnew==None:
        xnew = num.linspace(x_limits[0], x_limits[1], points[0])
        ynew = num.linspace(y_limits[0], y_limits[1],points[1])
    gridded_data = griddata(x_array, y_array, z_array, xnew, ynew, interp = interpolation)
    return gridded_data



def plot_Bn_slice(Bn_List, q95_array, Bn_Div_Li_array, coil1_abs_array, coil1_angle_array, ROTE_value, probe):
    fig = pt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
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

    leg = ax.legend(loc=2, fancybox = True)
    leg.get_frame().set_alpha(0.5)
    leg = ax2.legend(loc=4, fancybox = True)
    leg.get_frame().set_alpha(0.5)
    ax.grid()
    ax2.grid()
    ax2.set_ylim([0,360])
    ax.set_xlim([2,7])
    ax2.set_xlim([2,7])
    ax2.set_xlabel('q95')
    ax.set_title(str(ROTE_value) + 'deg ' + probe[iii])
    ax.set_ylabel('abs(output)')
    ax.set_ylabel('Magnitude (G/kA)')
    ax2.set_ylabel('Phase (deg)')
    fig.canvas.draw()
    fig.show()

def plot_q95_slice(q_list, project_dict, theta, probe, calc, calc2, calc3, calc4, iii):
    fig = pt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    q95_range = 0.1

    #        q_list = [2.5, 5, 7]
    for q95_i in range(0,len(q_list)):
        q95_value = q_list[q95_i]
        color_list = ['bo-','ko-','yo-','ro-']
        color_list2 = ['bo-','ko-','yo-','ro-']
        #q95_value = 4
        Bn_Div_Li_list = []
        q95_list = []
        coil_filt = []
        qmult = []
        pmult = []
        fails = 0
        for jjj in project_dict['sims'].keys():
            try:
                if  (q95_value - q95_range) < project_dict['sims'][jjj]['Q95'] < (q95_value + q95_range):
                    Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
                    q95_list.append(project_dict['sims'][jjj]['Q95'])
                    qmult.append(project_dict['sims'][jjj]['QMULT'])
                    pmult.append(project_dict['sims'][jjj]['PMULT'])
                    if calc2==None:
                        coil_filt.append(project_dict['sims'][jjj][calc][iii])
                    else:
                        #coil_filt.append(project_dict['sims'][jjj][calc][iii]-project_dict['sims'][jjj][calc2][iii])
                        upper_value = project_dict['sims'][jjj][calc][iii]-project_dict['sims'][jjj][calc2][iii]
                        lower_value = project_dict['sims'][jjj][calc3][iii]-project_dict['sims'][jjj][calc4][iii]
                        coil_filt.append(upper_value + lower_value * (num.cos(theta/180.*num.pi)+1j*num.sin(theta/180.*num.pi)))
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
        print len(Bn_Div_Li_list)
        print len(coil_filt)
        print len(angle_list)
        ax.plot(Bn_Div_Li_list, num.abs(coil_filt),color_list[q95_i], label='q95='+str(q95_value))
        ax2.plot(Bn_Div_Li_list, angle_list*180./num.pi, color_list2[q95_i], label='q95='+str(q95_value))
        #print str(q95_value), len(coil_filt)
        #print str(q95_value), len(coil_filt)

    #    ax.plot(Bn_Div_Li_list, q95_list, 'o')
    leg = ax.legend(loc=4)
    leg.get_frame().set_alpha(0.5)
    leg = ax2.legend(loc=4)
    leg.get_frame().set_alpha(0.5)
    ax.grid()
    ax2.grid()
    ax2.set_ylim([0,360])
    ax.set_ylim([0,2.5])
    ax.set_xlim([0.75,3])
    ax2.set_xlim([0.75,3])
    ax2.set_xlabel('Beta_n')
    ax.set_title('n=2 ' +str(theta) + 'deg ' + probe[iii])
    ax.set_xlabel('Beta_n')
    ax.set_ylabel('Magnitude (G/kA)')
    ax2.set_ylabel('Phase (deg)')
    fig.canvas.draw()
    fig.show()

def plot_BetaN_q95_space(project_dict):
    q95_list = [];Bn_Div_Li_list = [];passes = 0;fails = 0
    Bn_list = []; Li_list = [];

    for jjj in project_dict['sims'].keys():
        try:
            q95_list.append(project_dict['sims'][jjj]['Q95'])
            Bn_Div_Li_list.append(project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI'])
            Li_list.append(project_dict['sims'][jjj]['LI'])
            Bn_list.append(project_dict['sims'][jjj]['BETAN'])
            passes+=1
            #print probe[iii], coil1[-1], 'pmult', project_dict['sims'][jjj]['PMULT'], 'q95:', q95_list[-1], 'Bn/Li', Bn_Div_Li_list[-1]
        except:
            fails+=1
            del project_dict['sims'][jjj]
    print 'pass : %d, fails : %d'%(passes, fails)

    q95_array = num.array(q95_list)
    Bn_Div_Li_array = num.array(Bn_Div_Li_list)
    Bn_array = num.array(Bn_list)
    Li_array = num.array(Li_list)
    fig = pt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(q95_array, Bn_Div_Li_array,'o')
    ax2.plot(q95_array, Bn_array,'o')
    ax2.set_xlabel('q95')
    ax1.set_ylabel('Beta_N / Li')
    ax2.set_ylabel('Beta_N')
    fig.canvas.draw()
    fig.show()
    return project_dict, q95_array, Bn_Div_Li_array, Bn_array, Li_array
