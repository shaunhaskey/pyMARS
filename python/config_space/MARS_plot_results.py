# Shaun Nov 17 2011 produces a plot of what a pickup coil will see as a function of Beta_n and q95
# can take 2 files as input, an upper only and lower only and can apply a phasing between them

import pickle, time, copy
import numpy as num
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from Post_Proc_Funcs import *
from scipy.interpolate import griddata as griddata

#########Need to set these variables before running!!!!######################
name = 'shot146388_stab_limit_post_processing.pickle'#'shot146388_post_processing.pickle' #must set this !
name = 'shot146388_stab_limit2_post_processing.pickle'
name = 'shot138344_comparison_post_processing.pickle'
name = 'shot146382_scan_post_processing.pickle'
name = '/home/srh112/NAMP_datafiles/24_mar/shot146382_scan_post_processing.pickle'
name = '/home/srh112/NAMP_datafiles/mars/equal_spacing/equal_spacing_post_processing_PEST.pickle'
#name = '/home/srh112/code/DIII-D_Work/24_mar/shot146382_scan_post_processing.pickle'

name2 = None #set to this to do a single calculation
#name = 'combined_upper.pickle'
#name2 = 'combined_lower.pickle'
#name2 = '9_project1_new_eq_COIL_lower_post_setup.pickle'
contains_both = True #Set this for the newer simulations when both upper and lower are included in the file

response_type = 'total' # 'plasma_only' # 'vac' 'total'
data_plot_BetaN = True # True/None

q_list = None#[3,4.5,6.5] #None/list to plot arrows
color_list3 = ['blue','black','yellow','red']

#range of upper lower phasings to include (will loop through all these phasings)
theta_range = num.arange(0,-360,-60)
theta_range = [0]
probe  = [ 'dBp_upper - 67A', 'dBp_mid - 66M', 'dBp_lower - 67B', 'dBr_ext - ESL', 'dBr_mid - ISL','dBr_upper - UISL','dBr_lower  - LISL','Inb_p','Inb_r']
probe2  = [ 'dBp_u', 'dBp_m', 'dBp_l', 'dBr_ext', 'dBr_m','dBr_u','dBr_l','Inb_p','Inb_r']
#Choose the probe to plot (see above for for reference)
probe_list = [0,1,2,3,4,5,6,7,8]
probe_list = [1]

#Plot line slices along the arrow lines
q95_slice_check = 0
Bn_slice_check = 0

q95_values =[2.,8.] #q95 limits
Bn_Div_Li_values = [0.5,5]#[0.75,3.] #Bn/Li limits
x_axis_limits = q95_values
y_axis_limits = Bn_Div_Li_values

#Bn_Div_Li_values = [0,3.]

start_title = 'n=2 ' #to make plots look nice...
deg_min = -10 #min on the scale for phase - to avoid ugly colour wrapping around
plot_simulation_points = True #True/None include plot of experimental data points
expt_data_filename = 'betan-q95-20111116.txt' #name of experimental data point file
expt_data_filename = None

#Colour limits for the plots - each one is for a different pickup coil
clim_list = [[0,1.5],[0,2],[0,1.5],[0,0.7],[0,2],[0,1.5],[0,2],[0,1],[0,1]]
clim_list = [[0,1.5],[0,10],[0,1.5],[0,0.7],[0,1],[0,1.5],[0,2],[0,1],[0,1]]
#clim_list = [[0,1.5],[0,2],[0,1.5],[0,0.7],[0,0.4],[0,1.5],[0,2],[0,1],[0,1]]
#clim_list = [[0,1.5],[0,2],[0,1.5],[0,0.7],[5,7],[0,1.5],[0,2],[0,1],[0,1]]
image_extent = [q95_values[0], q95_values[1], Bn_Div_Li_values[0], Bn_Div_Li_values[1]]
contour_list = [0,0.1, 0.2,0.5,0.8,1.1, 1.5, 2.2]

#expt_data_filename = 'betan-q95-n3-20111116.txt'
######### END ######################

if response_type =='plasma_only':
    calc = 'plasma_response4'
    calc2 = 'vacuum_response4'
    extra_title = ' Plasma Response'
elif response_type == 'total':
    calc = 'plasma_response4'
    calc2 = None # gives total response
    extra_title = ' Total Response'
elif response_type == 'vac':
    calc = 'vacuum_response4'
    calc2 = None # gives total response
    extra_title = ' Vacuum Response'

list_images = []


start = 1

project_dict = pickle.load(open(name))
if name2==None:
    pass
else:
    project_dict2 = pickle.load(open(name2))

#Search for the base case equilibria
if start == 1:
    initial_eq_serial = return_initial_eq(project_dict)
start = 0




def color_plot(ax, fig, plot_data, image_extent, clim_value, contour_list = None, cbar_label = '', include_cbar = True,color_map = cm.jet, set_clim = 1):
    ax_image = ax.imshow(plot_data,extent=image_extent, cmap = color_map, origin='lower')
    if set_clim != 0:
        ax_image.set_clim(clim_value)
    if contour_list != None:
        pt.contour(plot_data, contour_list,colors = 'k',origin='lower', extent=image_extent)
    if include_cbar ==True:
        cbar = fig.colorbar(ax_image)
        cbar.set_label(cbar_label)
    return ax_image

def axis_labelling(ax,ylabel,xlabel,title,xlim,ylim, xlabel_fontsize = 20,ylabel_fontsize = 20, title_fontsize = 14):
    ax.set_xlabel(xlabel,fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel,fontsize=ylabel_fontsize)
    ax.set_title(title,fontsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_experiment_data(file_name, axes_list, coil1_abs_array, coil1_angle_array,plot_figures=None):
    expt_data_data = num.loadtxt(file_name)
    expt_data_q95 = expt_data_data[:,5]
    expt_data_betan = expt_data_data[:,3]

    interp_points = num.ones((num.max(expt_data_q95.shape),2),dtype=float)
    interp_points[:,0] = expt_data_q95
    interp_points[:,1] = expt_data_betan

    existing_points = num.ones((num.max(q95_array.shape),2),dtype=float)
    existing_points[:,0] = q95_array
    existing_points[:,1] = Bn_array

    expt_data2_points_abs = griddata(existing_points,coil1_abs_array,interp_points,method='linear')
    expt_data2_points_angle = griddata(existing_points,coil1_angle_array,interp_points,method='linear')
    tmp1, tmp2 = expt_data_data.shape
    output_data = num.ones((tmp1,tmp2+2),dtype=float)
    output_data[:,0:tmp2] = expt_data_data
    output_data[:,tmp2] = expt_data2_points_abs
    output_data[:,tmp2+1] = expt_data2_points_angle
    num.savetxt('expt_data_output.txt',output_data,fmt='%.4f',delimiter = '    ')
    for ax in axes_list:
        ax.plot(expt_data_q95, expt_data_betan,'kx')
    if plot_figures ==None:
        pass
    else:
        fig_expt_data = pt.figure()
        ax1_expt_data = fig_expt_data.add_subplot(211)
        ax2_expt_data = fig_expt_data.add_subplot(212)
        ax1_expt_data.plot(expt_data_betan,expt_data2_points_abs,'o')
        ax2_expt_data.plot(expt_data_betan,expt_data2_points_angle,'o')
        ax1_expt_data.set_ylim(clim_list[iii])
        ax2_expt_data.set_ylim([-200,200])
        ax1_expt_data.set_title(start_title+ 'Magnitude'+extra_title)
        ax2_expt_data.set_title(start_title+ 'Phase' + extra_title)
        ax2_expt_data.set_xlabel(r'$\beta_N$')
        ax1_expt_data.set_ylabel('G/kA')
        ax2_expt_data.set_ylabel('deg')

        fig_expt_data.canvas.draw()
        fig_expt_data.show()
        fig_expt_data = pt.figure()
        ax1_expt_data = fig_expt_data.add_subplot(211)
        ax2_expt_data = fig_expt_data.add_subplot(212)
        ax1_expt_data.plot(expt_data_q95,expt_data2_points_abs,'o')
        ax2_expt_data.plot(expt_data_q95,expt_data2_points_angle,'o')
        ax1_expt_data.set_ylim(clim_list[iii])
        ax2_expt_data.set_ylim([-200,200])
        ax1_expt_data.set_title(start_title + 'Magnitude'+extra_title)
        ax2_expt_data.set_title(start_title + 'Phase' + extra_title)
        ax2_expt_data.set_xlabel('q95')
        ax1_expt_data.set_ylabel('G/kA')
        ax2_expt_data.set_ylabel('deg')

        fig_expt_data.canvas.draw()
        fig_expt_data.show()



def plot_DCON(dictionary, q95_values,Bn_Div_Li_values,image_extent,q95_filt=None,Bn_filt=None):
    serial_list = dictionary['sims'].keys()
    WTOTN1_values = num.array(return_dict_values(dictionary, 'WTOTN1', serial_list))
    WTOTN2_values = num.array(return_dict_values(dictionary, 'WTOTN2', serial_list))
    WTOTN3_values = num.array(return_dict_values(dictionary, 'WTOTN3', serial_list))
    WWTOTN1_values = num.array(return_dict_values(dictionary, 'WWTOTN1', serial_list))

    q95_array = num.array(return_dict_values(dictionary, 'Q95', serial_list))
    Bn_array = num.array(return_dict_values(dictionary, 'BETAN', serial_list))


    WTOTN1_grid = return_grid_data_new(q95_values, Bn_Div_Li_values, q95_array, Bn_array, WTOTN1_values, xnew=None,ynew=None, interpolation='linear', points = [100,100])
    WTOTN2_grid = return_grid_data_new(q95_values, Bn_Div_Li_values, q95_array, Bn_array, WTOTN2_values, xnew=None,ynew=None, interpolation='linear', points = [100,100])
    WTOTN3_grid = return_grid_data_new(q95_values, Bn_Div_Li_values, q95_array, Bn_array, WTOTN3_values, xnew=None,ynew=None, interpolation='linear', points = [100,100])
    WWTOTN1_grid = return_grid_data_new(q95_values, Bn_Div_Li_values, q95_array, Bn_array, WWTOTN1_values, xnew=None,ynew=None, interpolation='linear', points = [100,100])


    q95_unfilt = num.array(return_dict_values(dictionary, 'Q95', dictionary['sims'].keys()))
    Bn_unfilt = num.array(return_dict_values(dictionary, 'BETAN', dictionary['sims'].keys()))
    qmax = num.array(return_dict_values(dictionary, 'QMAX', dictionary['sims'].keys()))
    qmin = num.array(return_dict_values(dictionary, 'QMIN', dictionary['sims'].keys()))
    print num.max(qmax), '**************************'
    qmax_grid = return_grid_data_new(q95_values, Bn_Div_Li_values, q95_array, Bn_array, qmax, xnew=None,ynew=None, interpolation='linear', points = [100,100])
    qmin_grid = return_grid_data_new(q95_values, Bn_Div_Li_values, q95_array, Bn_array, qmin, xnew=None,ynew=None, interpolation='linear', points = [100,100])


    fig = pt.figure()
    ax = fig.add_subplot(221)
    list_images.append(color_plot(ax, fig, WTOTN1_grid, image_extent, [0,1], cbar_label = 'WTOTN1_DCON', include_cbar = True,color_map = cm.bone))
    ax2 = fig.add_subplot(222)
    list_images.append(color_plot(ax2, fig, WTOTN2_grid, image_extent, [0,1], cbar_label = 'WTOTN2_DCON', include_cbar = True,color_map = cm.bone))
    ax3 = fig.add_subplot(223)
    list_images.append(color_plot(ax3, fig, WTOTN3_grid, image_extent, [0,1], cbar_label = 'WTOTN3_DCON', include_cbar = True,color_map = cm.bone))
    ax4 = fig.add_subplot(224)
    #list_images.append(color_plot(ax4, fig, WWTOTN1_grid, image_extent, [0,1], cbar_label = 'WWTOTN1', include_cbar = True,color_map = cm.bone))

    contour_list = num.arange(0,20,dtype=float)
    contour_list = contour_list / (contour_list*0+1)
    #list_images.append(color_plot(ax4, fig, qmax_grid, image_extent, [0,12], cbar_label = 'QMAX', include_cbar = True,color_map = cm.jet,contour_list=contour_list))
    list_images.append(color_plot(ax4, fig, qmin_grid, image_extent, [1,1.2], cbar_label = 'QMIN', include_cbar = True,color_map = cm.jet,contour_list=contour_list))
    ax3.set_ylabel('Beta_N')
    ax3.set_xlabel('q95')

    plot_type = ['yo','bo']

    for i in range(0, len(q95_filt)):
        q95_filt_tmp = q95_filt[i]
        Bn_filt_tmp = Bn_filt[i]
        plot_style = plot_type[i]
        ax.plot(q95_filt_tmp,Bn_filt_tmp,plot_style)
        ax2.plot(q95_filt_tmp,Bn_filt_tmp,plot_style)
        ax3.plot(q95_filt_tmp,Bn_filt_tmp,plot_style)
        ax4.plot(q95_filt_tmp,Bn_filt_tmp,plot_style)

    ax.plot(q95_unfilt,Bn_unfilt,',')
    ax2.plot(q95_unfilt,Bn_unfilt,',')
    ax3.plot(q95_unfilt,Bn_unfilt,',')
    ax4.plot(q95_unfilt,Bn_unfilt,',')
    #else:

    fig.canvas.draw()
    fig.show()




for iii in probe_list:
    num_count = 1
    for theta_i in range(0,len(theta_range)):
        if contains_both:
            if response_type =='plasma_only':
                calc = 'plasma_upper_response4'
                calc2 = 'vacuum_upper_response4'
                calc3 = 'plasma_lower_response4'
                calc4 = 'vacuum_lower_response4'
                extra_title = ' Plasma Response'
            elif response_type == 'total':
                calc = 'plasma_upper_response4'
                calc2 = None # gives total response
                calc3 = 'plasma_lower_response4'
                calc4 = None # gives total response
                extra_title = ' Total Response'
            elif response_type == 'vac':
                calc = 'vacuum_upper_response4'
                calc2 = None # gives total response
                calc3 = 'vacuum_lower_response4'
                calc4 = None # gives total response
                extra_title = ' Vacuum Response'
            #calc = 'plasma_upper_response4'
            #calc2 = 'vacuum_upper_response4'
            project_dict, q95_array, Bn_Div_Li_array,Bn_array, Li_array, coil1, coil1_abs_array, coil1_angle_array,serial_list = return_probe_values(project_dict, calc, calc2, iii)
            project_dict, q95_array2, Bn_Div_Li_array2, Bn_array2, Li_array2, coil12, coil1_abs_array2, coil1_angle_array2,serial_list2 = return_probe_values(project_dict, calc3, calc4, iii)
            theta = theta_range[theta_i]/180.*num.pi #in radians
            print '****************************',theta,'****************************'
            print 'Check to see q95 array and Bn_Div_Li arrays are the same'
            print num.sum(num.abs(q95_array2-q95_array))
            print num.sum(num.abs(Bn_Div_Li_array2-Bn_Div_Li_array))
            coil1 = num.array(coil1) + num.array(coil12)*(num.cos(theta)+1j*num.sin(theta))
            coil1_abs_array = num.abs(num.array(coil1))
            coil1_angle_array = num.angle(num.array(coil1))*180./num.pi #degrees
        else:
            project_dict, q95_array, Bn_Div_Li_array,Bn_array, Li_array, coil1, coil1_abs_array, coil1_angle_array,serial_list = return_probe_values(project_dict, calc, calc2, iii)

            if name2==None:
                pass
            else:
                project_dict2, q95_array2, Bn_Div_Li_array2, Bn_array2, Li_array2, coil12, coil1_abs_array2, coil1_angle_array2,serial_list2 = return_probe_values(project_dict2, calc, calc2, iii)
                theta = theta_range[theta_i]/180.*num.pi #in radians
                print '****************************\n',theta,'****************************'
                print 'Check to see q95 array and Bn_Div_Li arrays are the same'
                print num.sum(num.abs(q95_array2-q95_array))
                print num.sum(num.abs(Bn_Div_Li_array2-Bn_Div_Li_array))
                coil1 = num.array(coil1) + num.array(coil12)*(num.cos(theta)+1j*num.sin(theta))
                coil1_abs_array = num.abs(num.array(coil1))
                coil1_angle_array = num.angle(num.array(coil1))*180./num.pi #degrees

        if data_plot_BetaN == True:
            B1grid_data, interp_data_angle = return_grid_data(q95_values, Bn_Div_Li_values,q95_array, Bn_array, coil1_angle_array, coil1_abs_array,deg_min=deg_min)
            plot_ylabel = r'$\beta_N$'
        else:
            B1grid_data, interp_data_angle = return_grid_data(q95_values, Bn_Div_Li_values, q95_array, Bn_Div_Li_array, coil1_angle_array, coil1_abs_array,deg_min=deg_min)
            plot_ylabel = r'$\beta_N/L_i$'

        fig = pt.figure()
        ax = fig.add_subplot(211)
        list_images.append(color_plot(ax, fig, B1grid_data, image_extent, clim_list[iii], contour_list = contour_list, cbar_label = '|B| G/kA', include_cbar = True))


        if plot_simulation_points == True:
            if data_plot_BetaN ==True:
                ax.plot(q95_array, Bn_array,'k,')
                #ax2.plot(q95_array, Bn_array,'k,')
            else:
                ax.plot(q95_array, Bn_Div_Li_array,'k,')
                #ax2.plot(q95_array, Bn_Div_Li_array,'k,')
        if q_list == None:
            pass
        else:
            for arrow_temp in range(0,len(q_list)):
                ax.arrow(q_list[arrow_temp],Bn_Div_Li_values[0], 0, (Bn_Div_Li_values[1]-Bn_Div_Li_values[0])-0.2, width=0.05,fc=color_list3[arrow_temp], head_width = 0.2, head_length = 0.2)

        #ax.plot(project_dict['sims'][initial_eq_serial]['Q95'],project_dict['sims'][initial_eq_serial]['BETAN']/project_dict['sims'][initial_eq_serial]['LI'], 'k*', markersize = 10, markerfacecolor='white')
        axis_labelling(ax,plot_ylabel,'',start_title + str(theta_range[theta_i]) + 'deg I-coil phasing ' + probe2[iii] + extra_title, q95_values,Bn_Div_Li_values, xlabel_fontsize = 20,ylabel_fontsize = 20, title_fontsize = 14)

        ax2 = fig.add_subplot(212)
        list_images.append(color_plot(ax2, fig, interp_data_angle, image_extent, [deg_min, deg_min+360], cbar_label = 'deg', include_cbar = True))


        if expt_data_filename !=None:
            plot_experiment_data(expt_data_filename,[ax,ax2],coil1_abs_array, coil1_angle_array, plot_figures=None)

        axis_labelling(ax2, plot_ylabel,'q95','', q95_values,Bn_Div_Li_values, xlabel_fontsize = 14,ylabel_fontsize = 20, title_fontsize = 14)
        fig.canvas.draw()
        fig.show()

        pt.savefig('%03d.png'%(num_count))
        num_count+=1
        #Bn Slice
        if Bn_slice_check == 1 :
            Bn_List = [1.2, 2., 2.8]
            plot_Bn_slice(Bn_List, q95_array, Bn_Div_Li_array, coil1_abs_array, coil1_angle_array, theta_range[theta_i], probe)

        #Q95 Slice
        if q95_slice_check ==1:
            plot_q95_slice(q_list, project_dict, theta_range[theta_i], probe, calc, calc2, calc3, calc4,iii)




'''
        stab_name1 = 'stab_setup_results.dat'
        stab_name2 = 'stab_setup_results_new.dat'
        unfiltered_dict = {}
        unfiltered_dict2 = {}
        unfiltered_dict2['sims'] = read_stab_results(stab_name1)
        unfiltered_dict['sims'] = read_stab_results(stab_name2)

        ser_max = num.max(unfiltered_dict['sims'].keys())+1
        print ser_max
        plot_DCON=None

        for i in unfiltered_dict2['sims'].keys():
            unfiltered_dict['sims'][ser_max] = unfiltered_dict2['sims'][i]
            ser_max += 1

        unfiltered_dict2=copy.deepcopy(unfiltered_dict)
        removed_DCON = 0
        q95_removed_DCON = []
        Bn_removed_DCON = []
        q95_removed_range = []
        Bn_removed_range = []
        q95_range = [2, 7]
        Bn_Div_Li_range = [0., 3]

        for i in unfiltered_dict2['sims'].keys():
            current_q95 = unfiltered_dict2['sims'][i]['Q95']
            current_Bn_Div_Li = unfiltered_dict2['sims'][i]['BETAN']/unfiltered_dict2['sims'][i]['LI']
            try:
                WTOTN1 = unfiltered_dict2['sims'][i]['WTOTN1']
                WTOTN2 = unfiltered_dict2['sims'][i]['WTOTN2']
                WTOTN3 = unfiltered_dict2['sims'][i]['WTOTN3']
                WWTOTN1 = unfiltered_dict2['sims'][i]['WWTOTN1']
                if (WTOTN1<=0) or  (WTOTN2<=0)  or (WTOTN3<=0) or (WWTOTN1<=0):
                    print 'removed item due to stability'
                    print WTOTN1, WTOTN2,WTOTN3, WWTOTN1, current_q95, unfiltered_dict2['sims'][i]['BETAN'], removed_DCON
                    removed_DCON += 1
                    q95_removed_DCON.append(current_q95)
                    Bn_removed_DCON.append(unfiltered_dict2['sims'][i]['BETAN'])
                    del unfiltered_dict2['sims'][i]
                elif (current_q95<q95_range[0]) or (current_q95>q95_range[1]):
                    q95_removed_range.append(current_q95)
                    Bn_removed_range.append(unfiltered_dict2['sims'][i]['BETAN'])
                    del unfiltered_dict2['sims'][i]
                    print 'removed item q95 out of range'
                    #removed_q95 += 1
                elif (current_Bn_Div_Li<Bn_Div_Li_range[0]) or (current_Bn_Div_Li>Bn_Div_Li_range[1]):
                    q95_removed_range.append(current_q95)
                    Bn_removed_range.append(unfiltered_dict2['sims'][i]['BETAN'])
                    del unfiltered_dict2['sims'][i]
                    print 'removed item Bn_Div_Li out of range'
                    #removed_Bn += 1

                else:
                    pass
            except:
                del unfiltered_dict2['sims'][i]
                print 'removed due to error reading'
                #removed_read_error += 1

        if plot_DCON==True:
            plot_DCON(unfiltered_dict, q95_values, Bn_Div_Li_values, image_extent, q95_filt = [q95_removed_DCON,q95_removed_range], Bn_filt = [Bn_removed_DCON,Bn_removed_range])
        
'''
