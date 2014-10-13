'''
SH : Nov 21 2012
This is useful for creating PEST plot images
It can make an animation and introduce different phasings
'''

from pyMARS.results_class import *
import pyMARS.results_class as res_class
from pyMARS.RZfuncs import I0EXP_calc
import pyMARS.generic_funcs as gen_funcs
import numpy as np
import matplotlib.pyplot as pt
import copy
N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
I = np.array([1.,-0.5,-0.5,1,-0.5,-0.5])
I0EXP = I0EXP_calc(N,n,I); facn = 1.0

I0EXP = I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

#various simulation directories to get the components
base_dir = '/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/'
base_dir = '/home/srh112/NAMP_datafiles/mars/shot158115_04780/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
#base_dir = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_n4/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
#base_dir = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/qmult1.000/exp1.000/marsrun/'
dir_loc_lower_t = base_dir + '/RUN_rfa_lower.p'
dir_loc_upper_t = base_dir + '/RUN_rfa_upper.p'
dir_loc_lower_v = base_dir + '/RUN_rfa_lower.vac'
dir_loc_upper_v = base_dir + '/RUN_rfa_upper.vac'

#Load data including PEST data
d_upper_t = data(dir_loc_upper_t, I0EXP=I0EXP)
d_lower_t = data(dir_loc_lower_t, I0EXP=I0EXP)
d_upper_v = data(dir_loc_upper_v, I0EXP=I0EXP)
d_lower_v = data(dir_loc_lower_v, I0EXP=I0EXP)
d_upper_t.get_PEST(facn=facn)
d_lower_t.get_PEST(facn=facn)
d_upper_v.get_PEST(facn=facn)
d_lower_v.get_PEST(facn=facn)

animation_phasings = 1
filename_list = []
amp_phase = False
annotate_plots = False

plots = ['v','p','t']
if animation_phasings:
    phasings = range(0,360,15)
    #phasings = [0]
    #phasings = range(0,360,120)
    for i, phasing in enumerate(phasings):
        print phasing
        R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_upper_t, d_lower_t, phasing)
        R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_upper_v, d_lower_v, phasing)
        #fig,ax = pt.subplots(ncols = 3, sharex=True, sharey=True)
        fig,ax = pt.subplots(ncols = len(plots), sharex=True, sharey=True)
        pt.subplots_adjust(wspace = .05)
        move_up = 0.15
        for j_tmp in range(0,len(ax)):
            ax_tmp = ax[j_tmp]
            start_point = ax_tmp.get_position().bounds
            if j_tmp==0:
                x_bot = start_point[0]
                y_bot = start_point[1]
                print x_bot
            if j_tmp==len(ax)-1:
                x_width = (start_point[0]+start_point[2])-x_bot
                print x_width
            start_point = (start_point[0],start_point[1]+move_up,start_point[2],start_point[3]-move_up)
            ax_tmp.set_position(start_point)
        cbar_axes = fig.add_axes([x_bot,y_bot,x_width,move_up-0.1])
        combined_t = copy.deepcopy(d_upper_t)
        combined_v = copy.deepcopy(d_upper_t)
        combined_p = copy.deepcopy(d_upper_t)
        combined_t.BnPEST = BnPEST_t
        combined_v.BnPEST = BnPEST_v
        combined_p.BnPEST = BnPEST_t-BnPEST_v
        plot_stuff = []
        titles = []
        for p in plots:
            if p == 'v':
                plot_stuff.append(combined_v)
                titles.append('Vac ({:3d}deg)')
            elif p == 't':
                plot_stuff.append(combined_t)
                titles.append('Total ({:3d}deg)')
            elif p == 'p':
                plot_stuff.append(combined_p)
                titles.append('Plas ({:3d}deg)')
        contour_levels1 = np.linspace(0,3.0,7)
        contour_levels2 = np.linspace(0,3.0,7)
        sqrt_flux = False
        rmax = 3.5
        color_plots = []
        for ax_tmp, cur_obj in zip(ax, plot_stuff):
            if amp_phase:
                color_plots.append(cur_obj.plot_BnPEST(ax_tmp, n=n, inc_contours = 1, contour_levels=contour_levels1, increase_grid_BnPEST = 1,cmap = 'RdBu', phase_ref = True, rmax = rmax, phase_ref_array = combined_v.BnPEST*0+1, sqrt_flux = sqrt_flux))
            else:
                color_plots.append(cur_obj.plot_BnPEST(ax_tmp, n=n, inc_contours = 1, contour_levels=contour_levels1, increase_grid_BnPEST = 1,cmap = 'hot', sqrt_flux = sqrt_flux))
        for tmp_loc in range(0,len(color_plots)):
            if amp_phase:
                color_plots[tmp_loc].set_clim([-np.pi,np.pi])
            else:
                color_plots[tmp_loc].set_clim([0,3])
            ax[tmp_loc].set_xlabel('m')
            ax[tmp_loc].set_title(titles[tmp_loc].format(phasing))
        if sqrt_flux:
            ax[0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
        else:
            ax[0].set_ylabel(r'$\psi_N$', fontsize = 14)
        ax[0].set_xlim([0,15])
        ax[0].set_ylim([0.4,0.995])

        ax[0].set_xlim([6,12])
        ax[0].set_ylim([0.90,0.995])
        ax[0].set_xlim([0,20])
        ax[0].set_ylim([0.80,0.995])
        ax[0].set_xlim([-20,20])
        ax[0].set_ylim([0.30,0.995])
        
        if annotate_plots:
            ax[1].annotate('kink-\nresonant', xy=(8, 0.9), xytext=(8.9, 0.6),arrowprops=dict(facecolor='black', shrink=0.05,ec='white'),color='white')
            ax[1].annotate('pitch-\nresonant', xy=(6.93, 0.952), xytext=(0.3, 0.7),arrowprops=dict(facecolor='black', shrink=0.05,ec='white'),color='white')
            #ax[1].text(3.5,0.95,"kink-resonant response",rotation=70,fontsize=15,horizontalalignment='left')
        if amp_phase:
            cbar = pt.colorbar(color_plot_v, cax = cbar_axes ,orientation='horizontal')
            #gen_funcs.create_cbar_ax(original_ax, pad = 3, loc = "right", prop = 5):
            cbar.ax.cla()
            #cbar = pt.colorbar(color_plot_v, cax = cbar_axes ,orientation='horizontal')
            res_class.hue_sat_cbar(cbar.ax, rmax = rmax)
            gen_funcs.setup_axis_publication(cbar.ax, n_xticks = 5, n_yticks = 2)
            cbar.ax.set_yticks([0,4.5])
        else:
            cbar = pt.colorbar(color_plots[0], cax = cbar_axes ,orientation='horizontal')
            cbar.ax.set_xlabel('G/kA')
        filename_list.append('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/plas_%03d.png'%(phasing))
        fig.savefig(filename_list[-1], bbox_inches = 'tight', dpi = 150)
os.system('convert -delay {} -loop 0 {} {}'.format(20, ' '.join(filename_list), 'changed_phasing_n{}.gif'.format(n)))
os.system('zip {} {}'.format('animation_images.zip', ' '.join(filename_list)))


animation_realspace = True
if animation_realspace:
    plots = ['v','p','t']
    plots = ['v']#,'p','t']
    #for plots in [['v'],['p'],['t']]:
    for plots in [['v','p','t']]:
        filename_list = []; 
        phasings = range(0,360,15)
        #phasings = [0]
        #phasings = range(0,360,120)
        for i, phasing in enumerate(phasings):
            titles = []
            print phasing
            R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_lower_t, d_upper_t, phasing)
            R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_lower_v, d_upper_v, phasing)
            #fig,ax = pt.subplots(ncols = 3, sharex=True, sharey=True)
            fig,ax = pt.subplots(ncols = len(plots), sharex=True, sharey=True)
            if len(plots)==1: ax = [ax]
            pt.subplots_adjust(wspace = .05)
            cmap = 'RdBu'
            plot_stuff = []
            for p in plots:
                if p == 'v':
                    plot_stuff.append(Bn_v)
                    titles.append('Vac ({:3d}deg)')
                elif p == 't':
                    plot_stuff.append(Bn_t)
                    titles.append('Total ({:3d}deg)')
                elif p == 'p':
                    plot_stuff.append(Bn_t - Bn_v)
                    titles.append('Plas ({:3d}deg)')
            clim = [-5,5]
            for ax_tmp, p, title in zip(ax, plot_stuff, titles):
                color_ax = ax_tmp.pcolormesh(R_t[:d_upper_t.Ns2,:],Z_t[:d_upper_t.Ns2,:], np.real(p[:d_upper_t.Ns2,:]), cmap=cmap, rasterized=True, shading='gouraud')
                color_ax.set_clim(clim)
                ax_tmp.set_title(title.format(phasing))
                if ax_tmp==ax[0]:
                    if ax.__class__!=list:
                        cbar = pt.colorbar(color_ax,ax = ax.tolist())
                    else:
                        cbar = pt.colorbar(color_ax,ax = ax)
                    cbar.set_label('$B_n$ (G/kA)')
                ax_tmp.set_ylim([-0.8,0.8])
                ax_tmp.plot(R_v[d_upper_t.Ns2,:],Z_v[d_upper_t.Ns2,:])
                ax_tmp.set_aspect('equal')
                ax_tmp.set_xlabel('R(m)')
                ax_tmp.set_xlim([0.5,1.5])
            #color_ax = ax[0].pcolormesh(R_v[:d_upper_t.Ns2,:],Z_v[:d_upper_t.Ns2,:], np.real(Bn_v[:d_upper_t.Ns2,:]), cmap=cmap, rasterized=True, shading='gouraud')
            #color_ax.set_clim(clim)
            ax[0].set_ylabel('Z(m)')
            #ax[0].set_title('Vac Only ({:3d}deg)'.format(phasing))
            #ax[1].set_title('Total ({:3d}deg)'.format(phasing))
            #ax[0].set_xlim([0.5,1.5])
            #ax[0].set_ylim([-0.8,0.8])
            filename_list.append('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/test_%03d.png'%(phasing))
            fig.savefig(filename_list[-1], bbox_inches = 'tight', dpi = 150)
            #fig.savefig(filename_list[-1])
            fig.canvas.draw(); fig.show()
        os.system('convert -delay {} -loop 0 {} {}'.format(20, ' '.join(filename_list), '2D_{}.gif'.format('_'.join(plots))))
    #os.system('zip {} {}'.format('animation_images.zip', ' '.join(filename_list)))

import mayavi.mlab as mlab
start_phi = np.pi/1.5
start_phi = np.deg2rad(120)
end_phi = start_phi + np.pi*2*0.75
f = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size = [750, 750])
filename_list = []
phasings = np.array([ -1.80000000e+02,  -1.71149167e+02,  -1.62593333e+02,
        -1.54327500e+02,  -1.46346667e+02,  -1.38645833e+02,
        -1.31220000e+02,  -1.24064167e+02,  -1.17173333e+02,
        -1.10542500e+02,  -1.04166667e+02,  -9.80408333e+01,
        -9.21600000e+01,  -8.65191667e+01,  -8.11133333e+01,
        -7.59375000e+01,  -7.09866667e+01,  -6.62558333e+01,
        -6.17400000e+01,  -5.74341667e+01,  -5.33333333e+01,
        -4.94325000e+01,  -4.57266667e+01,  -4.22108333e+01,
        -3.88800000e+01,  -3.57291667e+01,  -3.27533333e+01,
        -2.99475000e+01,  -2.73066667e+01,  -2.48258333e+01,
        -2.25000000e+01,  -2.03241667e+01,  -1.82933333e+01,
        -1.64025000e+01,  -1.46466667e+01,  -1.30208333e+01,
        -1.15200000e+01,  -1.01391667e+01,  -8.87333333e+00,
        -7.71750000e+00,  -6.66666667e+00,  -5.71583333e+00,
        -4.86000000e+00,  -4.09416667e+00,  -3.41333333e+00,
        -2.81250000e+00,  -2.28666667e+00,  -1.83083333e+00,
        -1.44000000e+00,  -1.10916667e+00,  -8.33333333e-01,
        -6.07500000e-01,  -4.26666667e-01,  -2.85833333e-01,
        -1.80000000e-01,  -1.04166667e-01,  -5.33333333e-02,
        -2.25000000e-02,  -6.66666667e-03,  -8.33333333e-04,
         0.00000000e+00,   8.33333333e-04,   6.66666667e-03,
         2.25000000e-02,   5.33333333e-02,   1.04166667e-01,
         1.80000000e-01,   2.85833333e-01,   4.26666667e-01,
         6.07500000e-01,   8.33333333e-01,   1.10916667e+00,
         1.44000000e+00,   1.83083333e+00,   2.28666667e+00,
         2.81250000e+00,   3.41333333e+00,   4.09416667e+00,
         4.86000000e+00,   5.71583333e+00,   6.66666667e+00,
         7.71750000e+00,   8.87333333e+00,   1.01391667e+01,
         1.15200000e+01,   1.30208333e+01,   1.46466667e+01,
         1.64025000e+01,   1.82933333e+01,   2.03241667e+01,
         2.25000000e+01,   2.48258333e+01,   2.73066667e+01,
         2.99475000e+01,   3.27533333e+01,   3.57291667e+01,
         3.88800000e+01,   4.22108333e+01,   4.57266667e+01,
         4.94325000e+01,   5.33333333e+01,   5.74341667e+01,
         6.17400000e+01,   6.62558333e+01,   7.09866667e+01,
         7.59375000e+01,   8.11133333e+01,   8.65191667e+01,
         9.21600000e+01,   9.80408333e+01,   1.04166667e+02,
         1.10542500e+02,   1.17173333e+02,   1.24064167e+02,
         1.31220000e+02,   1.38645833e+02,   1.46346667e+02,
         1.54327500e+02,   1.62593333e+02,   1.71149167e+02])

phasings = range(0,360,3)
#phasings = [0]
val = 5
#plots = ['v','t']
#for plots in [['v','t']]:
for plots in [['t']]:
    if len(plots)==3:
        offset = [2,0,-2]
    elif len(plots)==2:
        offset = [-1,1]
    else:
        offset = [0]

    os.system('rm  blah/*.png')
    for ii, phasing in enumerate(phasings):
        R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_lower_t, d_upper_t, phasing)
        R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_lower_v, d_upper_v, phasing)

        plot_stuff = []
        for p in plots:
            if p == 'v':
                plot_stuff.append(Bn_v)
                #titles.append('Vac ({:3d}deg)')
            elif p == 't':
                plot_stuff.append(Bn_t)
                #titles.append('Total ({:3d}deg)')
            elif p == 'p':
                plot_stuff.append(Bn_t - Bn_v)
                #titles.append('Plas ({:3d}deg)')

        for cur_B, off in zip(plot_stuff, offset):
            R = R_v[d_upper_t.Ns2-2,:]
            Z1 = Z_v[d_upper_t.Ns2-2,:]
            B1 = cur_B[d_upper_t.Ns2-2,:]
            #B2 = Bn_v[d_upper_t.Ns2-2,:]
            phi_list = np.linspace(start_phi, end_phi, 100)
            X = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            Y = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            Z = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            B = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            #BB = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            for i, phi in enumerate(phi_list):
                X[:,i] = R*np.cos(phi)
                Y[:,i] = R*np.sin(phi)
                Z[:,i] = +Z1
                B[:,i] = +np.real(B1*np.exp(1j*2*(phi-start_phi)))
                #BB[:,i] = +np.real(B2*np.exp(1j*2*(phi-start_phi)))

            mlab.mesh(X, Y, Z+off, scalars=B, colormap='RdBu', vmax=val, vmin=-val)
            #mlab.mesh(X, Y, Z-1, scalars = BB, colormap = 'RdBu', vmax = val, vmin = -val)
            for phi in [start_phi, end_phi]:
                R = R_v[:d_upper_t.Ns2, :]
                Z = Z_v[:d_upper_t.Ns2, :]
                X = R*np.cos(phi)
                Y = R*np.sin(phi)
                B = +np.real(cur_B[:d_upper_t.Ns2,:]*np.exp(1j*2*(phi-start_phi)))
                #BB = +np.real(Bn_v[:d_upper_t.Ns2,:]*np.exp(1j*2*(phi-start_phi)))
                mlab.mesh(X, Y, Z+off, scalars = B, colormap='RdBu', vmax = val, vmin = -val)
                #mlab.mesh(X, Y, Z-1, scalars = BB, colormap='RdBu', vmax = val, vmin = -val)
            #f.scene.camera.elevation(15);f.scene.render()
            #/home/srh112/code/NAMP_analysis/python/MARS_post_processing/
        filename_list.append('blah/test_%03d.png'%(ii))
        mlab.savefig(filename_list[-1], magnification = 3)
        mlab.clf()
    #os.system('mogrify -crop 1350x1680+477+228 {}'.format(' '.join(filename_list)))
    os.system('mogrify -crop 1572x1182+369+522 {}'.format(' '.join(filename_list)))
    #os.system('convert -delay {} -loop 0 {} {}'.format(20, ' '.join(filename_list), '3D_{}.gif'.format('_'.join(plots))))

count = +ii
for foo in range(3):
    for i in range(len(phasings)):
        count+=1
        print count
        os.system('cp {} {}'.format('blah/test_%03d.png'%(i), 'blah/test_%03d.png'%(count)))

#ffmpeg -loop 1 -framerate 20 -i test_%03d.png  -r 30 -t 15 -pix_fmt yuv420p out.mp4
#os.system('ffmpeg -q scale 1 -r 20 -i 9600 {} movie.mp4'.format(' '.join([' '.join(filename_list) for i in range(5)])))
#os.system('ffmpeg -q scale 1 -r 20 -i -b 9600 {} movie.mp4'.format(' '.join([' '.join(filename_list) for i in range(5)])))
#os.system('avconv -q scale 1 -r 20 -i {} movie.mp4'.format(' '.join([' '.join(filename_list) for i in range(5)])))
#os.system('ffmpeg -framerate 10 -i blah/test_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p vacuum_only.mp4')
os.system('ffmpeg -framerate 20 -i blah/test_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p total.mp4')

os.system('ffmpeg  -i blah/test_%03d.png -c:v libx264 -r 10 -pix_fmt yuv420p out.mp4')





import mayavi.mlab as mlab
start_phi = np.pi/1.5
start_phi = np.deg2rad(120)
end_phi = start_phi + np.pi*2*0.75
f = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size = [750, 750])
filename_list = []
phasings = range(0,360,15)
#phasings = [0]
val = 7
plots = ['v','p','t']
ul_phasing = 180
for plots in [['v', 'p', 't']]:
    if len(plots)==3:
        offset = [2,0,-2]
    elif len(plots)==2:
        offset = [-1,1]
    else:
        offset = [0]
    
    os.system('rm  blah/*.png')
    R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_lower_t, d_upper_t, ul_phasing)
    R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_lower_v, d_upper_v, ul_phasing)
    for ii, phasing in enumerate(phasings):
        plot_stuff = []
        for p in plots:
            if p == 'v':
                plot_stuff.append(Bn_v)
                #titles.append('Vac ({:3d}deg)')
            elif p == 't':
                plot_stuff.append(Bn_t)
                #titles.append('Total ({:3d}deg)')
            elif p == 'p':
                plot_stuff.append(Bn_t - Bn_v)
                #titles.append('Plas ({:3d}deg)')

        for cur_B, off in zip(plot_stuff, offset):
            R = R_v[d_upper_t.Ns2-2,:]
            Z1 = Z_v[d_upper_t.Ns2-2,:]
            B1 = cur_B[d_upper_t.Ns2-2,:]
            #B2 = Bn_v[d_upper_t.Ns2-2,:]
            phi_list = np.linspace(start_phi, end_phi, 100)
            X = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            Y = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            Z = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            B = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            #BB = np.zeros((R.shape[0],len(phi_list)),dtype = float)
            for i, phi in enumerate(phi_list):
                X[:,i] = R*np.cos(phi)
                Y[:,i] = R*np.sin(phi)
                Z[:,i] = +Z1
                B[:,i] = +np.real(B1*np.exp(1j*2*(phi-start_phi))*np.exp(1j*(np.deg2rad(phasing))))
                #BB[:,i] = +np.real(B2*np.exp(1j*2*(phi-start_phi)))

            mlab.mesh(X, Y, Z+off, scalars = B, colormap = 'RdBu', vmax = val, vmin = -val)
            #mlab.mesh(X, Y, Z-1, scalars = BB, colormap = 'RdBu', vmax = val, vmin = -val)
            for phi in [start_phi, end_phi]:
                R = R_v[:d_upper_t.Ns2,:]
                Z = Z_v[:d_upper_t.Ns2,:]
                X = R*np.cos(phi)
                Y = R*np.sin(phi)
                B = +np.real(cur_B[:d_upper_t.Ns2,:]*np.exp(1j*2*(phi-start_phi))*np.exp(1j*(np.deg2rad(phasing))))
                #BB = +np.real(Bn_v[:d_upper_t.Ns2,:]*np.exp(1j*2*(phi-start_phi)))
                mlab.mesh(X, Y, Z+off, scalars = B, colormap='RdBu', vmax = val, vmin = -val)
                #mlab.mesh(X, Y, Z-1, scalars = BB, colormap='RdBu', vmax = val, vmin = -val)
            #f.scene.camera.elevation(15);f.scene.render()
            #/home/srh112/code/NAMP_analysis/python/MARS_post_processing/
        filename_list.append('blah/test_%03d.png'%(ii))
        mlab.savefig(filename_list[-1], magnification = 1)
        mlab.clf()
    os.system('convert -delay {} -loop 0 {} {}'.format(20, ' '.join(filename_list), '3D_{}_{}.gif'.format('_'.join(plots), ul_phasing)))

#ffmpeg -loop 1 -framerate 20 -i test_%03d.png  -r 30 -t 15 -pix_fmt yuv420p out.mp4
os.system('ffmpeg -q scale 1 -r 20 -i -b 9600 {} movie.mp4'.format(' '.join([' '.join(filename_list) for i in range(5)])))

import numpy as np
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func

#Ideal cases
phasing = 0
n = 2
phase_machine_ntor = 0
s_surface = 0.94
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [4,0]
#reference_offset = [2,0]
sort_name = 'time_list'

file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_imp_grid/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04272_imp_grid/shot158115_04272_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_inc_MPID/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_imp_grid_0freq/shot158115_04780_post_processing_PEST.pickle'
file_name = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_imp_grid_0freq_B23_2/shot158115_04780_post_processing_PEST.pickle'
print file_name
reference_dB_kink = 'plasma'
a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'mean')
dBkink = dBres_dBkink.dBkink_calculations(a)
probe = dBres_dBkink.magnetic_probe(a,' 66M')
probe2 = dBres_dBkink.magnetic_probe(a,'Inner_pol')
#xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)

tmp_a = np.array(dBres.single_phasing_individual_harms(phasing,field='plasma'))
tmp_b = np.array(dBres.single_phasing_individual_harms(phasing,field='total'))
tmp_c = np.array(dBres.single_phasing_individual_harms(phasing,field='vacuum'))

# phases_res, vals_res = dBres.phasing_scan()
# fig, ax = pt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)
# ax.plot(phases, np.abs(vals))
# phases, vals = dBkink.phasing_scan(field = 'plasma')
# ax.plot(phases, np.abs(vals))
# phases, vals = dBkink.phasing_scan(field = 'vacuum')
# ax.plot(phases, np.abs(vals))
# fig.canvas.draw();fig.show()
m = 10
min_loc = np.argmin(np.abs(np.array(dBres.raw_data['res_m_vals'][0])-m))
rfa_title = 'RFA : m = {}, $\psi_N$ = {:.3f}'.format(dBkink.raw_data['plasma_max_mode_list_upper'][0], s_surface)
pitch_res_title = 'Pitch Resonant : m = {}, q = {}, $\psi_N$ = {:.3f}'.format(dBres.raw_data['res_m_vals'][0][min_loc],  dBres.raw_data['res_q_vals'][0][min_loc], dBres.raw_data['res_s_vals'][0][min_loc]**2,)

fig, ax = pt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = True)
for field, style in zip(['vacuum','plasma','total'], ['o-','x-','.-']):
    phases, vals = dBkink.phasing_scan(field = field,n_phases = 90)
    ax[0].plot(phases, np.abs(vals), style, label = field)
    vals = []
    for i in phases:
        vals.append(dBres.single_phasing_individual_harms(i,field=field)[0][min_loc])
    ax[1].plot(phases, np.abs(vals), style, label =field)
ax[-1].set_xlabel('Upper-lower I-coil phasing (deg)')
ax[0].set_title(rfa_title)
ax[1].set_title(pitch_res_title)
for i in ax: i.set_ylabel('Harmonic amplitude (G/kA)')
for i in ax: i.legend(loc='best')
for i in ax: i.grid()
ax[0].set_xlim([0,360])
fig.canvas.draw();fig.show()
field = 'plasma'


#Print out and plot the probe complex values
fig, ax = pt.subplots(nrows =3, sharex = True)
names = [' 66M', 'Inner_pol']
names = [' 66M', ' MPID1A', ' MPID1B']
print file_name
for name in names:
    ind = a.project_dict['details']['pickup_coils']['probe'].index(name)
    probe = dBres_dBkink.magnetic_probe(a,name)
    print ''
    print '###### {} #####'.format(name)
    tmp = a.project_dict['details']['pickup_coils']
    print 'R={:.3f}, Z={:.3f}m, l_probe={:.3f}m, inc={:.3f}rad, pol={}'.format(*[tmp[i][ind] for i in ['Rprobe', 'Zprobe', 'lprobe','tprobe','probe_type']])
    for field in ['plasma','vacuum','total']:
        for ul in ['upper','lower']:
            print '{}_{}='.format(field, ul), probe.raw_data['{}_probe_{}'.format(field, ul)][0]
    pu, pl = [probe.raw_data['plasma_probe_upper'], probe.raw_data['plasma_probe_lower']]
    ax[0].plot(np.linspace(0,360,200), np.abs(pu + pl * np.exp(1j*np.linspace(0,2.*np.pi,200))),label=name)
    ax[1].plot(np.linspace(0,360,200), np.rad2deg(np.angle(pu + pl * np.exp(1j*np.linspace(0,2.*np.pi,200))))%360,label=name)

#Print out and plot the res metric values
print ''
print '###### {} #####'.format('res_metric')
pitch_res_title = 'Pitch Resonant : m = {}, q = {}, $\psi_N$ = {:.3f}'.format(dBres.raw_data['res_m_vals'][0][min_loc],  dBres.raw_data['res_q_vals'][0][min_loc], dBres.raw_data['res_s_vals'][0][min_loc]**2,)
print pitch_res_title
for field in ['plasma','vacuum','total']:
    for ul in ['upper','lower']:
        print '{}_{}='.format(field, ul), dBres.raw_data['{}_res_{}'.format(field, ul)][0][min_loc]
field = 'vacuum'
u, l = dBres.raw_data['{}_res_{}'.format(field, 'upper')][0][min_loc], dBres.raw_data['{}_res_{}'.format(field, 'lower')][0][min_loc]
ax[2].plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))), label = 'res')

print ''
print '###### {} #####'.format('rfa_metric')
rfa_title = 'RFA : m = {}, $\psi_N$ = {:.3f}'.format(dBkink.raw_data['plasma_max_mode_list_upper'][0], s_surface)
print rfa_title
#Print out and plot the rfa metric
for field in ['plasma','vacuum','total']:
    for ul in ['upper','lower']:
        print '{}_{} ='.format(field, ul), dBkink.raw_data['{}_kink_harm_{}'.format(field, ul)][0]
field = 'plasma'
u, l = dBkink.raw_data['{}_kink_harm_{}'.format(field, 'upper')][0], dBkink.raw_data['{}_kink_harm_{}'.format(field, 'lower')][0]
ax[2].plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))), label = 'rfa')
ax[-1].set_xlim([0,360])
ax[0].set_ylabel('mod(B) G/kA')
ax[1].set_ylabel('arg(B) deg')
ax[-1].set_xlabel('phasing')
leg = ax[2].legend(loc = 'best')
leg.draw_frame(False)
leg = ax[0].legend(loc = 'best')
leg.draw_frame(False)
leg = ax[1].legend(loc = 'best')
leg.draw_frame(False)
fig.suptitle(file_name.replace('_','-'))
fig.canvas.draw()
fig.canvas.draw(); fig.show()


#print the answers
for field in ['plasma','vacuum','total']:
    for ul in ['upper','lower']:
        print field, ul, dBres.raw_data['{}_res_{}'.format(field, ul)][0][min_loc]
vu = -0.4510498+0.1534607987j
vl = -0.360474-0.0030321j
pu =  -0.42832265530391112+0.12657972675744139j
pl =  -0.44415741157986277-0.083120258183512041j
tu =  -0.87937245682739207+0.28004052546419594j
tl =  -0.80463181197766065-0.086152446592039494j
fig, ax = pt.subplots()
for u, l in zip([vu,pu,tu],[vl,pl,tl]):
    ax.plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))))
fig.canvas.draw()

pu =0.567318252715+0.3578529234j
pl =0.265773583817-0.443374003173j
vu =-0.572224181872-0.35443269971j
vl =-0.268458170049+0.447234420123j
tu =-0.00490592915767+0.00342022368969j
tl =-0.00268458623213+0.0038604169496j
fig, ax = pt.subplots()
for u, l in zip([vu,pu,tu],[vl,pl,tl]):
    ax.plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))))
fig.canvas.draw()



#For Carlos email 19July2014
pl= 0.29259901-0.25436739j
pu= -0.27163093-0.20034227j
fig, ax = pt.subplots(nrows =3, sharex = True)
ax[0].plot(np.linspace(0,360,100), np.abs(pu + pl * np.exp(1j*np.linspace(0,2.*np.pi,100))),label='66M')
ax[1].plot(np.linspace(0,360,100), np.rad2deg(np.angle(pu + pl * np.exp(1j*np.linspace(0,2.*np.pi,100))))%360,label='66M')
pl=-0.12666662-0.26925804j
pu= 0.10206248-0.32727205j
ax[0].plot(np.linspace(0,360,100), np.abs(pu + pl * np.exp(1j*np.linspace(0,2.*np.pi,100))), label='HFS')
ax[1].plot(np.linspace(0,360,100), np.rad2deg(np.angle(pu + pl * np.exp(1j*np.linspace(0,2.*np.pi,100))))%360,label='HFS')
ax[0].set_ylabel('mod(B) G/kA')
ax[1].set_ylabel('arg(B) deg')
ax[-1].set_xlabel('phasing')
ax[1].set_ylim([0,360])
ax[0].set_xlim([0,360])
pu =0.567318252715+0.3578529234j
pl =0.265773583817-0.443374003173j
vu =-0.572224181872-0.35443269971j
vl =-0.268458170049+0.447234420123j
tu =-0.00490592915767+0.00342022368969j
tl =-0.00268458623213+0.0038604169496j
#for u, l in zip([vu,pu,tu],[vl,pl,tl]):
for u, l in zip([vu],[vl]):
    ax[2].plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))),label='res')
vu = -0.4510498+0.1534607987j
vl = -0.360474-0.0030321j
pu =  -0.42832265530391112+0.12657972675744139j
pl =  -0.44415741157986277-0.083120258183512041j
tu =  -0.87937245682739207+0.28004052546419594j
tl =  -0.80463181197766065-0.086152446592039494j
#for u, l in zip([vu,pu,tu],[vl,pl,tl]):
for u, l in zip([pu],[pl]):
    ax[2].plot(np.linspace(0,360,100), np.abs(u + l * np.exp(1j*np.linspace(0,2.*np.pi,100))),label='rfa (plas)')
fig.canvas.draw()
leg = ax[2].legend(loc = 'best')
leg.draw_frame(False)
leg = ax[0].legend(loc = 'best')
leg.draw_frame(False)
leg = ax[1].legend(loc = 'best')
leg.draw_frame(False)
fig.canvas.draw()
