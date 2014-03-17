import  results_class
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt
import copy
import cPickle as pickle
import PythonMARS_funcs as pyMARS
import multiprocessing
import itertools, os
import pyMARS.generic_funcs as gen_funcs

def plot_scan(args):
    #print 'hello'                                                                                                           
    print 'hellop plots', os.getpid(), os.getcwd()
    replacement_kwargs = {'savefig.dpi':150}
    valid_sim, im_name, I0EXP, facn, subplot_plot, n, inc_contours, clim, fig_title = copy.deepcopy(args)
    combs = itertools.product(['upper','lower'],['plasma','vacuum'])
    simuls = {}
    for loc, typ in combs:
        tmp = results_class.data(valid_sim['dir_dict']['mars_{}_{}_dir'.format(loc, typ)], I0EXP = I0EXP, getpest = True)
        tmp.get_VPLASMA()
        print '---',  os.getcwd(), os.getcwd(), np.sum(np.abs((tmp.Bn))), np.sum(np.abs((tmp.BnPEST)))
        #print '---',  os.getpid(), os.getcwd(), np.sum(np.abs((tmp.Bn)))
        #tmp.get_PEST(facn=facn)
        simuls['{}_{}'.format(loc,typ)] = copy.deepcopy(tmp)
    #for i in simuls.keys():
    #    print '!!',os.getpid(), i, 
    #for i in simuls.keys(): simuls[i].get_PEST(facn = facn)
    #for i in simuls.keys():
    #    print '!!', os.getpid(), i, np.sum(np.abs((simuls[i].BnPEST)))
    phasings = [0,90,180,270]
    fig2,ax2 = pt.subplots(ncols = 3, sharex =True, sharey = True)
    gen_funcs.setup_publication_image(fig2, height_prop = 1./1.618, single_col = False, replacement_kwargs = replacement_kwargs)

    fig_Vn_Bn,ax_Vn_Bn = pt.subplots(ncols = 3, sharex = True, sharey = True)
    gen_funcs.setup_publication_image(fig_Vn_Bn, single_col = False, replacement_kwargs = replacement_kwargs)
    ax_Vn_Bn[0].set_ylabel('R (m)')
    for i_ax in ax_Vn_Bn: i_ax.set_xlabel('Z (m)')

    for ind, subplot_plot in enumerate(['vac','plasma','total']):
        fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =True, sharey = True)
        fig_Bn,ax_Bn = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
        gen_funcs.setup_publication_image(fig_Bn, height_prop = 1., single_col = True, replacement_kwargs = replacement_kwargs)
        ax_Bn[0,0].set_ylabel('R (m)')
        ax_Bn[1,0].set_ylabel('R (m)')
        ax_Bn[1,0].set_xlabel('Z (m)')
        ax_Bn[1,1].set_xlabel('Z (m)')
        ax_Bn = ax_Bn.flatten()

        if subplot_plot=='plasma':
            fig_Vn,ax_Vn = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True)
            gen_funcs.setup_publication_image(fig_Vn, height_prop = 1., single_col = True, replacement_kwargs = replacement_kwargs)
            ax_Vn[0,0].set_ylabel('R (m)')
            ax_Vn[1,0].set_ylabel('R (m)')
            ax_Vn[1,0].set_xlabel('Z (m)')
            ax_Vn[1,1].set_xlabel('Z (m)')
            ax_Vn = ax_Vn.flatten()

        gen_funcs.setup_publication_image(fig, height_prop = 1./1.618, single_col = False, replacement_kwargs = replacement_kwargs)
        ax[0,0].set_ylabel(r'$\sqrt{\psi_N}$')
        ax[1,0].set_ylabel(r'$\sqrt{\psi_N}$')
        ax[1,0].set_xlabel('m')
        ax[1,1].set_xlabel('m')
        ax = ax.flatten()
        color_plots = []
        for i, phasing in enumerate(phasings):
            print phasing
            combined = copy.deepcopy(simuls['upper_plasma'])
            #Combine the upper and lower data with the appropriate phasing
            R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = results_class.combine_data(simuls['upper_plasma'], simuls['lower_plasma'], phasing)
            R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = results_class.combine_data(simuls['upper_vacuum'], simuls['lower_vacuum'], phasing)
            #if i==0: print 'totals are ', os.getpid(), fig_title, np.sum(np.abs(BnPEST_t)), np.sum(np.abs(BnPEST_v))
            #print 'obtained Vn'
            #Choose which plot to create
            cmap = 'spectral'
            if subplot_plot=='total':
                combined.BnPEST = +BnPEST_t
                combined.Bn = +Bn_t
            elif subplot_plot =='vac':
                combined.BnPEST = +BnPEST_v
                combined.Bn = +Bn_v
            elif subplot_plot == 'plasma':
                combined.BnPEST = BnPEST_t - BnPEST_v
                combined.Bn = Bn_t - Bn_v
            combined.BnPEST[np.isnan(combined.BnPEST)] = 0
            combined.Bn[np.isnan(combined.Bn)] = 0
            #print 'asdf total ', phasing, np.sum(np.abs(combined.BnPEST))
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours, cmap = cmap))

            combined.plot_Bn_surface(ax = ax_Bn[i], multiplier = 0.0074)
            ax_Bn[i].set_title(r'real,imag Bn $\Delta \phi_{ul} = %d^o$'%(phasing))
            if subplot_plot == 'plasma':
                Vn = results_class.combine_fields_displacement([simuls['lower_plasma'],simuls['lower_vacuum'],simuls['upper_plasma'],simuls['lower_vacuum']] , 'Vn', theta=np.deg2rad(phasing), field_type = 'plas')
                combined.Vn = Vn
                #combined.plot_Vn_surface(ax = ax_Vn[i], multiplier = 10)
                combined.plot_Vn_surface(ax = ax_Vn[i], multiplier = 20)
                ax_Vn[i].set_title(r'real imag disp $\Delta \phi_{ul} = %d^o$'%(phasing))
                if phasing == 0:
                    combined.plot_Vn_surface(ax = ax_Vn_Bn[0], multiplier = 20)
                    ax_Vn_Bn[0].set_title(r'real imag disp $\Delta \phi_{ul} = %d^o$'%(phasing), fontsize = 7)
                    combined.plot_Bn_surface(ax = ax_Vn_Bn[1], multiplier = 0.0074)
                    ax_Vn_Bn[1].set_title(r'real imag BnPlasma $\Delta \phi_{ul} = %d^o$'%(phasing), fontsize = 7)
            if subplot_plot == 'total' and phasing == 0:
                combined.plot_Bn_surface(ax = ax_Vn_Bn[2], multiplier = 0.0074)
                ax_Vn_Bn[2].set_title(r'real imag BnTotal $\Delta \phi_{ul} = %d^o$'%(phasing), fontsize = 7)
            if phasing==0:
                tmp_clr = combined.plot_BnPEST(ax2[ind], n=n, inc_contours = inc_contours, cmap = cmap)
                ax2[ind].set_title('%s $\Delta \phi_{ul} = %d^o$'%(subplot_plot, phasing))
                tmp_clr.set_clim(clim)
            ax[i].set_title(r'$\Delta \phi_{ul} = %d^o$'%(phasing))
            color_plots[-1].set_clim(clim)

        if subplot_plot == 'plasma':
            for i_ax in ax_Vn:
                i_ax.grid()
                #ax[1].axhline(0.6*np.min(plas_z[-1,:]))
                i_ax.set_ylim([-1.25,1.25])
                i_ax.set_xlim([1.0,2.5])
            ax_Vn[0].set_xticks(ax_Vn[0].get_xticks()[::2])
            #fig_Vn.tight_layout(pad = 0.3)
            fig_Vn.suptitle('{} {}'.format(fig_title,'displacement'))
            fig_Vn.savefig('{}_{}.png'.format(im_name, 'displacement'))
            fig_Vn.clf()

        for i_ax in ax_Bn:
            i_ax.grid()
            #ax[1].axhline(0.6*np.min(plas_z[-1,:]))
            i_ax.set_ylim([-1.25,1.25])
            i_ax.set_xlim([1.0,2.5])
        ax_Bn[0].set_xticks(ax_Bn[0].get_xticks()[::2])
        #fig_Vn.tight_layout(pad = 0.3)
        fig_Bn.suptitle('{} {} {}'.format(fig_title,'Bn', subplot_plot))
        fig_Bn.savefig('{}_{}.png'.format(im_name, 'Bn'+subplot_plot))
        fig_Bn.clf()

        ax[0].set_xlim([0,25])
        ax[0].set_ylim([0.4,1])
        fig.suptitle('{} {}'.format(fig_title,subplot_plot))
        #pt.colorbar(ax[-1],ax = ax.flatten().tolist())
        fig.savefig('{}_{}.png'.format(im_name, subplot_plot))
        fig.clf()
        #fig.close()

    ax2[0].set_xlim([0,25])
    ax2[0].set_ylim([0.4,1])
    ax2[0].set_ylabel(r'$\sqrt{\psi_N}$')
    for i in ax2: i.set_xlabel('m')
    fig2.suptitle(fig_title)
    #pt.colorbar(ax[-1],ax = ax.flatten().tolist())
    fig2.savefig('{}_all.png'.format(im_name))
    fig2.clf()
    #fig.close()
    for i_ax in ax_Vn_Bn:
        i_ax.grid()
        i_ax.set_ylim([-1.25,1.25])
        i_ax.set_xlim([1.0,2.5])
    ax_Vn_Bn[0].set_xticks(ax_Vn_Bn[0].get_xticks()[::2])
    #fig_Vn.tight_layout(pad = 0.3)
    fig_Vn_Bn.suptitle('{} {}'.format(fig_title, subplot_plot))
    fig_Vn_Bn.savefig('{}_{}.png'.format(im_name, 'BnDisp' + subplot_plot), fontsize = 7)
    fig_Vn_Bn.clf()
