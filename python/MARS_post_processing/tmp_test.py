import  results_class
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt
import copy
import cPickle as pickle
import PythonMARS_funcs as pyMARS
import multiprocessing
import itertools, os
def plot_scan(args):
    #print 'hello'                                                                                                           
    print 'hello plots', os.getpid(), os.getcwd()
    valid_sim, im_name, I0EXP, facn, subplot_plot, n, inc_contours, clim, fig_title = copy.deepcopy(args)
    combs = itertools.product(['upper','lower'],['plasma','vacuum'])
    simuls = {}
    for loc, typ in combs:
        tmp = results_class.data(valid_sim['dir_dict']['mars_{}_{}_dir'.format(loc, typ)], I0EXP = I0EXP, getpest = True)
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
    fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex =True, sharey = True)
    ax[0,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1,0].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
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
        if i==0: print 'totals are ', os.getpid(), fig_title, np.sum(np.abs(BnPEST_t)), np.sum(np.abs(BnPEST_v))
        #Choose which plot to create
        cmap = 'spectral'
        if subplot_plot=='total':
            combined.BnPEST = BnPEST_t
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours, cmap = cmap))
            #color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = 1, increase_grid_BnPEST=1, gauss_filter = [0,0.05]))
        elif subplot_plot=='vac':
            combined.BnPEST = BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours, cmap = cmap))
        elif subplot_plot=='plasma':
            combined.BnPEST = BnPEST_t - BnPEST_v
            color_plots.append(combined.plot_BnPEST(ax[i], n=n, inc_contours = inc_contours, cmap = cmap))

        ax[i].set_title(r'$\Delta \phi_{ul} = %d^o$'%(phasing),fontsize = 18)
        color_plots[-1].set_clim()
    ax[0].set_xlim([0,25])
    ax[0].set_ylim([0.4,1])
    fig.suptitle(fig_title)
    fig.savefig('{}'.format(im_name))
    fig.clf()
    #fig.close()
