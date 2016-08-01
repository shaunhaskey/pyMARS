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
import PythonMARS_funcs as pyMARS
N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I); facn = 1.0

I0EXP = 1.0e+3*0.863 #PMZ real
I0EXP = I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

'''

158103.03796 (Reference) - lots
161541.02423 (Low beta) - Nothing?
161549.02257 (High beta) - Nothing
161205.03215 (L-mode) - betaN ramp and single available
161198.03550 (High collisionality) - few different ones here
159346.02120 (High q95) - single : shot159346_02120_betaN_ramp_carlos_q95


'''
n = 2
def return_data(directory):
    #directory ='/home/shaskey/NAMP_datafiles/mars/146382_thetac_020/qmult1.000/exp1.000/marsrun/RUNrfa.p'
    d = data(directory, I0EXP=I0EXP)
    d.get_PEST(facn = facn)
    file_name = directory + '/PROFEQ.OUT'
    qn, sq, q, s, mq = return_q_profile(d.mk,file_name=file_name, n=n)
    d.q_qn = qn
    d.q_sq = sq
    d.q = q
    d.q_s = s
    d.q_mq = mq
    return d

import cPickle as pickle
single = True
#{Total, vacuum}, {upper, lower}, q-profile, s_vals, m_vals
job_name = 'shot156746_02113_single_test2_mar'
mars_dir = '/home/shaskey/NAMP_datafiles/mars/'
pickle_file = '{}{}/{}_post_processing_PEST.pickle'.format(mars_dir, job_name, job_name)
a = pickle.load(file(pickle_file,'r'))
return_list = {}
for up_low in ['lower','upper']:
    for tot_vac in ['plasma','vacuum']:
        rel_dir = a['sims'][1]['dir_dict']['mars_{}_{}_dir'.format(up_low, tot_vac)]
        print up_low, tot_vac, rel_dir
        return_list['{}_{}'.format(tot_vac.replace('plasma','total'), up_low)] = return_data(rel_dir.replace('/u/haskeysr/','/home/shaskey/NAMP_datafiles/'))

from scipy.io import netcdf
f = netcdf.netcdf_file('test.nc','w')
for ii, i in enumerate(return_list.keys()):
    z = return_list[i]
    if ii==0:
        s = z.ss
        m = z.Mm[:,0]

        f.createDimension('s', len(s))
        f_s = f.createVariable('s','d',('s',))
        f_s[:] = s

        f.createDimension('q_s', len(z.q_s))
        f_qs = f.createVariable('q_s','d',('q_s',))
        f_qs[:] = z.q_s

        f.createDimension('q', len(z.q))
        f_q = f.createVariable('q','d',('q',))
        f_q[:] = z.q

        f.createDimension('m', len(m))
        f_m = f.createVariable('m','d',('m',))
        f_m[:] =m 

    BnPEST = z.BnPEST
    f_BnPEST_real = f.createVariable('{}_real'.format(i),'d',('s','m',))
    f_BnPEST_imag = f.createVariable('{}_imag'.format(i),'d',('s','m',))
    f_BnPEST_real[:] =np.real(BnPEST)
    f_BnPEST_imag[:] =np.imag(BnPEST)
f.close()


1/0

    
    # fig,ax = pt.subplots()
    # color_plot = d.plot_BnPEST(ax, n=n, inc_contours = 1)
    # color_plot.set_clim([0,1.5])
    # color_plot.set_clim([0,4.5])
    # cbar = pt.colorbar(color_plot, ax = ax)
    # ax.set_xlabel('m')
    # ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    # cbar.ax.set_ylabel('G/kA')
    # fig.canvas.draw(); fig.show()
