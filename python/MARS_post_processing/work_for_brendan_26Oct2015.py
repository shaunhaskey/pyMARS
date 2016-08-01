'''
SH : Nov 21 2012
This is useful for creating PEST plot images
It can make an animation and introduce different phasings
'''

import cPickle as pickle
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
facn = 1.0 #
sim_num = 10 # 13, 5, 10
#{Total, vacuum}, {upper, lower}, q-profile, s_vals, m_vals
job_name = 'shot158103_03796_betaN_ramp_carlos_2'
#job_name = 'shot156746_02113_single_test2_mar'
mars_dir = '/home/shaskey/NAMP_datafiles/mars/'
mars_dir = '/u/haskeysr/mars/'

'''

158103.03796 (Reference) - lots
161541.02423 (Low beta) - Nothing?
161549.02257 (High beta) - Nothing
161205.03215 (L-mode) - betaN ramp and single available
161198.03550 (High collisionality) - few different ones here
159346.02120 (High q95) - single : shot159346_02120_betaN_ramp_carlos_q95



'shot158103_03796_betaN_ramp_carlos_2' for the following three
158103.03796 (Reference) - betaN ramp case #10
161541.02423 (Low beta) - Uses the Reference betaN ramp with case #5
161549.02257 (High beta) - Uses the Reference betaN ramp with case #13

shot161205_03215_betaN_ramp_carlos_lmode2, or shot161205_03215_betaN_ramp_carlos_lmode
161205.03215 (L-mode) - betaN ramp case #4

shot161198_03550_betaN_ramp_carlos_hicol, shot161198_03550_betaN_ramp_carlos_hicol2
161198.03550 (High collisionality) - betaN ramp case #10

shot159346_02120_betaN_ramp_carlos_q95
159346.02120 (High q95) - betaN ramp case #17
'''
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

pickle_file = '{}{}/{}_post_processing_PEST.pickle'.format(mars_dir, job_name, job_name)
a = pickle.load(file(pickle_file,'r'))

sims = np.array(a['sims'].keys())
betan = np.array([a['sims'][i]['BETAN'] for i in sims])
li = np.array([a['sims'][i]['LI'] for i in sims])
ordered = np.argsort(betan/li)
sim_num2 = sims[ordered[sim_num]]
print sim_num2

return_list = {}
for up_low in ['lower','upper']:
    for tot_vac in ['plasma','vacuum']:
        rel_dir = a['sims'][sim_num2]['dir_dict']['mars_{}_{}_dir'.format(up_low, tot_vac)]
        print up_low, tot_vac, rel_dir
        if False:
            return_list['{}_{}'.format(tot_vac.replace('plasma','total'), up_low)] = return_data(rel_dir.replace('/u/haskeysr/','/home/shaskey/NAMP_datafiles/'))
        else:
            return_list['{}_{}'.format(tot_vac.replace('plasma','total'), up_low)] = return_data(rel_dir)

from scipy.io import netcdf
f = netcdf.netcdf_file('{}_{}_{}.nc'.format(job_name,sim_num,sim_num2),'w')
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

    BnPEST = z.BnPEST_SURF
    f_BnPEST_real = f.createVariable('{}_real'.format(i),'d',('s','m',))
    f_BnPEST_imag = f.createVariable('{}_imag'.format(i),'d',('s','m',))
    f_BnPEST_real[:] =np.real(BnPEST)
    f_BnPEST_imag[:] =np.imag(BnPEST)
f.close()


fig,ax = pt.subplots(nrows = 2, ncols = 2, sharex = True, sharey=True)
ax = ax.flatten()
count = 0
for loc in ['upper','lower']:
    for field in ['total','vacuum']:
        d = return_list['{}_{}'.format(field, loc)]
        color_plot = d.plot_BnPEST(ax[count], n=n, inc_contours = 1)
        color_plot.set_clim([0,1.5])
        color_plot.set_clim([0,2.5])
        cbar = pt.colorbar(color_plot, ax = ax[count])
        ax[count].set_xlabel('m')
        ax[count].set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
        ax[count].set_title('{}_{}'.format(loc,field))
        cbar.ax.set_ylabel('G/kA')
        count += 1 
fig.canvas.draw(); fig.show()
