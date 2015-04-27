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

#I = np.array([1.,0.5,-0.5,1,0.5,-0.5])
I0EXP = I0EXP_calc(N,n,I); facn = 1.0

I0EXP = I0EXP_calc_real(n,I)
facn = 1.0 #WHAT IS THIS WEIRD CORRECTION FACTOR?

#various simulation directories to get the components
#base_dir = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_single_raffi3_C-coil/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
#base_dir = '/home/srh112/NAMP_datafiles/mars/shot158115_04702_single_raffi3_C-coil/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
#base_dir = '/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/'
base_dir = '/home/srh112/NAMP_datafiles/mars/shot158115_04780/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
#base_dir = '/home/srh112/NAMP_datafiles/mars/shot158115_04780_n4/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
#base_dir = '/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/qmult1.000/exp1.000/marsrun/'
base_dir = '/home/srh112/NAMP_datafiles/mars/shot158115_04702_single_raffi_test/qmult1.000/exp1.000/RES-100000000.0000_ROTE-100.0000/'
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
phasings = range(0,360,5)
BnPEST_t_list = []
BnPEST_v_list = []
for i, phasing in enumerate(phasings):
    print phasing
    R_t, Z_t, B1_t, B2_t, B3_t, Bn_t, BMn_t, BnPEST_t = combine_data(d_upper_t, d_lower_t, phasing)
    R_v, Z_v, B1_v, B2_v, B3_v, Bn_v, BMn_v, BnPEST_v = combine_data(d_upper_v, d_lower_v, phasing)
    BnPEST_t_list.append(+BnPEST_t)
    BnPEST_v_list.append(+BnPEST_v)

dat = np.abs(BnPEST_t_list[0] - BnPEST_v[0])
U, s, V = np.linalg.svd(dat, full_matrices = False)
#C = np.zeros((U.shape[1], V.shape[0]), dtype=complex)
#C[:s.shape[0],:s.shape[0]] = +s
C = np.diag(+s)
output = np.real(np.dot(U, np.dot(C,V)))

fig, ax = pt.subplots(nrows = 3, sharex = True, sharey = True)
tmp1 = ax[0].imshow(dat, aspect = 'auto')
tmp = ax[1].imshow(output, aspect='auto')
tmp.set_clim(tmp1.get_clim())
tmp = ax[2].imshow(np.abs(output-dat), aspect='auto')
tmp.set_clim(tmp1.get_clim())
fig.canvas.draw(); fig.show()


fig, ax = pt.subplots(nrows = 5, ncols = 5, sharex = True, sharey = True)
ax = ax.flatten()
for tmp_ax, i in zip(ax, range(25)):
    C = +s*0
    C[i] = +s[i]
    C = np.diag(C)
    output = np.real(np.dot(U, np.dot(C,V)))
    tmp = tmp_ax.imshow(output, aspect='auto')
    tmp.set_clim([0,1.])
fig.canvas.draw(); fig.show()

inp_data = np.zeros((len(BnPEST_t_list), len(BnPEST_v.flatten())),dtype = complex)
for i in range(len(BnPEST_t_list)):
    inp_data[i,:]=BnPEST_t_list[i].flatten() - BnPEST_v_list[i].flatten()



file_name = d_upper_t.directory + '/PROFEQ.OUT'
import results_class as results
qn, sq, q, s_new, mq = results.return_q_profile(d_upper_t.mk,file_name=file_name, n = 2)
#if not sqrt_flux: s = s**2
#        if not sqrt_flux: sq = sq**2
U, s, V = np.linalg.svd(inp_data, full_matrices = False)
fig, ax = pt.subplots(nrows = 3, ncols = 3, sharex = True, sharey = True)
fig2, ax2 = pt.subplots(nrows = 3, ncols = 3, sharex = True, sharey = True)
fig3, ax3 = pt.subplots(nrows = 2, ncols = 4, sharex = True, sharey = True)
ax = ax.flatten()
ax2 = ax2.flatten()
for tmp_ax, tmp_ax2, i in zip(ax, ax2, range(4)):
    C = +s*0
    C[i] = +s[i]
    C = np.diag(C)
    output = np.abs(np.dot(U, np.dot(C,V)))
    #tmp = tmp_ax.imshow(output, aspect='auto')
    cmap = 'RdBu'
    tmp_plot_mk = d_upper_t.mk.flatten()
    ss_plas_edge = np.argmin(np.abs(d_upper_t.ss-0.999))
    s_ax = d_upper_t.ss.flatten()#[:ss_plas_edge]
    tmp = tmp_ax.pcolormesh(tmp_plot_mk,s_ax, np.abs(V[i,:]).reshape(BnPEST_t_list[0].shape)*s[i], cmap='hot', shading='gouraud')
    tmp2 = tmp_ax2.pcolormesh(tmp_plot_mk,s_ax, np.angle(V[i,:]).reshape(BnPEST_t_list[0].shape), cmap='RdBu', shading='gouraud')

    tmp3 = ax3[0,i].pcolormesh(tmp_plot_mk,s_ax, np.abs(V[i,:]).reshape(BnPEST_t_list[0].shape)*s[i], cmap='hot')#, shading='gouraud')
    tmp4 = ax3[1,i].pcolormesh(tmp_plot_mk,s_ax, np.angle(V[i,:]).reshape(BnPEST_t_list[0].shape), cmap='RdBu')#, shading='gouraud')
    #tmp = tmp_ax.imshow(np.abs(V[i,:]).reshape(BnPEST_t_list[0].shape)*s[i], aspect='auto')
    #tmp2 = tmp_ax2.imshow(np.angle(V[i,:]).reshape(BnPEST_t_list[0].shape), aspect='auto',cmap = 'RdBu')

    tmp_ax.plot(q*n,s_new,'w--') 
    tmp_ax.plot(mq,sq,'w+',markersize = 15)
    tmp_ax2.plot(q*n,s_new,'k--') 
    tmp_ax2.plot(mq,sq,'k+',markersize = 15)


    ax3[0,i].plot(q*n,s_new,'w--') 
    ax3[0,i].plot(mq,sq,'w+',markersize = 15)
    ax3[1,i].plot(q*n,s_new,'k--') 
    ax3[1,i].plot(mq,sq,'k+',markersize = 15)

    tmp.set_clim([0,1])
    tmp2.set_clim([-np.pi,np.pi])

    tmp3.set_clim([0,1])
    tmp4.set_clim([-np.pi,np.pi])
ax[0].set_ylim([0,1])
ax2[0].set_ylim([0,1])
ax3[0,0].set_ylim([0,1])
fig.canvas.draw(); fig.show()
fig2.canvas.draw(); fig2.show()
fig3.canvas.draw(); fig2.show()
figure();plot(np.abs(U[:,0])); plot(np.abs(U[:,1])); plot(np.abs(U[:,2]));grid()

# fig, ax = pt.subplots(ncols = 4, sharex = True, sharey = True)
# #ax[0].imshow(np.abs(U))
# for i in range(4):
#     ax[0].imshow(V)
    
# fig.canvas.draw(); fig.show()
