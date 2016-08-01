import pyMARS.results_class as res
import pyMARS.RZfuncs as RZ_funcs
import matplotlib.pyplot as pt
import numpy as np
import copy
I = np.cos(np.deg2rad(np.arange(0,360,60)))
I0EXP = RZ_funcs.I0EXP_calc_real(1,I,discrete_pts=1000)

base_dirs = ['/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp1.200/marsrun/',
             '/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp0.700/marsrun/']
labels = ['high_beta', 'low_beta']
data = []
data_t = []
data_p = []
data_v = []
data_pest_t = []
data_pest_p = []
data_pest_v = []
fname_stub = '/u/haskeysr/files_for_josh2/'
data_obj = []
for base_dir, lab in zip(base_dirs,labels):
    dat_tot = res.data(base_dir + '/RUNrfa.p',getpest=True, I0EXP=I0EXP)
    dat_vac = res.data(base_dir + '/RUNrfa.vac',getpest=True, I0EXP=I0EXP)
    np.savetxt('{}/{}_Bn_tot_re.txt'.format(fname_stub, lab),np.real(dat_tot.Bn))
    np.savetxt('{}/{}_Bn_vac_re.txt'.format(fname_stub, lab),np.real(dat_vac.Bn))
    np.savetxt('{}/{}_Bn_tot.txt'.format(fname_stub, lab),dat_tot.Bn)
    np.savetxt('{}/{}_Bn_vac.txt'.format(fname_stub, lab),dat_vac.Bn)
    np.savetxt('{}/{}_Bn_tot_im.txt'.format(fname_stub, lab),np.imag(dat_tot.Bn))
    np.savetxt('{}/{}_Bn_vac_im.txt'.format(fname_stub, lab),np.imag(dat_vac.Bn))
    np.savetxt('{}/{}_R.txt'.format(fname_stub, lab),dat_tot.R*dat_tot.R0EXP)
    np.savetxt('{}/{}_Z.txt'.format(fname_stub, lab),dat_tot.Z*dat_tot.R0EXP)
    np.savetxt('{}/{}_m.txt'.format(fname_stub, lab),dat_tot.mk.flatten())
    np.savetxt('{}/{}_sqrtPsiN.txt'.format(fname_stub, lab),dat_tot.ss.flatten())
    np.savetxt('{}/{}_BnPEST_tot_re.txt'.format(fname_stub, lab),np.real(dat_tot.BnPEST_SURF))
    np.savetxt('{}/{}_BnPEST_vac_re.txt'.format(fname_stub, lab),np.real(dat_vac.BnPEST_SURF))
    np.savetxt('{}/{}_BnPEST_tot_im.txt'.format(fname_stub, lab),np.imag(dat_tot.BnPEST_SURF))
    np.savetxt('{}/{}_BnPEST_vac_im.txt'.format(fname_stub, lab),np.imag(dat_vac.BnPEST_SURF))
    np.savetxt('{}/{}_BnPEST_tot.txt'.format(fname_stub, lab),dat_tot.BnPEST_SURF)
    np.savetxt('{}/{}_BnPEST_vac.txt'.format(fname_stub, lab),dat_vac.BnPEST_SURF)
    file_name = dat_tot.directory + '/PROFEQ.OUT'
    n=1
    qn, sq, q, s, mq = res.return_q_profile(dat_tot.mk,file_name=file_name, n=n)
    np.savetxt('{}/{}_qn.txt'.format(fname_stub, lab),q*n)
    np.savetxt('{}/{}_qn_s.txt'.format(fname_stub, lab),s)
    np.savetxt('{}/{}_qn_res.txt'.format(fname_stub, lab),mq)
    np.savetxt('{}/{}_qn_s_res.txt'.format(fname_stub, lab),sq)

    data_t.append(dat_tot.Bn)
    data_p.append(dat_tot.Bn - dat_vac.Bn)
    data_v.append(dat_vac.Bn)
    data_pest_p.append(dat_tot.BnPEST - dat_vac.BnPEST)
    data_pest_v.append(dat_vac.BnPEST)
    data_pest_t.append(dat_tot.BnPEST)
    data_obj.append(copy.deepcopy(dat_tot))

fig,axes = pt.subplots(ncols = 2, sharex = True, sharey = True)
cplots = []
clim = [-10,10]
plot_kwargs = {'rasterized':True}
#plot_kwargs = {}
plot_type = 'vacuum'
if plot_type == 'plasma':
    data_to_use = data_p
    title = 'plasma-only'
elif plot_type == 'total':
    data_to_use = data_t
    title = 'plasma-plus-vacuum'
elif plot_type == 'vacuum':
    data_to_use = data_v
    title = 'vacuum-only'
else:
    raise ValueError()
for dat,dat_obj_tmp, ax in zip(data_to_use, data_obj, axes):
    a = dat_obj_tmp.plot_Bn(np.real(dat), axis = ax, end_surface = dat_tot.Ns1, cmap = "RdBu", plot_kwargs = plot_kwargs)
    #grid_R = dat_obj.R*dat_tot.R0EXP
    #grid_Z = dat_obj.Z*dat_tot.R0EXP
    #print 'sh_mod', np.max(grid_R), np.min(grid_R), np.max(grid_Z), np.min(grid_Z)
    #start_surface = 0; end_surface = dat_obj.Ns1; skip = 1; plot_quantity = np.real(dat)
    #color_plot = ax.pcolormesh(grid_R[start_surface:end_surface:skip,:], grid_Z[start_surface:end_surface:skip,:], plot_quantity[start_surface:end_surface:skip,:],cmap = 'RdBu', **plot_kwargs)
    #color_plot.set_clim(clim)
    cplots.append(a)
    cplots[-1].set_clim(clim)
axes[0].set_xlim([1,2.4])
axes[0].set_ylim([-1.25,1.25])
cbar = pt.colorbar(cplots[-1], ax = axes.tolist())
cbar.set_label('Re(Bn) (G/kA) : {}'.format(title))
for i in axes:i.set_xlabel('R (m)')
axes[0].set_ylabel('Z (m)')
print ' saving figure'
fig.savefig('/u/haskeysr/josh_real_space_{}_re_Bn.pdf'.format(title))


fig,axes = pt.subplots(ncols = 2, sharex = True, sharey = True)
cplots = []
clim = [0,10]
plot_kwargs = {'rasterized':True}
for dat,dat_obj_tmp, ax in zip(data_to_use, data_obj, axes):
    a = dat_obj_tmp.plot_Bn(np.abs(dat), axis = ax, end_surface = dat_tot.Ns1, cmap = "hot", plot_kwargs = plot_kwargs)
    #grid_R = dat_obj.R*dat_tot.R0EXP
    #grid_Z = dat_obj.Z*dat_tot.R0EXP
    #print 'sh_mod', np.max(grid_R), np.min(grid_R), np.max(grid_Z), np.min(grid_Z)
    #start_surface = 0; end_surface = dat_obj.Ns1; skip = 1; plot_quantity = np.real(dat)
    #color_plot = ax.pcolormesh(grid_R[start_surface:end_surface:skip,:], grid_Z[start_surface:end_surface:skip,:], plot_quantity[start_surface:end_surface:skip,:],cmap = 'RdBu', **plot_kwargs)
    #color_plot.set_clim(clim)
    cplots.append(a)
    cplots[-1].set_clim(clim)
axes[0].set_xlim([1,2.4])
axes[0].set_ylim([-1.25,1.25])
cbar = pt.colorbar(cplots[-1], ax = axes.tolist())
cbar.set_label('abs(Bn) (G/kA) : {}'.format(title))
for i in axes:i.set_xlabel('R (m)')
axes[0].set_ylabel('Z (m)')
print ' saving figure'
fig.savefig('/u/haskeysr/josh_real_space_{}_abs_Bn.pdf'.format(title))



fig,axes = pt.subplots(ncols = 2, sharex = True, sharey = True)
cplots = []
clim = [0,4]
plot_kwargs = {'rasterized':True}
#plot_kwargs = {}
if plot_type == 'plasma':
    data_to_use = data_pest_p
    title = 'plasma-only'
elif plot_type == 'total':
    data_to_use = data_pest_t
    title = 'plasma-plus-vacuum'
elif plot_type == 'vacuum':
    data_to_use = data_pest_v
    title = 'vacuum-only'
else:
    raise ValueError()

for dat,dat_obj_tmp, ax in zip(data_to_use, data_obj, axes):
    dat_obj_tmp2 = copy.deepcopy(dat_obj_tmp)
    dat_obj_tmp2.BnPEST = +dat
    a = dat_obj_tmp2.plot_BnPEST(ax, inc_contours = False, n=1)
    #grid_R = dat_obj.R*dat_tot.R0EXP
    #grid_Z = dat_obj.Z*dat_tot.R0EXP
    #print 'sh_mod', np.max(grid_R), np.min(grid_R), np.max(grid_Z), np.min(grid_Z)
    #start_surface = 0; end_surface = dat_obj.Ns1; skip = 1; plot_quantity = np.real(dat)
    #color_plot = ax.pcolormesh(grid_R[start_surface:end_surface:skip,:], grid_Z[start_surface:end_surface:skip,:], plot_quantity[start_surface:end_surface:skip,:],cmap = 'RdBu', **plot_kwargs)
    #color_plot.set_clim(clim)
    cplots.append(a)
    cplots[-1].set_clim(clim)
cbar = pt.colorbar(cplots[-1], ax = axes.tolist())
cbar.set_label('abs(Bn) (G/kA) : {}'.format(title))
#axes[0].set_xlim([1,2.4])
#axes[0].set_ylim([-1.25,1.25])
#pt.colorbar(cplots[-1], ax = axes.tolist())
print 'saving figure'
for i in axes:i.set_xlabel('m')
axes[0].set_ylabel('$\sqrt{\psi_N}$')
fig.savefig('/u/haskeysr/josh_harmonics_{}_Bn.pdf'.format(title))


# dat_tot = res.data('/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp0.700/marsrun/RUNrfa.p',getpest=True)
# dat_vac = res.data('/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp0.700/marsrun/RUNrfa.vac',getpest=True)
# '/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp1.200/marsrun/RUNrfa.p'
# High Pressure Vacuum: /u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp1.200/marsrun/RUNrfa.vac
