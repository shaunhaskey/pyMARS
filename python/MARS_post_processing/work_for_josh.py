import pyMARS.results_class as res
import matplotlib.pyplot as pt
import numpy as np

fig,axes = pt.subplots(ncos = 2, sharex = True, sharey = True)

base_dirs = ['/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp1.200/marsrun/',
             '/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp0.700/marsrun/']
for base_dir, ax in zip(base_dirs, axes):
    dat_tot = res.data(base_dir + '/RUNrfa.p',getpest=True)
    dat_vac = res.data(base_dir + '/RUNrfa.vac',getpest=True)
    dat_tot.plot_Bn(np.real(dat_tot - dat_vac), axis = ax, end_surface = dat_tot.Ns1, cmap = "RdBu")
fig.canvas.draw(); fig.show()


# dat_tot = res.data('/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp0.700/marsrun/RUNrfa.p',getpest=True)
# dat_vac = res.data('/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp0.700/marsrun/RUNrfa.vac',getpest=True)
# '/u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp1.200/marsrun/RUNrfa.p'
# High Pressure Vacuum: /u/kingjd/mars/marsk/sh153480_q43_gt_nw/king/096/sh153480_4marsk/qmult0.900/exp1.200/marsrun/RUNrfa.vac
