from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt

import PythonMARS_funcs as pyMARS

N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I); facn = 1.0

#I0EXP = 1.0e+3*3.**1.5/(2.*np.pi)
#I0EXP = 1.0e+3*0.954 #PMZ ideal
I0EXP = 1.0e+3*0.863 #PMZ real
#I0EXP = 1.0e+3*0.827 #MPM ideal
#I0EXP = 1.0e+3*0.748 #MPM real
#I0EXP = 1.0e+3*0.412 #MPM n4 real
#I0EXP = 1.0e+3*0.528 #PMZ n4 real

#print I0EXP, 1.0e+3 * 3./np.pi
dir_loc ='/home/srh112/NAMP_datafiles/mars/plotk_rzplot/146382/qmult1.000/exp1.000/marsrun/RUNrfa.vac' 
d = data(dir_loc, I0EXP=I0EXP)
d.get_PEST(facn = facn)
fig,ax = pt.subplots()
color_plot = d.plot_BnPEST(ax, n=n, inc_contours = 1)
color_plot.set_clim([0,1.5])
cbar = pt.colorbar(color_plot, ax = ax)
ax.set_xlabel('m')
ax.set_ylabel(r'$\sqrt{\psi_N}$', fontsize = 14)
cbar.ax.set_ylabel('G/kA')

fig.canvas.draw(); fig.show()
