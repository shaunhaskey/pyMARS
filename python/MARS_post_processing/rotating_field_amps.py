import pyMARS.results_class as results_class
import numpy as np
import matplotlib.pyplot as pt
import os

n = 2
filename_list = []
for i,phase in enumerate(np.linspace(0,2.*np.pi,24)):
    I = np.cos(2*np.deg2rad(np.arange(0,360,60))+phase)
    #I = [1,-0.5,-0.5,1,-0.5,-0.5]
    I0EXP = results_class.I0EXP_calc_real(n,I, produce_plot = True, inc_phase = True)
    fig = pt.gcf()
    filename_list.append('{:03d}.png'.format(i))
    fig.savefig(filename_list[-1])
os.system('convert -delay {} -loop 0 {} {}'.format(20, ' '.join(filename_list), 'harm_phasings.gif'))
