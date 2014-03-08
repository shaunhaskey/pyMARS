import numpy as np
import matplotlib.pyplot as pt
base_dir = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/efit/'
base_dir2 = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/times/'
fig, ax_orig = pt.subplots(ncols = 2, nrows = 2); ax2 = ax_orig.flatten()
fig2, ax_orig2 = pt.subplots(ncols = 2, nrows = 2); ax3 = ax_orig2.flatten()
times = np.arange(1495,2015+40,40)
times = [1615,1815,2015]

profeq_cols = ['s','q','jeq','peq','rho','rot','resist','gamarr', 'gmunu', 'tempi', 'tempe','dpsids', 't', 'omegasi','omegase']
for ax, ax3, profeq_name, profile in zip(ax2,ax3,['rot','rho','resist','tempe'],['PROFROT','PROFDEN','PROFTI','PROFTE']):
    for i in times:
        a = np.loadtxt('{}{}/{}'.format(base_dir,str(i),profile), skiprows = 1)
        b = np.loadtxt('{}{}//RES2.0000_ROTE-100.0000/RUN_rfa_upper.p/{}'.format(base_dir2,str(i),'PROFEQ.OUT'), skiprows = 1)
        ax.plot((a[:,0]), a[:,1])
        ind = profeq_cols.index(profeq_name)
        ax3.plot((b[:,0]), b[:,ind])
        tmp = np.argmax(a[:,1])
        ax.text((a[tmp, 0]), a[tmp,1], str(i)+'ms')
    ax.set_title(profile)
    ax3.set_title(profeq_name)
for i in ax_orig[-1,:]: i.set_xlabel('s')
fig.tight_layout(pad= 0.01)
fig.canvas.draw(); fig.show()
fig2.canvas.draw(); fig2.show()
