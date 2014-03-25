import numpy as np
import matplotlib.pyplot as pt
base_dir = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/efit/'
base_dir2 = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/times/'
fig, ax_orig = pt.subplots(ncols = 2, nrows = 2); ax2 = ax_orig.flatten()
fig2, ax_orig2 = pt.subplots(ncols = 2, nrows = 2); ax3 = ax_orig2.flatten()
times = np.arange(1495,2015+40,40)
times = [1615,1815,1935,2015]

profeq_cols = ['s','q','jeq','peq','rho','rot','resist','gamarr', 'gmunu', 'tempi', 'tempe','dpsids', 't', 'omegasi','omegase']
#for ax, ax3, profeq_name, profile in zip(ax2,ax3,['rot','rho','resist','tempe'],['PROFROT','PROFDEN','PROFTI','PROFTE']):
for ax, ax3, profeq_name, profile in zip(ax2,ax3,['rot','rho','resist','q'],['PROFROT','PROFDEN','PROFTI','PROFTE']):
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


base_dir = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/efit/'
base_dir2 = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/times/'
fig2, ax_orig2 = pt.subplots(ncols = 2, nrows = 2, sharex = True); ax3 = ax_orig2.flatten()
import pyMARS.generic_funcs as gen_funcs
gen_funcs.setup_publication_image(fig2, height_prop = 1./1.618*1.2, single_col = True)
times = [1615]
profeq_cols = ['s','q','jeq','peq','rho','rot','resist','gamarr', 'gmunu', 'tempi', 'tempe','dpsids', 't', 'omegasi','omegase']
plot_profs = ['rot','rho','resist','q']
plot_ylabels = ['$\omega/\omega_A$','Normalised density', 'Resistivity 1/S', 'q']
log_y_list = [False, False, True, False]

for ax3_tmp, profeq_name, ylabel, log_y in zip(ax3,plot_profs,plot_ylabels, log_y_list):
    for i in times:
        b = np.loadtxt('{}{}//RES2.0000_ROTE-100.0000/RUN_rfa_upper.p/{}'.format(base_dir2,str(i),'PROFEQ.OUT'), skiprows = 1)
        ind = profeq_cols.index(profeq_name)
        ax3_tmp.plot((b[:,0]), b[:,ind])
        ax3_tmp.set_ylabel(ylabel)
        if log_y: ax3_tmp.set_yscale('log')
ax3[2].set_ylim([1.e-9, 1.e-6])
for i in ax_orig2[-1,:]: i.set_xlabel('$\sqrt{\psi_N}$')
for i in [0,1,3]: gen_funcs.setup_axis_publication(ax3[i], n_xticks = 5, n_yticks = 5); 
for i in ax3: i.grid()

fig2.tight_layout(pad = 0.1)
fig2.savefig('profiles.pdf')
fig2.savefig('profiles.eps')
fig2.canvas.draw(); fig2.show()
