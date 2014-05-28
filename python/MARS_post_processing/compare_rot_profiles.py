import numpy as np
import pyMARS.PythonMARS_funcs as pyf
import matplotlib.pyplot as pt

various_times = False
if various_times:
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

single_based_on_time = True

if single_based_on_time:
    base_dir = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/efit/'
    base_dir2 = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/times/'
    #NEED TO MODIFY THIS BASE DIR HERE.... to the spitzer ones

    fig2, ax_orig2 = pt.subplots(ncols = 2, nrows = 2, sharex = True); ax3 = ax_orig2.flatten()
    import pyMARS.generic_funcs as gen_funcs
    gen_funcs.setup_publication_image(fig2, height_prop = 1./1.618*1.2, single_col = True)
    times = [1615]
    profeq_cols = ['s','q','jeq','peq','rho','rot','resist','gamarr', 'gmunu', 'tempi', 'tempe','dpsids', 't', 'omegasi','omegase']
    plot_profs = ['rot','rho','resist','q']
    plot_ylabels = ['$\omega_0$','Normalised density', '$\eta_0$', 'q']
    log_y_list = [False, False, True, False]

    for ax3_tmp, profeq_name, ylabel, log_y in zip(ax3,plot_profs,plot_ylabels, log_y_list):
        for i in times:
            b = np.loadtxt('{}{}//RES2.0000_ROTE-100.0000/RUN_rfa_upper.p/{}'.format(base_dir2,str(i),'PROFEQ.OUT'), skiprows = 1)
            if profeq_name == 'resist':
                c = np.loadtxt('{}{}//RES2.0000_ROTE-100.0000/RUN_rfa_upper.p/PROFTE'.format(base_dir2,str(i)), skiprows = 1)
                with file('{}{}//RES2.0000_ROTE-100.0000/RUN_rfa_upper.p/RUN'.format(base_dir2,str(i)),'r') as filehandle:
                    tmp_RUN_lines = filehandle.readlines()
                for i in tmp_RUN_lines:
                    if i.find('ETA')>=0: 
                        ETA = float(i.split('=')[1].rstrip('\n').rstrip(','))
                        ETA = 7.e-8
                        print ETA
                d = c[:,1]**(-3./2)
                d = d/d[0]*ETA
                print d
                ax3_tmp.plot((c[:,0]), d)
            ind = profeq_cols.index(profeq_name)
            if profeq_name!= 'resist': ax3_tmp.plot((b[:,0]), b[:,ind])
            ax3_tmp.set_ylabel(ylabel)
            if log_y: ax3_tmp.set_yscale('log')
    ax3[2].set_ylim([1.e-8, 5.e-5])
    for i in ax_orig2[-1,:]: i.set_xlabel('$\sqrt{\psi_N}$')
    for i in [0,1,3]: gen_funcs.setup_axis_publication(ax3[i], n_xticks = 5, n_yticks = 5); 
    for i in ax3: i.grid()

    fig2.tight_layout(pad = 0.1)
    fig2.savefig('profiles.pdf')
    fig2.savefig('profiles.eps')
    fig2.savefig('profiles.svg')
    fig2.canvas.draw(); fig2.show()

single_based_on_dir = True
if single_based_on_dir:
    base_dirs = ['/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_med_rote/qmult1.000/exp1.000/RES48.3293_ROTE0.0113/RUN_rfa_upper.p/',
                 '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_med_rote/qmult1.000/exp1.000/RES100.0000_ROTE0.0113/RUN_rfa_upper.p/']
    base_dirs = ['/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/qmult1.000/exp1.000/RES112.8838_ROTE0.1438/RUN_rfa_upper.p/',
                 '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/qmult1.000/exp1.000/RES54.5559_ROTE0.1438/RUN_rfa_upper.p/']
    fig2, ax_orig2 = pt.subplots(ncols = 2, nrows = 2, sharex = True); ax3 = ax_orig2.flatten()
    import pyMARS.generic_funcs as gen_funcs
    gen_funcs.setup_publication_image(fig2, height_prop = 1./1.618*1.2, single_col = True)
    times = [1615]
    profeq_cols = ['s','q','jeq','peq','rho','rot','resist','gamarr', 'gmunu', 'tempi', 'tempe','dpsids', 't', 'omegasi','omegase']
    plot_profs = ['rot','rho','resist','q']
    plot_ylabels = ['$\omega/\omega_A$','Normalised density', 'Resistivity 1/S', 'q']
    log_y_list = [False, False, True, False]
    for tmp_base_dir in base_dirs:
        for ax3_tmp, profeq_name, ylabel, log_y in zip(ax3,plot_profs,plot_ylabels, log_y_list):
            for i in times:
                b = np.loadtxt('{}/{}'.format(tmp_base_dir,'PROFEQ.OUT'), skiprows = 1)
                ind = profeq_cols.index(profeq_name)
                ax3_tmp.plot((b[:,0]), b[:,ind])
                ax3_tmp.set_ylabel(ylabel)
                if log_y: ax3_tmp.set_yscale('log')
    #ax3[2].set_ylim([1.e-9, 1.e-6])
    for i in ax_orig2[-1,:]: i.set_xlabel('$\sqrt{\psi_N}$')
    for i in [0,1,3]: gen_funcs.setup_axis_publication(ax3[i], n_xticks = 5, n_yticks = 5); 
    for i in ax3: i.grid()

    fig2.tight_layout(pad = 0.1)
    fig2.savefig('profiles.pdf')
    fig2.savefig('profiles.eps')
    fig2.canvas.draw(); fig2.show()
