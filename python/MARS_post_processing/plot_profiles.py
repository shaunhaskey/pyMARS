import numpy as np
import matplotlib.pyplot as pt
import pyMARS.generic_funcs as gen_funcs
import pyMARS.PythonMARS_funcs as pyM
all_times = range(1655,2140,40)
select_times = [1615,1815,2015, 2135]
styles = [{'color':'k'},{'color':'b'},{'color':'r'},{'color':'g'}]

#profeq = np.loadtxt('/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/qmult1.000/exp1.000/RES112.8838_ROTE1.0000/RUN_rfa_lower.p/PROFEQ.OUT',skiprows = 1)

def plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = '', figname = None, modifier = None, scale = 1., scaled = False, ax = None, inc_text = True, log_y = False, apply_func = None, apply_func_kwargs = None,legend = False):
    if apply_func_kwargs == None: apply_func_kwargs = {}
    profeq = np.loadtxt(profeq_loc)
    s = profeq[:,0]
    q = profeq[:,1]
    tmp_max = int(np.round(np.max(q) * 3))
    np.min(q)*3
    tmp_min = int(np.round(np.min(q) * 3))
    if tmp_min<(np.min(q) * 3): tmp_min += 1
    if tmp_max>(np.max(q) * 3): tmp_max -= 1
    print tmp_min, tmp_max
    vert_lines = []
    for i in range(tmp_min,tmp_max):
        vert_lines.append(s[np.argmin(np.abs(3*q - i))])
        ax.axvline(vert_lines[-1], linestyle = '-')
    print vert_lines
    for count,(i, style) in enumerate(zip(select_times, styles)):
        a = np.loadtxt(base_directory + '{}{}.dat'.format(prof_prefix, i), skiprows = 1)
        vals = a[:,1]
        if apply_func!=None: vals = apply_func(vals, **apply_func_kwargs)
        if count==0: base_profile = +vals
        scaled_prof = base_profile / base_profile[0] * vals[0]
        label_string = str(i)+'ms'
        if scaled:
            ax.plot(a[:,0], scaled_prof*scale, label = label_string, **style)
            if inc_text:ax.text(a[0,0], scaled_prof[0]*scale, label_string)
        else:
            ax.plot(a[:,0], vals*scale, label = label_string, **style)
            if inc_text: ax.text(a[0,0],vals[1]*scale, label_string)
    if log_y: ax.set_yscale('log')
    if legend:ax.legend(loc = 'best')
#ax.legend(loc='best')
prof_prefix = 'dtrot142614.0'
figname = 'toroidal_rotation_profiles_NC.pdf'
ylabel = 'Toroidal Rotation (krad/s)'

#base_directory = '/u/haskeysr/efit/142614/1795/profiles129-omegaNC/'
base_directory = '/u/haskeysr/efit/142614/1795/orig_matt/profiles129-omegaNC/'
profeq_loc = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/qmult1.000/exp1.000/RES112.8838_ROTE1.0000/RUN_rfa_lower.p/PROFEQ.OUT'
fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True)
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*2.0, single_col = True)
plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = ylabel, figname= figname, scaled = False, scale = 1./1000, ax = ax[0])
plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = ylabel, figname= figname, scaled = True, scale = 1./1000, ax = ax[1])
for i in ax:i.grid()
ax[-1].set_xlabel(r'$\sqrt{\Psi_N}$')
for i in ax:i.set_ylabel(ylabel)
fig.tight_layout(pad = 0.1)
if figname!=None: fig.savefig(figname)
fig.canvas.draw(); fig.show()

#pyM.spitz_eta_func(Te, Z = 1, e = -1.602176565*10**(-19), m = 9.10938291*10**(-31), coul_log = 15, e_0 = 8.85418782*10**(-12), K = 1.38*10**(-23), chen_H_approx = False):

fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True)
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*2.0, single_col = True)
prof_prefix = 'dte142614.0'
figname = 'te_profiles_NC.pdf'
ylabel =  '$\eta$ (Ohm-m)'
plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = ylabel, figname= figname, scaled = False, scale = 1., ax = ax[0], apply_func = pyM.spitz_eta_func, log_y = True, legend = True, inc_text = False)
plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = ylabel, figname= figname, scaled = True, scale = 1., ax = ax[1], apply_func = pyM.spitz_eta_func, log_y = True, inc_text = False)
for i in ax:i.grid()
ax[-1].set_xlabel(r'$\sqrt{\Psi_N}$')
for i in ax:i.set_ylabel(ylabel)
fig.tight_layout(pad = 0.1)
if figname!=None: fig.savefig(figname)
fig.canvas.draw(); fig.show()


prof_prefix = 'dtrot142614.0'
figname = 'combined_profiles_NC.pdf'
#base_directory = '/u/haskeysr/efit/142614/1795/profiles129-omegaNC/'
base_directory = '/u/haskeysr/efit/142614/1795/orig_matt/profiles129-omegaNC/'
profeq_loc = '/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/qmult1.000/exp1.000/RES112.8838_ROTE1.0000/RUN_rfa_lower.p/PROFEQ.OUT'
fig, ax = pt.subplots(nrows = 3, sharex = True, sharey = False)
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*2.5, single_col = True)
plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = ylabel, figname= figname, scaled = False, scale = 1./1000, ax = ax[0])
plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = ylabel, figname= figname, scaled = True, scale = 1./1000, ax = ax[1])
prof_prefix = 'dte142614.0'
plot_profile(base_directory, profeq_loc, prof_prefix, ylabel = ylabel, figname= figname, scaled = False, scale = 1., ax = ax[2], apply_func = pyM.spitz_eta_func, log_y = True, legend = True, inc_text = False)
for i in ax:i.grid()
ax[-1].set_xlabel(r'$\sqrt{\Psi_N}$')
for i in range(2): ax[i].set_ylabel('Toroidal Rotation (krad/s)')
ax[2].set_ylabel('$\eta$ (Ohm-m)')
fig.tight_layout(pad = 0.1)
if figname!=None: fig.savefig(figname)
fig.canvas.draw(); fig.show()
