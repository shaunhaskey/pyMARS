import numpy as np
import matplotlib.pyplot as pt
import pyMARS.generic_funcs as gen_funcs
fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True)
gen_funcs.setup_publication_image(fig, height_prop = 1./1.618*2.0, single_col = True)
all_times = range(1655,2140,40)
select_times = [1615,1815,2015, 2135]
styles = [{'color':'k'},{'color':'b'},{'color':'r'},{'color':'g'}]

#profeq = np.loadtxt('/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/qmult1.000/exp1.000/RES112.8838_ROTE1.0000/RUN_rfa_lower.p/PROFEQ.OUT',skiprows = 1)
base_directory = '/u/haskeysr/efit/142614/1795/orig_matt/profiles129-omegaNC/'
profeq = np.loadtxt(file('/u/haskeysr/mars/shot_142614_rote_res_scan_20x20_kpar1_low_rote/qmult1.000/exp1.000/RES112.8838_ROTE1.0000/RUN_rfa_lower.p/PROFEQ.OUT','r'))
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
    for i in ax: i.axvline(vert_lines[-1], linestyle = '-')
print vert_lines
for count,(i, style) in enumerate(zip(select_times, styles)):
    a = np.loadtxt(base_directory + 'dtrot142614.0{}.dat'.format(i), skiprows = 1)
    ax[0].plot(a[:,0], a[:,1]/1000, label = str(i), **style)
    ax[0].text(a[0,0],a[0,1]/1000, str(i)+'ms')
    if count==0: base_profile = a[:,1]/1000
    scaled_prof = base_profile / base_profile[0] * a[0,1]/1000
    ax[1].plot(a[:,0], scaled_prof, label = str(i), **style)
    ax[1].text(a[0,0], scaled_prof[0], str(i)+'ms')

#ax.legend(loc='best')
for i in ax:i.grid()
ax[-1].set_xlabel(r'$\sqrt{\Psi_N}$')
for i in ax:i.set_ylabel('Toroidal Rotation (krad/s)')
fig.tight_layout(pad = 0.1)
fig.savefig('toroidal_rotation_profiles_NC.pdf')
fig.canvas.draw(); fig.show()
