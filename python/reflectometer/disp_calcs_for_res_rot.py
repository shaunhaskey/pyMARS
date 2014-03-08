import matplotlib.pyplot as pt
import numpy as np
from  results_class import *
from RZfuncs import I0EXP_calc
import matplotlib.pyplot as pt
import time
import PythonMARS_funcs as pyMARS
import generic_funcs as gen_func

def extract_data(base_dir, I0EXP, ul=0, Nchi=513, get_VPLASMA=0, plas_vac = True):
    if ul==0:
        c = data(base_dir + 'RUNrfa.p', I0EXP = I0EXP, Nchi=Nchi)
        if plas_vac:
            d = data(base_dir + 'RUNrfa.vac', I0EXP = I0EXP, Nchi=Nchi)
        else:
            d = None
        if get_VPLASMA:
            c.get_VPLASMA()
            if plas_vac:
                d.get_VPLASMA()
        return (c,d)
    else:
        a = data(base_dir + 'RUN_rfa_lower.p', I0EXP = I0EXP, Nchi=Nchi)
        c = data(base_dir + 'RUN_rfa_upper.p', I0EXP = I0EXP, Nchi=Nchi)
        if plas_vac:
            d = data(base_dir + 'RUN_rfa_upper.vac', I0EXP = I0EXP, Nchi=Nchi)
            b = data(base_dir + 'RUN_rfa_lower.vac', I0EXP = I0EXP, Nchi=Nchi)
        else:
            d = None
            b = None
        if get_VPLASMA:
            a.get_VPLASMA()
            c.get_VPLASMA()
            if plas_vac:
                b.get_VPLASMA()
                d.get_VPLASMA()
        return (a,b,c,d)

N = 6; n = 2; I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)
print I0EXP, 1.0e+3 * 3./np.pi

ul = 1;
plot_field = 'Vn'; field_type = 'plas'
Nchi=513
base_dir = '/home/srh112/NAMP_datafiles/mars/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/'
base_dir = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/times/1935/RES2.0000_ROTE-100.0000/'
run_data = extract_data(base_dir, I0EXP, ul=ul, Nchi=Nchi,get_VPLASMA=1)
out = disp_calcs(run_data, n_zones = 100, phasing_vals = [0,45], ul= ul)




theta_deg = 0
print '===== %d ====='%(theta_deg)
theta = float(theta_deg)/180*np.pi;

plot_quantity = combine_fields_displacement(run_data, plot_field, theta=theta, field_type=field_type)

grid_r = run_data[0].R*run_data[0].R0EXP
grid_z = run_data[0].Z*run_data[0].R0EXP

plas_r = grid_r[0:plot_quantity.shape[0],:]
plas_z = grid_z[0:plot_quantity.shape[0],:]

fig,ax = pt.subplots(ncols = 2, sharex = True, sharey = True)
# cm_to_inch=0.393701
# import matplotlib as mpl
# old_rc_Params = mpl.rcParams
# mpl.rcParams['font.size']=8.0
# mpl.rcParams['axes.titlesize']=8.0#'medium'
# mpl.rcParams['xtick.labelsize']=8.0
# mpl.rcParams['ytick.labelsize']=8.0
# mpl.rcParams['lines.markersize']=5.0
# mpl.rcParams['savefig.dpi']=300
# fig.set_figwidth(8.48*cm_to_inch)
# fig.set_figheight(8.48*0.7*cm_to_inch)
gen_func.setup_publication_image(fig,)
mesh = ax[0].pcolormesh(plas_r, plas_z, np.real(plot_quantity), rasterized = True)
mesh.set_clim([-0.001,0.001])

fig2, ax2 = pt.subplots()
surfaces = range(0,plas_r.shape[0],30)
surfaces = [-1]
for i in surfaces: 
    ax[1].plot(plas_r[i,:],plas_z[i,:],'-')
    norm_z = np.diff(plas_r[i,:])
    norm_r = -np.diff(plas_z[i,:])
    dl = np.sqrt(norm_r**2+norm_z**2)
    norm_r/=dl
    norm_z/=dl
    disp_quant = np.real(plot_quantity[i,:])
    ax[1].plot(plas_r[i,:-1]+10*norm_r*disp_quant[:-1],plas_z[i,:-1]+10*norm_z*disp_quant[:-1])
    angle = np.arctan2(plas_z[i,:], plas_r[i,:]-run_data[0].R0EXP)
    angle = np.rad2deg(angle)
    ax2.plot(angle, disp_quant)
for i in ax: i.grid()
for i in ax: i.set_xlabel('R (m)')
ax[0].set_ylabel('Z (m)')
ax[1].axhline(0.6*np.min(plas_z[-1,:]))
ax[0].set_ylim([-1.25,1.25])
for i in ax: i.set_xlim([1.0,2.5])
ax[0].set_xticks(ax[0].get_xticks()[::2])
fig.tight_layout(pad = 0.3)
fig.savefig('/home/srh112/code/NAMP_analysis/python/reflectometer/edge_displacement.pdf')
fig.canvas.draw();fig.show()
fig2.canvas.draw();fig2.show()

1/0
fig, ax = pt.subplots()
max_min_z = [np.max(plas_z[-1,:]), np.min(plas_z[-1,:])]
upper_values = np.linspace(0,np.max(plas_z[-1,:]),50,endpoint = True)
lower_values = np.linspace(0,np.min(plas_z[-1,:]),50,endpoint = True)
r_vals = plas_r[-1,:]
z_vals = plas_z[-1,:]
dz = np.diff(z_vals)
dl = np.sqrt(np.diff(z_vals)**2 + np.diff(r_vals)**2)
angle = np.arctan2(plas_z[-1,:], plas_r[-1,:]-run_data[0].R0EXP)
z_vals_red = z_vals[:-1]
r_vals_red = r_vals[:-1]
plot_quantity_red = plot_quantity[-1,:-1]
disp_below_LFS = []
disp_below_HFS = []
disp_above_LFS = []
disp_above_HFS = []

ang_below_LFS = []
ang_below_HFS = []
ang_above_LFS = []
ang_above_HFS = []

for i in range(1,len(upper_values)):
    truth = (z_vals_red>=upper_values[i-1])*(z_vals_red<upper_values[i])
    #print upper_values[i-1], upper_values[i], np.sum(truth)
    truth1 = truth*dz<0
    disp_above_HFS.append(np.sum(np.abs(plot_quantity_red[truth1]))/np.sum(truth1))
    ang_above_HFS.append(np.mean(angle[truth1]))
    ax.plot(r_vals_red[truth1], z_vals_red[truth1],'--')
    truth2 = truth*dz>0
    disp_above_LFS.append(np.sum(np.abs(plot_quantity_red[truth2]))/np.sum(truth2))
    ang_above_LFS.append(np.mean(angle[truth2]))
    ax.plot(r_vals_red[truth2], z_vals_red[truth2],'-')
for i in range(1,len(lower_values)):
    truth = (z_vals_red<lower_values[i-1])*(z_vals_red>=lower_values[i])
    #print upper_values[i-1], upper_values[i], np.sum(truth)
    truth1 = truth*dz<0
    disp_below_HFS.append(np.sum(np.abs(plot_quantity_red[truth1]))/np.sum(truth1))
    ang_below_HFS.append(np.mean(angle[truth1]))
    ax.plot(r_vals_red[truth1], z_vals_red[truth1],'--')
    truth2 = truth*dz>0
    disp_below_LFS.append(np.sum(np.abs(plot_quantity_red[truth2]))/np.sum(truth2))
    ang_below_LFS.append(np.mean(angle[truth2]))
    ax.plot(r_vals_red[truth2], z_vals_red[truth2],'-')
fig.canvas.draw(); fig.show()

fig, ax = pt.subplots(ncols = 4, sharex =True, sharey = True)
fig, ax = pt.subplots()
tmp = [disp_below_LFS, disp_below_HFS, disp_above_HFS, disp_above_LFS]
tmp2 = [ang_below_LFS, ang_below_HFS, ang_above_HFS, ang_above_LFS]

tmp3 = ['disp_below_LFS', 'disp_below_HFS', 'disp_above_HFS', 'disp_above_LFS']
tmp4 = ['ang_below_LFS', 'ang_below_HFS', 'ang_above_HFS', 'ang_above_LFS']
for i in range(4):
    #ax[i].plot(tmp2[i],tmp[i])
    ax.plot(tmp2[i],tmp[i])
    ax.plot(out[theta_deg][tmp4[i]],out[theta_deg][tmp3[i]], '--')
fig.canvas.draw(); fig.show()
