'''
Generates plots of 'kink amplification' as a function of phasing
Will also create the files for an animation of plasma, vac, and total 
components in PEST co-ordinates

'''

import results_class
from RZfuncs import I0EXP_calc
import numpy as np
import matplotlib.pyplot as pt
import PythonMARS_funcs as pyMARS

N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)
facn = 1.0; psi = 0.92
q_range = [2,4]; ylim = [0,10]

upper_data_tot = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.p',I0EXP=I0EXP)
lower_data_tot = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.p', I0EXP=I0EXP)
upper_data_vac = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/RUN_rfa_upper.vac',I0EXP=I0EXP)
lower_data_vac = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/RUN_rfa_lower.vac', I0EXP=I0EXP)

upper_data_tot.get_PEST(facn = facn)
lower_data_tot.get_PEST(facn = facn)
upper_data_vac.get_PEST(facn = facn)
lower_data_vac.get_PEST(facn = facn)

mk_upper, ss_upper, relevant_values_upper_tot = upper_data_tot.kink_amp(psi, q_range, n = n)
mk_lower, ss_lower, relevant_values_lower_tot = lower_data_tot.kink_amp(psi, q_range, n = n)
mk_upper, ss_upper, relevant_values_upper_vac = upper_data_vac.kink_amp(psi, q_range, n = n)
mk_lower, ss_lower, relevant_values_lower_vac = lower_data_vac.kink_amp(psi, q_range, n = n)

phasings = np.arange(0,360,1)
amps_tot = []; amps_vac = []; amps_plasma = []
fig, ax = pt.subplots()

for phasing in phasings:
    phasing = phasing/180.*np.pi
    phasor = (np.cos(phasing)+1j*np.sin(phasing))
    amps_vac.append(np.sum(np.abs(relevant_values_upper_vac + relevant_values_lower_vac*phasor)))
    amps_tot.append(np.sum(np.abs(relevant_values_upper_tot + relevant_values_lower_tot*phasor)))
    amps_plasma.append(np.sum(np.abs((relevant_values_upper_tot-relevant_values_upper_vac) + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)))

max_loc = np.argmax(amps_tot); min_loc = np.argmin(amps_tot)
ax.vlines([phasings[max_loc],phasings[min_loc]],ylim[0],ylim[1])
max_loc = np.argmax(amps_vac); min_loc = np.argmin(amps_vac)
ax.vlines([phasings[max_loc],phasings[min_loc]],ylim[0],ylim[1])
max_loc = np.argmax(amps_plasma); min_loc = np.argmin(amps_plasma)
ax.vlines([phasings[max_loc], phasings[min_loc]],ylim[0],ylim[1])

ax.plot(phasings, amps_vac, 'bx-', label = 'Vac')
ax.plot(phasings, amps_plasma, 'rx-', label = 'Plasma')
ax.plot(phasings, amps_tot, 'kx-', label='Total')

ax.set_xlabel('Phasing (deg)')
ax.set_ylabel('Kink amplitude')

ax.set_xlim([0,360])
ax.set_ylim([0,10])
ax.legend(loc='best')
ax.grid(); fig.canvas.draw(); fig.show()

#Total phasing animation
phasings = np.linspace(0,360,15)
for phasing in phasings:
    fig_anim, ax_anim = pt.subplots()
    phasing = phasing/180.*np.pi
    phasor = (np.cos(phasing)+1j*np.sin(phasing))
    BnPEST_new = upper_data_tot.BnPEST  + lower_data_tot.BnPEST*phasor
    color_ax = ax_anim.pcolor(upper_data_tot.mk.flatten(),upper_data_tot.ss.flatten(),np.abs(BnPEST_new),cmap='hot')
    pt.colorbar(color_ax,ax=ax_anim)
    ax_anim.plot(mk_upper, mk_upper * 0 + ss_upper,'bo')
    color_ax.set_clim([0,2])
    ax_anim.plot(upper_data_tot.mq,upper_data_tot.sq,'bo')
    ax_anim.plot(upper_data_tot.q_profile*n,upper_data_tot.q_profile_s,'b--') 
    ax_anim.set_xlim([-29,29])
    ax_anim.set_ylim([0,1])
    ax_anim.set_title('phasing : %d deg'%(phasing/np.pi*180.))
    fig_anim.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/tot_tmp_%d.png'%(phasing/np.pi*180.))
    pt.close()

#Vacuum phasing animation
phasings = np.linspace(0,360,15)
for phasing in phasings:
    fig_anim, ax_anim = pt.subplots()
    phasing = phasing/180.*np.pi
    phasor = (np.cos(phasing)+1j*np.sin(phasing))
    BnPEST_new = upper_data_vac.BnPEST  + lower_data_vac.BnPEST*phasor
    color_ax = ax_anim.pcolor(upper_data_vac.mk.flatten(),upper_data_vac.ss.flatten(),np.abs(BnPEST_new),cmap='hot')
    pt.colorbar(color_ax,ax=ax_anim)
    ax_anim.plot(mk_upper, mk_upper * 0 + ss_upper,'bo')
    color_ax.set_clim([0,2])
    ax_anim.plot(upper_data_vac.mq,upper_data_vac.sq,'bo')
    ax_anim.plot(upper_data_vac.q_profile*n,upper_data_vac.q_profile_s,'b--') 
    ax_anim.set_xlim([-29,29])
    ax_anim.set_ylim([0,1])
    ax_anim.set_title('phasing : %d deg'%(phasing/np.pi*180.))
    fig_anim.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/vac_tmp_%d.png'%(phasing/np.pi*180.))
    pt.close()

#Plasma phasing animation
for phasing in phasings:
    fig_anim, ax_anim = pt.subplots()
    phasing = phasing/180.*np.pi
    phasor = (np.cos(phasing)+1j*np.sin(phasing))
    BnPEST_new = (upper_data_tot.BnPEST - upper_data_vac.BnPEST)  + (lower_data_tot.BnPEST-lower_data_vac.BnPEST)*phasor
    color_ax = ax_anim.pcolor(upper_data_vac.mk.flatten(),upper_data_vac.ss.flatten(),np.abs(BnPEST_new),cmap='hot')
    pt.colorbar(color_ax,ax=ax_anim)
    ax_anim.plot(mk_upper, mk_upper * 0 + ss_upper,'bo')
    color_ax.set_clim([0,2])
    ax_anim.plot(upper_data_vac.mq,upper_data_vac.sq,'bo')
    ax_anim.plot(upper_data_vac.q_profile*n,upper_data_vac.q_profile_s,'b--') 
    ax_anim.set_xlim([-29,29])
    ax_anim.set_ylim([0,1])
    ax_anim.set_title('phasing : %d deg'%(phasing/np.pi*180.))
    fig_anim.savefig('/home/srh112/code/NAMP_analysis/python/MARS_post_processing/plas_tmp_%d.png'%(phasing/np.pi*180.))
    pt.close()

