'''
Generates plots of 'kink amplification' as a function of phasing Will
also create the files for an animation of plasma, vac, and total
components in PEST co-ordinates


This generates the plot that is going into Matt's paper Shaun Haskey
12/12/2012

Essentially this works by choosing a specific vacuum
harmonic. The amplitude of this harmonic will be plotted as a function
of phasing. You also need to select a s_surface. This surface should
be slightly displaced from the specific vacuum harmonic surface (to
avoid problems resonance on that surface. The maximum amplitude
harmonic (based on all phasings) on this surface is chosen, and the
plas, tot and vac amplitudes for this harmonic are plotted as a
function of phasing.

SH : 22May2013 : modified script so that you can look at several
s_surfaces and vacuum harmonics on the same plot - as this was
requested by Matt/paper referee

'''

import results_class
from RZfuncs import I0EXP_calc
from RZfuncs import I0EXP_calc_real
import numpy as np
import matplotlib.pyplot as pt
import PythonMARS_funcs as pyMARS


def plot_data_Matt_paper(s_surface,specific_vac_mode, upper_data_tot, lower_data_tot, upper_data_vac, lower_data_vac, SURFMN_coords, ax, phasing_range, n, linestyle, phase_machine_ntor, force_harmonic =None):
    #Get the kink amplitude data
    mk_upper, ss_upper, relevant_values_upper_tot,tmp_useless = upper_data_tot.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
    mk_lower, ss_lower, relevant_values_lower_tot,tmp_useless = lower_data_tot.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
    mk_upper, ss_upper, relevant_values_upper_vac,tmp_useless = upper_data_vac.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)
    mk_lower, ss_lower, relevant_values_lower_vac,tmp_useless = lower_data_vac.kink_amp(s_surface, q_range, n = n, SURFMN_coords=SURFMN_coords)

    #Get the resonant strenght data
    a, upper_vac_res = upper_data_vac.resonant_strength(SURFMN_coords=SURFMN_coords)
    a, lower_vac_res = lower_data_vac.resonant_strength(SURFMN_coords=SURFMN_coords)
    a, upper_tot_res = upper_data_tot.resonant_strength(SURFMN_coords=SURFMN_coords)
    a, lower_tot_res = lower_data_tot.resonant_strength(SURFMN_coords=SURFMN_coords)

    #number_points = len(relevant_values_lower_vac)
    phasings = np.arange(phasing_range[0], phasing_range[1]+1,0.1)
    amps_vac_comp = [];amps_tot_comp = [];amps_plasma_comp = []
    vac_qn = []

    #Apply the different phasings
    for phasing in phasings:
        phasing = phasing/180.*np.pi
        if phase_machine_ntor:
            phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
        else:
            phasor = (np.cos(phasing)+1j*np.sin(phasing))
        amps_vac_comp.append(relevant_values_upper_vac + relevant_values_lower_vac*phasor)
        amps_tot_comp.append(relevant_values_upper_tot + relevant_values_lower_tot*phasor)
        amps_plasma_comp.append(relevant_values_upper_tot-relevant_values_upper_vac + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)
        vac_qn.append(np.abs(upper_vac_res + lower_vac_res*phasor))
    vac_qn = np.array(vac_qn)
    if force_harmonic==None:
        print 'using an unforced harmonic'
        tmp_loc = np.argmax(np.max(np.abs(amps_tot_comp),axis=1))#/number_points)
        tmp_max_phasing = phasings[tmp_loc]
        best_harmonic = np.argmax(np.abs(np.array(amps_tot_comp)[tmp_loc,:]))
        print 'best_harmonic_loc: %d, m:%d, phasing machine:%.2f, phasing MARS:%.2f, s: %.4f'%(best_harmonic, mk_upper[best_harmonic], tmp_max_phasing,-tmp_max_phasing*n, upper_data_tot.q_profile_s[np.argmin(np.abs(upper_data_tot.q_profile-4.5))])
    else:
        print 'using a forced harmonic'
        best_harmonic = np.argmin(np.abs(mk_upper - force_harmonic))
        print 'best_harmonic_loc: %d, m:%d, s: %.4f'%(best_harmonic, mk_upper[best_harmonic], upper_data_tot.q_profile_s[np.argmin(np.abs(upper_data_tot.q_profile-4.5))])

    # for phasing in phasings:
    #     phasing = phasing/180.*np.pi
    #     if phase_machine_ntor:
    #         phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    #     else:
    #         phasor = (np.cos(phasing)+1j*np.sin(phasing))
    #     #amps_tot.append(np.sum(np.abs(relevant_values_upper_tot + relevant_values_lower_tot*phasor)))
    #     #amps_plasma.append(np.sum(np.abs((relevant_values_upper_tot-relevant_values_upper_vac) + (relevant_values_lower_tot-relevant_values_lower_vac)*phasor)))
    # vac_qn = np.array(vac_qn)
    for i, j in enumerate(upper_data_tot.qn):
        if upper_data_tot.mq[i]==specific_vac_mode:
            if seperate_res_plot:
                #ax[1].plot(phasings,vac_qn[:,i], color = 'gray', linestyle = '-', label = 'q=%.2f,m=%d'%(j, upper_data_tot.mq[i]))
                ax[1].plot(phasings,vac_qn[:,i], color = 'green', linestyle = linestyle, label = r'Vac q=%.2f,m=-%d,$\Psi_N=%.3f$'%(j, upper_data_tot.mq[i], upper_data_tot.sq[i]**2))
            else:
                #ax[0].plot(phasings,vac_qn[:,i], color = 'gray', linestyle = '-',  label = 'q=%.2f,m=%d'%(j, upper_data_tot.mq[i]))
                ax[0].plot(phasings,vac_qn[:,i], color = 'green', linestyle = linestyle,  label = r'Vac q=%.2f,m=-%d,$\Psi_N=%.3f$'%(j, upper_data_tot.mq[i], upper_data_tot.sq[i]**2))
    ax[0].plot(phasings, np.abs(amps_vac_comp)[:,best_harmonic], 'b'+linestyle, label = r'Vac m=-%d $\Psi_N=%.3f$'%(mk_upper[best_harmonic], s_surface**2))
    ax[0].plot(phasings, np.abs(amps_plasma_comp)[:,best_harmonic], 'r'+linestyle, label = r'Plas m=-%d, $\Psi_N=%.3f$'%(mk_upper[best_harmonic], s_surface**2))
    ax[0].plot(phasings, np.abs(amps_tot_comp)[:,best_harmonic], 'k'+linestyle, label= r'Tot m=-%d, $\Psi_N=%.3f$'%(mk_upper[best_harmonic], s_surface**2))


SURFMN_coords = 1
N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])
#I = np.array([1.,-0.5,-0.5,1,-0.5,-0.5])
#I0EXP = I0EXP_calc(N,n,I)
#print I0EXP
I0EXP = I0EXP_calc_real(n,I,discrete_pts=1000, produce_plot=0)
print I0EXP

facn = 1.0
q_range = [2,6]; ylim = [0,3.7]
#phasing_range = [-180.,180.]
#phasing_range = [0.,360.]
phasing_range = [-90.,90.]
phase_machine_ntor = 1
make_animations = 0
include_discrete_comparison = 0
seperate_res_plot = 0
include_vert_lines = 0

plot_type = 'best_harmonic'
#plot_type = 'normalised'
#plot_type = 'normalised_average'
#plot_type = 'standard_average'

#using a few different ones
#-sim
#base_dir = '/home/srh112/NAMP_datafiles/mars/shot146398_upper_lower/qmult1.000/exp1.000/marsrun/'
base_dir = '/home/srh112/NAMP_datafiles/mars/shot146394_upper_lower/qmult1.000/exp1.000/marsrun/'
upper_data_tot = results_class.data(base_dir + 'RUN_rfa_upper.p',I0EXP=I0EXP)
lower_data_tot = results_class.data(base_dir + 'RUN_rfa_lower.p', I0EXP=I0EXP)
upper_data_vac = results_class.data(base_dir + 'RUN_rfa_upper.vac',I0EXP=I0EXP)
lower_data_vac = results_class.data(base_dir + 'RUN_rfa_lower.vac', I0EXP=I0EXP)

upper_data_tot.get_PEST(facn = facn)
lower_data_tot.get_PEST(facn = facn)
upper_data_vac.get_PEST(facn = facn)
lower_data_vac.get_PEST(facn = facn)

#Generate plots for Matt's paper
fig,ax = pt.subplots(); ax = [ax]
s_surface =0.933;specific_vac_mode = 6;linestyle='-.' #This is in Matts paper for a s that is slightly above q=4,m=8 resonance
plot_data_Matt_paper(s_surface,specific_vac_mode, upper_data_tot, lower_data_tot, upper_data_vac, lower_data_vac, SURFMN_coords, ax, phasing_range, n, linestyle, phase_machine_ntor)
s_surface =0.965;specific_vac_mode = 7;linestyle='-' #This is in Matts paper for a s that is slightly above q=3.5,m=7 resonance
plot_data_Matt_paper(s_surface,specific_vac_mode, upper_data_tot, lower_data_tot, upper_data_vac, lower_data_vac, SURFMN_coords, ax, phasing_range, n, linestyle, phase_machine_ntor)
s_surface =0.987;specific_vac_mode = 8;linestyle='--' #This is in Matts paper for a s that is slightly above q=4,m=8 resonance
plot_data_Matt_paper(s_surface,specific_vac_mode, upper_data_tot, lower_data_tot, upper_data_vac, lower_data_vac, SURFMN_coords, ax, phasing_range, n, linestyle, phase_machine_ntor)
ylim = [0,4.3]

ax[0].set_xlabel(r'$\Delta \phi_{ul}$ (n=2) (deg)',fontsize = 14)
ax[0].set_ylabel('Amplitude (G/kA)', fontsize = 14)
ax[0].set_xlim(phasing_range)
ax[0].set_ylim(ylim)
leg = ax[0].legend(loc='best',prop={'size':10})
leg.get_frame().set_alpha(0.5)
minor_ticks = range(-90,91,15)
major_ticks = range(-90,91,45)
for i in major_ticks: minor_ticks.remove(i)
ax[0].xaxis.set_ticks(major_ticks,minor=False)
ax[0].xaxis.set_ticks(minor_ticks, minor=True)
ax[0].grid(b=True, which='major', linestyle='-',axis='x')
ax[0].grid(b=True, which='major', linestyle=':',axis='y')
ax[0].grid(b=True, which='minor', linestyle=':')
fig.canvas.draw(); fig.show()



if include_discrete_comparison:
    #single_answers
    single_phasings = range(0,360,60)
    #single_phasings = [0,120]
    single_data_vac_dict = {}
    single_data_tot_dict = {}

    for i in single_phasings:
        single_data_vac_dict[str(i)] = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_%ddeg/qmult1.000/exp1.000/marsrun/RUNrfa.vac'%(i),I0EXP=I0EXP)
        single_data_tot_dict[str(i)] = results_class.data('/home/srh112/NAMP_datafiles/mars/shot146398_%ddeg/qmult1.000/exp1.000/marsrun/RUNrfa.p'%(i),I0EXP=I0EXP)

    kink_values_vac = []; kink_values_tot = [];kink_values_plas = []
    resonant_values_vac = []

    for i in single_phasings:
        single_data_vac_dict[str(i)].get_PEST(facn = facn)
        single_data_tot_dict[str(i)].get_PEST(facn = facn)
        mk_upper, ss_upper, relevant_tmp_vac = single_data_vac_dict[str(i)].kink_amp(s_surface, q_range, n = n, SURFMN_coords = SURFMN_coords)
        kink_values_vac.append(np.sum(np.abs(relevant_tmp_vac)))
        mk_upper, ss_upper, relevant_tmp_tot = single_data_tot_dict[str(i)].kink_amp(s_surface, q_range, n = n, SURFMN_coords = SURFMN_coords)
        kink_values_tot.append(np.sum(np.abs(relevant_tmp_tot)))
        kink_values_plas.append(np.sum(np.abs(relevant_tmp_tot-relevant_tmp_vac)))

        a, res_tmp_vac = single_data_vac_dict[str(i)].resonant_strength(SURFMN_coords = SURFMN_coords)
        #vac_qn.append(np.abs(upper_vac_res + lower_vac_res*phasor))
        resonant_values_vac.append(np.abs(res_tmp_vac))

    #for i in range(0, len(single_phasings)):
    #    if single_phasings[i]>phasing_range[1]:
    #        single_phasings[i] = single_phasings[i] - 360

    #plot_angles = np.array(single_angles)*n*-1
    if phase_machine_ntor:
        plot_angles = (np.array(single_phasings)*-1)/2.
    else:
        plot_angles = single_phasings

    for i in range(0, len(plot_angles)):
        while (plot_angles[i]>phasing_range[1]) or (plot_angles[i]<=phasing_range[0]):
            if plot_angles[i]>phasing_range[1]:
                plot_angles[i] = plot_angles[i] - 360
            elif plot_angles[i]<=phasing_range[0]:
                plot_angles[i] = plot_angles[i] + 360

    ax[0].plot(plot_angles,kink_values_tot, 'ks-')
    ax[0].plot(plot_angles,kink_values_vac, 'bs-')
    ax[0].plot(plot_angles,kink_values_plas, 'rs-')

    resonant_values_vac = np.array(resonant_values_vac)
    for i,j in enumerate(single_phasings):
        ax[1].plot(plot_angles[i]*np.ones(len(resonant_values_vac[i,:])), resonant_values_vac[i,:], 'ys')

ax[0].set_xlabel(r'$\Delta \phi_{ul}$ (n=2) (deg)',fontsize = 14)
ax[0].set_ylabel('Amplitude (G/kA)', fontsize = 14)

ax[0].set_xlim(phasing_range)
ax[0].set_ylim(ylim)
leg = ax[0].legend(loc='best',prop={'size':10})
leg.get_frame().set_alpha(0.5)
minor_ticks = range(-90,91,15)
major_ticks = range(-90,91,45)
for i in major_ticks: minor_ticks.remove(i)

ax[0].xaxis.set_ticks(major_ticks,minor=False)
ax[0].xaxis.set_ticks(minor_ticks, minor=True)
ax[0].grid(b=True, which='major', linestyle='-',axis='x')
ax[0].grid(b=True, which='major', linestyle=':',axis='y')
ax[0].grid(b=True, which='minor', linestyle=':')
fig.canvas.draw(); fig.show()




if make_animations:
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

