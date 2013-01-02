'''
Generate animations of the magnetic field with vac, plasma and combination
for comparing standing wave and single wave options...
Good for putting into talks
SH: 11/12/2012
'''

from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as num
import matplotlib.pyplot as pt
import time
import PythonMARS_funcs as pyMARS

def extract_data(base_dir, I0EXP, ul=0):
    if ul==0:
        c = data(base_dir + 'RUNrfa.p', I0EXP = I0EXP)
        d = data(base_dir + 'RUNrfa.vac', I0EXP = I0EXP)
        return (c,d)
    else:
        a = data(base_dir + 'RUN_rfa_lower.p', I0EXP = I0EXP)
        b = data(base_dir + 'RUN_rfa_lower.vac', I0EXP = I0EXP)
        c = data(base_dir + 'RUN_rfa_upper.p', I0EXP = I0EXP)
        d = data(base_dir + 'RUN_rfa_upper.vac', I0EXP = I0EXP)
        return (a,b,c,d)


def combine_fields(input_data, attr_name, theta = 0, field_type='plas'):
    print 'combining property : ', attr_name
    if len(input_data)==2:
        if field_type=='plas':
            answer = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
        elif field_type== 'total':
            answer = getattr(input_data[0], attr_name)
        elif field_type == 'vac':
            answer = getattr(input_data[1], attr_name)
        print 'already combined, ', field_type
    elif len(input_data) == 4:
        if field_type=='plas':
            lower_data = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
            upper_data = getattr(input_data[2], attr_name) - getattr(input_data[3], attr_name)
        elif field_type=='total':
            lower_data = getattr(input_data[0], attr_name)
            upper_data = getattr(input_data[2], attr_name)
        elif field_type=='vac':
            lower_data = getattr(input_data[1], attr_name)
            upper_data = getattr(input_data[3], attr_name)
        answer = upper_data + lower_data*(num.cos(theta)+1j*num.sin(theta))
        print 'combine %s, theta : %.2f'%(field_type, theta)
    return answer


N = 6; n = 2; I = num.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)

print I0EXP, 1.0e+3 * 3./num.pi

fig_save_dir = '/home/srh112/Desktop/animations/'
base_dir = '/home/srh112/NAMP_datafiles/mars/shot146382_single_ul/qmult1.000/exp1.000/marsrun/'
#base_dir = '/home/srh112/code/pyMARS/shot146382_single_tog/qmult1.000/exp1.000/marsrun/'
ul = 0; plot_field = 'Bn'; theta = 0
base_dir_pp = '/home/srh112/NAMP_datafiles/mars/yueqiang_standing_wave_pp/qmult1.000/exp1.000/marsrun/'
base_dir_mp = '/home/srh112/NAMP_datafiles/mars/yueqiang_standing_wave_mp/qmult1.000/exp1.000/marsrun/'
base_dir_pm = '/home/srh112/NAMP_datafiles/mars/yueqiang_standing_wave_pm/qmult1.000/exp1.000/marsrun/'
base_dir_mm = '/home/srh112/NAMP_datafiles/mars/yueqiang_standing_wave_mm/qmult1.000/exp1.000/marsrun/'


data_pp = extract_data(base_dir_pp, I0EXP, ul=ul)
data_mp = extract_data(base_dir_mp, I0EXP, ul=ul)
data_pm = extract_data(base_dir_pm, I0EXP, ul=ul)
data_mm = extract_data(base_dir_mm, I0EXP, ul=ul)


#run_data = extract_data(base_dir, I0EXP, ul=ul)


#plot_quantity = combine_fields(run_data, plot_field, theta=theta)



#c = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.p', I0EXP = I0EXP)
#c = data('/home/srh112/code/pyMARS/test_shot/marsrun/RUNrfa_COILlower.p', I0EXP = I0EXP)

#d = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.vac', I0EXP = I0EXP)

#c.get_VPLASMA()
title = plot_field
phi_locations = np.arange(0,2.*np.pi,np.pi/15)
counter=1
#for t in num.linspace(0,0.1, num=30):
for t in [0.2]:
    fig, ax = pt.subplots(nrows=2)
    non_standing = np.zeros((len(phi_locations),data_pp[0].Bn.shape[1]),dtype=complex)
    standing = non_standing *0
    standing2 = non_standing *0
    theta_values = np.real(non_standing) * 0
    phi_values = np.real(non_standing) * 0
    for i, phi_location in enumerate(phi_locations):
        start_time = time.time()
        #fig = pt.figure(figsize=(20,10))
        #ax1 = fig.add_subplot(131)
        #ax2 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
        #ax3 = fig.add_subplot(133, sharex=ax1, sharey=ax1)
        phasor_m = num.exp(1j*10.*num.pi*2.*t)
        phasor_p = num.exp(-1j*10.*num.pi*2.*t)

        phasor_phi_p = num.exp(1j*n*phi_location)
        phasor_phi_m = num.exp(-1j*n*phi_location)

        non_standing_wave = 1

        print 'combining for non standing wave'
        plas_field_non = 0.5*(combine_fields(data_mp, plot_field, theta = theta, field_type='plas')*phasor_p*phasor_phi_m + \
                                  combine_fields(data_pm, plot_field, theta = theta, field_type='plas')*phasor_m*phasor_phi_p)
        total_field_non = 0.5*(combine_fields(data_mp, plot_field, theta = theta, field_type='total')*phasor_p*phasor_phi_m + \
                                   combine_fields(data_pm, plot_field, theta = theta, field_type='total')*phasor_m*phasor_phi_p)
        vac_field_non = 0.5*(combine_fields(data_mp, plot_field, theta = theta, field_type='vac')*phasor_p*phasor_phi_m + \
                                 combine_fields(data_pm, plot_field, theta = theta, field_type='vac')*phasor_m*phasor_phi_p)

        vac_field_stand2 = 0.5*(combine_fields(data_pp, plot_field, theta = theta, field_type='vac')*phasor_p*phasor_phi_p +\
                                    combine_fields(data_pm, plot_field, theta = theta, field_type='vac')*phasor_m*phasor_phi_p)

        Ns = 170
        #non_standing[i,:] = vac_field_non[Ns,:]
        non_standing[i,:] = np.real(vac_field_stand2[Ns,:])

        phi_values[i,:] += phi_location
        theta_values[i,:] = np.arctan2(data_pp[0].Z[Ns,:], data_pp[0].R[Ns,:]-1)
        theta_values[i,theta_values[i,:]<theta_values[i,0]]+=2.*np.pi
        print 'non-standing imag: %5.3f, %5.3f, %5.3f'%(np.max(np.abs(np.imag(plas_field_non))), np.max(np.abs(np.imag(total_field_non))), np.max(np.abs(np.imag(vac_field_non))))
        print 'non-standing real: %5.3f, %5.3f, %5.3f'%(np.max(np.abs(np.real(plas_field_non))), np.max(np.abs(np.real(total_field_non))), np.max(np.abs(np.real(vac_field_non))))

        # fig_tmp, ax_tmp = pt.subplots(nrows = 2)
        # real_plot = data_pp[0].plot_Bn(np.real(vac_field_non), ax_tmp[0],start_surface = 0,end_surface = data_pp[0].Ns1+data_pp[0].NW+4, skip=1, cmap='hot',plot_coils_switch=0, plot_boundaries=1, wall_grid = data_pp[0].NW)
        # imag_plot = data_pp[0].plot_Bn(np.imag(vac_field_non), ax_tmp[1],start_surface = 0,end_surface = data_pp[0].Ns1+data_pp[0].NW+4, skip=1, cmap='hot',plot_coils_switch=0, plot_boundaries=1, wall_grid = data_pp[0].NW)
        # cbar1 = pt.colorbar(real_plot,ax=ax_tmp[0])
        # cbar1 = pt.colorbar(imag_plot,ax=ax_tmp[1])
        # real_plot.set_clim([-10,10])
        # imag_plot.set_clim([-10,10])
        # fig_tmp.canvas.draw(); fig_tmp.show()


        plas_field_stand = 0.25*(combine_fields(data_mp, plot_field, theta = theta, field_type='plas')*phasor_p*phasor_phi_m +\
                                     combine_fields(data_pm, plot_field, theta = theta, field_type='plas')*phasor_m*phasor_phi_p + \
                                     combine_fields(data_pp, plot_field, theta = theta, field_type='plas')*phasor_p*phasor_phi_p +\
                                     combine_fields(data_mm, plot_field, theta = theta, field_type='plas')*phasor_m*phasor_phi_m)
        total_field_stand = 0.25*(combine_fields(data_mp, plot_field, theta = theta, field_type='total')*phasor_p*phasor_phi_m + \
                                      combine_fields(data_pm, plot_field, theta = theta, field_type='total')*phasor_m*phasor_phi_p + \
                                      combine_fields(data_pp, plot_field, theta = theta, field_type='total')*phasor_p*phasor_phi_p + \
                                      combine_fields(data_mm, plot_field, theta = theta, field_type='total')*phasor_m*phasor_phi_m)
        vac_field_stand = 0.25*(combine_fields(data_mp, plot_field, theta = theta, field_type='vac')*phasor_p*phasor_phi_m + \
                                    combine_fields(data_pm, plot_field, theta = theta, field_type='vac')*phasor_m*phasor_phi_p +\
                                    combine_fields(data_pp, plot_field, theta = theta, field_type='vac')*phasor_p*phasor_phi_p +\
                                    combine_fields(data_mm, plot_field, theta = theta, field_type='vac')*phasor_m*phasor_phi_m)

        plas_field_stand2 = 0.5*(combine_fields(data_pp, plot_field, theta = theta, field_type='plas')*phasor_p*phasor_phi_p +\
                                    combine_fields(data_pm, plot_field, theta = theta, field_type='plas')*phasor_m*phasor_phi_p)

        #standing[i,:] = np.real(plas_field_stand2[Ns,:])
        standing[i,:] = plas_field_stand[Ns,:]



        print 'standing imag : %5.3f, %5.3f, %5.3f'%(np.max(np.abs(np.imag(plas_field_stand))), np.max(np.abs(np.imag(total_field_stand))), np.max(np.abs(np.imag(vac_field_stand))))
        print 'standing real : %5.3f, %5.3f, %5.3f'%(np.max(np.abs(np.real(plas_field_stand))), np.max(np.abs(np.real(total_field_stand))), np.max(np.abs(np.real(vac_field_stand))))


    tmp = (standing / np.sqrt(np.real(standing)**2 + np.imag(standing)**2)).flatten()
    fig_tmp, ax_tmp = pt.subplots()
    ax_tmp.plot(np.real(tmp), np.imag(tmp),'.')
    ax_tmp.set_xlim((-1,1))
    ax_tmp.set_ylim((-1,1))
    fig_tmp.canvas.draw(); fig_tmp.show()
    if np.min(np.abs(np.imag(non_standing.flatten())))<1.e-8:
        non_standing = np.real(non_standing)
    if np.min(np.abs(np.imag(standing.flatten())))<1.e-8:
        standing = np.real(standing)


    ax[0].imshow(non_standing.transpose(),aspect='auto', cmap='hot', extent=[np.min(phi_values[0,:]),np.max(phi_values[-1,:]), theta_values[0,0], theta_values[0,-1]], origin='upper')
    ax[1].imshow(standing.transpose(),aspect='auto', cmap='hot', extent=[np.min(phi_values[0,:]),np.max(phi_values[-1,:]), theta_values[0,0], theta_values[0,-1]],origin='lower')
    ax[1].set_xlabel('phi')
    ax[1].set_ylabel('theta')
    ax[0].set_ylabel('theta')
    #ax[1].set_title('Standing : Xpp+Xmp+Xpm+Xpp, t=%.2f'%(t,))
    ax[1].set_title('Re{Xpp,p+Xpm,p}, t=%.2fs, Ns = %d'%(t,Ns))

    #ax[0].set_title('Travelling : Xmp+Xpm, t=%.2f'%(t,))
    ax[0].set_title('Re{Xpp,v+Xpm,v}, t=%.2fs, Ns = %d'%(t,Ns))

    #fig.canvas.draw();fig.show()
    fig.savefig('/%s_testing_standing2_%02d.png'%(fig_save_dir,counter))
    counter+=1
        # fig_tmp, ax_tmp = pt.subplots(nrows = 2)
        # real_plot = data_pp[0].plot_Bn(np.real(vac_field_stand), ax_tmp[0],start_surface = 0,end_surface = data_pp[0].Ns1+data_pp[0].NW+4, skip=1, cmap='hot',plot_coils_switch=0, plot_boundaries=1, wall_grid = data_pp[0].NW)
        # imag_plot = data_pp[0].plot_Bn(np.imag(vac_field_stand), ax_tmp[1],start_surface = 0,end_surface = data_pp[0].Ns1+data_pp[0].NW+4, skip=1, cmap='hot',plot_coils_switch=0, plot_boundaries=1, wall_grid = data_pp[0].NW)
        # cbar1 = pt.colorbar(real_plot,ax=ax_tmp[0])
        # cbar1 = pt.colorbar(imag_plot,ax=ax_tmp[1])
        # real_plot.set_clim([-10,10])
        # imag_plot.set_clim([-10,10])

        # fig_tmp.canvas.draw(); fig_tmp.show()


        #plas_field = np.real(plas_field)
        #total_field = np.real(total_field)
        #vac_field = np.real(vac_field)

        ##amp_plot = run_data_pp[0].plot_Bn(total_field, ax1,start_surface = 0,end_surface = run_data[0].Ns1+run_data[0].NW+4, skip=1, cmap='hot',plot_coils_switch=1, plot_boundaries=1, wall_grid = run_data[0].NW)
        ##amp_plot2 = run_data_pp[0].plot_Bn(vac_field, ax2,start_surface = 0,end_surface = run_data[0].Ns1+run_data[0].NW+4, skip=1, cmap='hot',plot_coils_switch=1, plot_boundaries=1, wall_grid = run_data[0].NW)
        ##amp_plot3 = run_data_pp[0].plot_Bn(plas_field, ax3,start_surface = 0,end_surface = run_data[0].Ns1+run_data[0].NW+4, skip=1, cmap='hot',plot_coils_switch=1, plot_boundaries=1, wall_grid = run_data[0].NW)

        #amp_plot = d.plot_Bn(total_field, ax1,start_surface = 0,end_surface = d.Ns1+30, skip=1, cmap='hot',plot_boundaries=1)
        #amp_plot2 = d.plot_Bn(vac_field, ax2,start_surface = 0,end_surface = d.Ns1+30, skip=1, cmap='hot',plot_boundaries=1)
        #amp_plot3 = d.plot_Bn(plas_field, ax3,start_surface = 0,end_surface = d.Ns1+30, skip=1, cmap='hot',plot_boundaries=1)



        # cbar1 = pt.colorbar(amp_plot,ax=ax1)
        # cbar1 = pt.colorbar(amp_plot2,ax=ax2)
        # cbar1 = pt.colorbar(amp_plot3,ax=ax3)

        # amp_plot.set_clim([-10,10])
        # amp_plot2.set_clim([-10,10])
        # amp_plot3.set_clim([-4,4])



        # ax1.set_title(plot_field + ' total t=%.3fs'%(t))
        # ax2.set_title(plot_field + ' vac t=%.3fs'%(t))
        # ax3.set_title(plot_field + ' plas t=%.3fs'%(t))

        # ax1.set_xlabel('R (m)')
        # ax1.set_ylabel('Z (m)')
        # fig.canvas.draw()
        # fig.show()
        # fig.savefig(fig_save_dir + 'testing_standing'+str(n)+'.png')
        # pt.close('all')

        #counter+=1
        #print 'finished in %.3fs'%(time.time()-start_time)

