'''
Generate animations of the magnetic field with vac, plasma and combination
Good for putting into talks
SH: 11Sept2012
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
base_dir = '/home/srh112/code/pyMARS/shot146382_single_ul/qmult1.000/exp1.000/marsrun/'
#base_dir = '/home/srh112/code/pyMARS/shot146382_single_tog/qmult1.000/exp1.000/marsrun/'
ul = 1; theta = 300./180*num.pi; plot_field = 'Bn'
run_data = extract_data(base_dir, I0EXP, ul=ul)
plot_quantity = combine_fields(run_data, plot_field, theta=theta)



#c = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.p', I0EXP = I0EXP)
#c = data('/home/srh112/code/pyMARS/test_shot/marsrun/RUNrfa_COILlower.p', I0EXP = I0EXP)

#d = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.vac', I0EXP = I0EXP)

#c.get_VPLASMA()
title = plot_field

n=1
for t in num.linspace(0,0.1, num=15):
    start_time = time.time()
    fig = pt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, sharex=ax1, sharey=ax1)
    phasor = num.exp(-1j*10.*num.pi*2.*t)
    print t,phasor
    plas_field = num.real(combine_fields(run_data, plot_field, theta = theta, field_type='plas')*phasor)
    total_field = num.real(combine_fields(run_data, plot_field, theta = theta, field_type='total')*phasor)
    vac_field = num.real(combine_fields(run_data, plot_field, theta = theta, field_type='vac')*phasor)

    amp_plot = run_data[0].plot_Bn(total_field, ax1,start_surface = 0,end_surface = run_data[0].Ns1+run_data[0].NW+4, skip=1, cmap='hot',plot_coils_switch=1, plot_boundaries=1, wall_grid = run_data[0].NW)
    amp_plot2 = run_data[0].plot_Bn(vac_field, ax2,start_surface = 0,end_surface = run_data[0].Ns1+run_data[0].NW+4, skip=1, cmap='hot',plot_coils_switch=1, plot_boundaries=1, wall_grid = run_data[0].NW)
    amp_plot3 = run_data[0].plot_Bn(plas_field, ax3,start_surface = 0,end_surface = run_data[0].Ns1+run_data[0].NW+4, skip=1, cmap='hot',plot_coils_switch=1, plot_boundaries=1, wall_grid = run_data[0].NW)

    #amp_plot = d.plot_Bn(total_field, ax1,start_surface = 0,end_surface = d.Ns1+30, skip=1, cmap='hot',plot_boundaries=1)
    #amp_plot2 = d.plot_Bn(vac_field, ax2,start_surface = 0,end_surface = d.Ns1+30, skip=1, cmap='hot',plot_boundaries=1)
    #amp_plot3 = d.plot_Bn(plas_field, ax3,start_surface = 0,end_surface = d.Ns1+30, skip=1, cmap='hot',plot_boundaries=1)

    cbar1 = pt.colorbar(amp_plot,ax=ax1)
    cbar1 = pt.colorbar(amp_plot2,ax=ax2)
    cbar1 = pt.colorbar(amp_plot3,ax=ax3)

    #cbar2 = pt.colorbar(phase_plot,ax=ax2)
    amp_plot.set_clim([-10,10])
    amp_plot2.set_clim([-10,10])
    amp_plot3.set_clim([-4,4])

#c.plot_Bn(start_surface = 0,end_surface = c.Ns1+24+2,skip=1, modification = d.Bn, field = 'Bn')

    ax1.set_title(plot_field + ' total t=%.3fs'%(t))
    ax2.set_title(plot_field + ' vac t=%.3fs'%(t))
    ax3.set_title(plot_field + ' plas t=%.3fs'%(t))

    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    fig.canvas.draw()
    fig.show()
    fig.savefig(fig_save_dir + 'testing'+str(n)+'.png')
    pt.close('all')

    n+=1
    print 'finished in %.3fs'%(time.time()-start_time)

