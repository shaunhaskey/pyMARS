import matplotlib.pyplot as pt
import numpy as np
from  results_class import *
from RZfuncs import I0EXP_calc
import matplotlib.pyplot as pt
import time
import PythonMARS_funcs as pyMARS

def extract_data(base_dir, I0EXP, ul=0, Nchi=513, get_VPLASMA=0):
    if ul==0:
        c = data(base_dir + 'RUNrfa.p', I0EXP = I0EXP, Nchi=Nchi)
        d = data(base_dir + 'RUNrfa.vac', I0EXP = I0EXP, Nchi=Nchi)
        if get_VPLASMA:
            c.get_VPLASMA()
            d.get_VPLASMA()
        return (c,d)
    else:
        a = data(base_dir + 'RUN_rfa_lower.p', I0EXP = I0EXP, Nchi=Nchi)
        b = data(base_dir + 'RUN_rfa_lower.vac', I0EXP = I0EXP, Nchi=Nchi)
        c = data(base_dir + 'RUN_rfa_upper.p', I0EXP = I0EXP, Nchi=Nchi)
        d = data(base_dir + 'RUN_rfa_upper.vac', I0EXP = I0EXP, Nchi=Nchi)
        if get_VPLASMA:
            a.get_VPLASMA()
            b.get_VPLASMA()
            c.get_VPLASMA()
            d.get_VPLASMA()
        return (a,b,c,d)

def combine_fields(input_data, attr_name, theta = 0, field_type='plas'):
    print 'combining property : ', attr_name
    if len(input_data)==2:
        if field_type=='plas':
            answer = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
        elif field_type == 'total':
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
        answer = upper_data + lower_data*(np.cos(theta)+1j*np.sin(theta))
        print 'combine %s, theta : %.2f'%(field_type, theta)
    return answer

base_dir_list = ['/home/srh112/mars/shot146397_3305/qmult1.000/exp1.000/marsrun/', '/home/srh112/mars/shot146397_3815/qmult1.000/exp1.000/marsrun/', '/home/srh112/mars/shot146397_3515/qmult1.000/exp1.000/marsrun/']

base_dir_list = ['/home/srh112/mars/shot146398_3305/qmult1.000/exp1.000/marsrun/', '/home/srh112/mars/shot146398_3815/qmult1.000/exp1.000/marsrun/', '/home/srh112/mars/shot146398_3515/qmult1.000/exp1.000/marsrun/', '/home/srh112/code/pyMARS/other_scripts/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/']


label_list = ['3305', '3815', '3515', 'kinetic']
include_MARS = 1

fig_tmp, ax_tmp = pt.subplots(nrows = 2, sharex=1)
for tmp_loc, base_dir in enumerate(base_dir_list):
    try:
        N = 6; n = 2; I = np.array([1.,-1.,0.,1,-1.,0.])
        I0EXP = I0EXP_calc(N,n,I)
        print I0EXP, 1.0e+3 * 3./np.pi

        ul = 1;
        plot_field = 'Vn'; field_type = 'plas'
        Nchi=513
        run_data = extract_data(base_dir, I0EXP, ul=ul, Nchi=Nchi,get_VPLASMA=1)


        for theta_deg in [0]:
            print '===== %d ====='%(theta_deg)
            theta = float(theta_deg)/180*np.pi;

            plot_quantity = combine_fields(run_data, plot_field, theta=theta, field_type=field_type)

            grid_r = run_data[0].R*run_data[0].R0EXP
            grid_z = run_data[0].Z*run_data[0].R0EXP

            plas_r = grid_r[0:plot_quantity.shape[0],:]
            plas_z = grid_z[0:plot_quantity.shape[0],:]

            R_values=np.linspace(run_data[0].R0EXP, np.max(plas_r),10000)
            Z_values=R_values * 0

            Vn_values = scipy_griddata((plas_r.flatten(),plas_z.flatten()), plot_quantity.flatten(), (R_values.flatten(),Z_values.flatten()))
            #ax_freq[0].plot(R_values*100, np.abs(Vn_values)*100,'k-', label='MARS-Vn')

        plot_field = 'Vr'; field_type = 'plas'
        for theta_deg in [0]:
            print '===== %d ====='%(theta_deg)
            theta = float(theta_deg)/180*np.pi;

            plot_quantity = combine_fields(run_data, plot_field, theta=theta, field_type=field_type)

            grid_r = run_data[0].R*run_data[0].R0EXP
            grid_z = run_data[0].Z*run_data[0].R0EXP

            plas_r = grid_r[0:plot_quantity.shape[0],:]
            plas_z = grid_z[0:plot_quantity.shape[0],:]
            R_values=np.linspace(run_data[0].R0EXP, np.max(plas_r),10000)
            Z_values=R_values * 0

            Vr_values = scipy_griddata((plas_r.flatten(),plas_z.flatten()), plot_quantity.flatten(), (R_values.flatten(),Z_values.flatten()))
            #ax_freq[0].plot(R_values*100, np.abs(Vr_values)*100,'r.', label='MARS-Vr')

        ax_tmp[0].plot(R_values*100, np.abs(Vn_values)*100,'.', label=label_list[tmp_loc])
        ax_tmp[1].plot(R_values*100, np.angle(Vn_values,deg=True),'.', label=label_list[tmp_loc])
    except:
        print 'failed on ', base_dir

ax_tmp[0].legend(loc='best')
ax_tmp[1].legend(loc='best')
fig_tmp.canvas.draw(); fig_tmp.show()
