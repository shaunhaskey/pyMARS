from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as num
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
        answer = upper_data + lower_data*(num.cos(theta)+1j*num.sin(theta))
        print 'combine %s, theta : %.2f'%(field_type, theta)
    return answer

N = 6; n = 2; I = num.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)
print I0EXP, 1.0e+3 * 3./num.pi

base_dir = '/u/haskeysr/mars/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/'
ul = 1;
plot_field = 'Vn'; field_type = 'plas'
Nchi=513
run_data = extract_data(base_dir, I0EXP, ul=ul, Nchi=Nchi,get_VPLASMA=1)
fig, ax = pt.subplots()

for theta_deg in range(0,360,60):
    print '===== %d ====='%(theta_deg)
    theta = float(theta_deg)/180*num.pi;

    plot_quantity = combine_fields(run_data, plot_field, theta=theta, field_type=field_type)

    grid_r = run_data[0].R*run_data[0].R0EXP
    grid_z = run_data[0].Z*run_data[0].R0EXP

    plas_r = grid_r[0:plot_quantity.shape[0],:]
    plas_z = grid_z[0:plot_quantity.shape[0],:]
    R_values=num.linspace(run_data[0].R0EXP, num.max(plas_r),10000)
    Z_values=R_values * 0

    tmp_values = scipy_griddata((plas_r.flatten(),plas_z.flatten()), plot_quantity.flatten(), (R_values.flatten(),Z_values.flatten()))
    ax.plot(R_values, num.abs(tmp_values)*100,'-', label=str(theta_deg))

ax.legend()
ax.set_xlabel('R (m)')
ax.set_ylabel('norm disp cm/kA')
ax.set_title('146398 MARS midplane normal displacement - different phasings')
ax.set_xlim([1.8,2.3])
ax.set_ylim([0,0.6])

ax.grid()
fig.canvas.draw()
fig.show()
