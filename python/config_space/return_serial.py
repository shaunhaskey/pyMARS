import pickle
import numpy as num
import os

def find_serial(project_dict, Bn_Div_Li_target,q95_target):
    passes = 0
    fails = 0
    overall_distance = 1000.
    for jjj in project_dict['sims'].keys():
        try:
            current_Bn_Div_Li = project_dict['sims'][jjj]['BETAN']/project_dict['sims'][jjj]['LI']
            current_q95 = project_dict['sims'][jjj]['Q95']
            distance = (current_Bn_Div_Li-Bn_Div_Li_target)**2+(current_q95-q95_target)**2
            if distance < overall_distance:
                overall_distance = distance *1.
                current_serial = jjj*1
            passes+=1
        except:
            fails+=1
            print 'failed'
    return current_serial


ROTE_value = -300
Bn_Div_Li_target = 2.8
q95_target = 4.

name = '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
project_dict = pickle.load(open(name))
Bn_values = num.arange(1,3,0.2)

mat_commands = "addpath('/u/haskeysr/matlab/RZplot3/')\n"

for i in Bn_values:
    pic_name = '/scratch/haskeysr/temp/plot_'+str(ROTE_value)+'_'+'q95_'+str(q95_target)+'_BnLi_'+str(i)+'.png'
    current_serial = find_serial(project_dict, i,q95_target)
    mars_dir = project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']
    os.chdir(mars_dir)
    #Copy the template!

    os.system('cp /scratch/haskeysr/mars/Plot_Results1.m Plot_Results1.m')

    #modify the template!

    SDIR_newline = "SDIR='"+ project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']+"';"

    modify_input_file('Plot_Results1.m', 'SDIR=', SDIR_newline)
    replace_value('Plot_Results1.m','Mac.Nm2', ';', str(1+int(abs(project_dict['sims'][current_serial]['M1'])+abs(project_dict['sims'][current_serial]['M2']))))

    mat_commands += 'close all;clear all;\n'
    mat_commands += 'cd ' + project_dict['sims'][i]['dir_dict']['mars_plasma_dir']+'\n'
    mat_commands += "Plot_Results1\n"
    mat_commands += "print(gcf,'-dpng'," + pic_name + ')\n'

    print i, matlab_dir_txt


mat_commands += "quit\n"
os.chdir(project_dict['details']['base_dir'])
file = open('mat_plot_mars_commands.txt','w')
file.write(mat_commands)
file.close()



