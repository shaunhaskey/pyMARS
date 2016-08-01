import pickle
import numpy as num
import os
from PythonMARS_funcs import *


PEST = 1
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
q95_target = 6.
project_name = 'project1_new_eq'
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'

name = project_dir + '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
project_dict = pickle.load(open(name))
Bn_values = num.arange(1,3,0.1)#1,3,0.1)

if PEST == 1:
    mat_commands = "addpath('/u/haskeysr/matlab/RZplot_new/')\n"
else:
    mat_commands = "addpath('/u/haskeysr/matlab/RZplot3/')\n"

for i in Bn_values:
    if PEST==1:
        pic_name = '/scratch/haskeysr/temp/V2_plot_'+str(ROTE_value)+'_'+'q95_'+str(q95_target)+'_BnLi_'+str(i)+'_PEST.png'
    else:
        pic_name = '/scratch/haskeysr/temp/V2_plot_'+str(ROTE_value)+'_'+'q95_'+str(q95_target)+'_BnLi_'+str(i)+'.png'
    print pic_name
    current_serial = find_serial(project_dict, i,q95_target)
    mars_dir = project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']
    #print '*****************************'
    
    #print project_dict['sims'][current_serial]['dir_dict']['chease_dir']
    #print len(project_dict['details']['base_dir'])
    #print project_dict['sims'][current_serial]['dir_dict']['chease_dir'].lstrip(project_dict['details']['base_dir'])
    #template_dir = '/u/haskeysr/mars/project1_new_eq/' + project_dict['sims'][current_serial]['dir_dict']['chease_dir'][len(project_dict['details']['base_dir']):]
    #chease_dir = project_dict['sims'][current_serial]['dir_dict']['chease_dir']
    #command = 'cp ' + template_dir + '* ' + chease_dir
    #os.system(command)
    #print command

    #print '********************************'
    os.chdir(mars_dir)
    if PEST == 1:
        os.system('cp /u/haskeysr/mars/templates/Plot_Results_PEST.m Plot_Results_PEST.m')
        os.system('cp /u/haskeysr/mars/templates/MacMainTest.m MacMainTest.m')
        os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F_EQAC')
        os.system('ln -sf ../../cheaserun_PEST/RMZM_F RMZM_F_PEST')
        os.system('ln -sf PROFEQ.OUT PROFEQ_PEST')
        SDIR_newline = "SDIR='"+ project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']+"';"
        modify_input_file('MacMainTest.m', 'SDIR=', SDIR_newline)
        modify_input_file('Plot_Results_PEST.m', 'SDIR=', SDIR_newline)
        #replace_value('Plot_Results1.m','Mac.Nm2', ';', str(1+int(abs(project_dict['sims'][current_serial]['M1'])+abs(project_dict['sims'][current_serial]['M2']))))
        #
    else:
        os.system('cp /scratch/haskeysr/mars/Plot_Results1.m Plot_Results1.m')
        SDIR_newline = "SDIR='"+ project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']+"';"
        modify_input_file('Plot_Results1.m', 'SDIR=', SDIR_newline)
        replace_value('Plot_Results1.m','Mac.Nm2', ';', str(1+int(abs(project_dict['sims'][current_serial]['M1'])+abs(project_dict['sims'][current_serial]['M2']))))

    
    #Copy the template!


    mat_commands += 'close all;clear all;\n'
    mat_commands += 'cd ' + project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']+'\n'
    if PEST ==1:
        mat_commands += "Plot_Results_PEST\n"
        mat_commands += "set(gca,'CLim',[0, 0.9])\n"
    else:
        mat_commands += "Plot_Results1\n"
        
    Bn_Div_Li_real = project_dict['sims'][current_serial]['BETAN']/ project_dict['sims'][current_serial]['LI']
    title_command = "title('|Bn| G/kA,%ddeg,q95:%.2f,Bn/Li:%.2f ')\n"%(ROTE_value, project_dict['sims'][current_serial]['Q95'], Bn_Div_Li_real)
#    title_command = "title('Re[Bn] G/kA,%ddeg,q95:%.2f,Bn/Li:%.2f ')\n"%(ROTE_value, project_dict['sims'][current_serial]['Q95'], Bn_Div_Li_real)
#    title_command = "[ax, h3]=suplabel('[Bn] %ddeg,q95:%.2f,Bn/Li:%.2f ','t',[.075 .075 .75 .75])\n"%(ROTE_value, project_dict['sims'][current_serial]['Q95'], Bn_Div_Li_real)
    mat_commands += title_command
#    mat_commands += "set(h3,'FontSize',15)\n"
    mat_commands += "print(gcf,'" + pic_name + "', '-dpng','-r200')\n"

    print i, mars_dir


mat_commands += "quit\n"
os.chdir(project_dict['details']['base_dir'])
file = open('mat_plot_mars_commands.m','w')
file.write(mat_commands)
file.close()

