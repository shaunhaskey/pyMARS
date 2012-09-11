import pickle
import numpy as num
import os
from PythonMARS_funcs import *
import RZfuncs as post_func

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
Bn_values = num.arange(1,3,0.5)#1,3,0.1)
#Bn_values = num.array([1,2])


for i in Bn_values:
    if PEST==1:
        pic_name = '/scratch/haskeysr/temp/V2_plot_'+str(ROTE_value)+'_'+'q95_'+str(q95_target)+'_BnLi_'+str(i)+'_PEST_Python.png'
    else:
        pic_name = '/scratch/haskeysr/temp/V2_plot_'+str(ROTE_value)+'_'+'q95_'+str(q95_target)+'_BnLi_'+str(i)+'.png'
    print pic_name
    current_serial = find_serial(project_dict, i,q95_target)
    mars_dir = project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']

    os.chdir(mars_dir)
    if PEST == 1:
        os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F_EQAC')
        os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F')
        os.system('ln -sf ../../cheaserun_PEST/RMZM_F RMZM_F_PEST')
        os.system('ln -sf PROFEQ.OUT PROFEQ_PEST')
    else:
        pass
    
    #Copy the template!


        
    Bn_Div_Li_real = project_dict['sims'][current_serial]['BETAN']/ project_dict['sims'][current_serial]['LI']
    #title_command = "title('|Bn| G/kA,%ddeg,q95:%.2f,Bn/Li:%.2f ')\n"%(ROTE_value, project_dict['sims'][current_serial]['Q95'], Bn_Div_Li_real)
    title = "|Bn| G/kA,%ddeg,q95:%.2f,Bn/Li:%.2f"%(ROTE_value, project_dict['sims'][current_serial]['Q95'], Bn_Div_Li_real)
    post_func.pest_plot(mars_dir,pic_name,title)
