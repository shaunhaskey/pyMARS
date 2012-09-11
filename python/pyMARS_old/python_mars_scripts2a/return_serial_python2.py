import pickle
import numpy as num
import os
from PythonMARS_funcs import *
import RZfuncs as post_func
from results_class import *
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

def plot_pest(dir_base, pic_name, title,phasing):
    dir1_v_u = dir_base + 'RUNrfa_COILupper.vac/'
    dir1_v_l = dir_base + 'RUNrfa_COILlower.vac/'
    dir1_p_u = dir_base + 'RUNrfa_COILupper.p/'
    dir1_p_l = dir_base + 'RUNrfa_COILlower.p/'
    combined_data = results_combine(dir1_v_u, dir1_v_l, dir1_p_u, dir1_p_l)
    combined_data.get_plasma_data_upper()
    combined_data.get_plasma_data_lower()
    combined_data.get_vac_data_upper()
    combined_data.get_vac_data_lower()

    combined_data.data_p_l.get_PEST()
    combined_data.data_p_u.get_PEST()
    combined_data.data_v_l.get_PEST()
    combined_data.data_v_u.get_PEST()

    combined_data.comb_p = copy.deepcopy(combined_data.data_p_l)
    combined_data.comb_v = copy.deepcopy(combined_data.data_v_l)
    combined_data.comb_p_only = copy.deepcopy(combined_data.data_p_l)
    combined_data.comb_p.R,combined_data.comb_p.Z, combined_data.comb_p.B1, combined_data.comb_p.B2, combined_data.comb_p.B3, combined_data.comb_p.Bn, combined_data.comb_p.BMn,combined_data.comb_p.BnPEST = combine_data(combined_data.data_p_u, combined_data.data_p_l, phasing)
    combined_data.comb_v.R,combined_data.comb_v.Z, combined_data.comb_v.B1, combined_data.comb_v.B2, combined_data.comb_v.B3, combined_data.comb_v.Bn, combined_data.comb_v.BMn,combined_data.comb_v.BnPEST = combine_data(combined_data.data_v_u, combined_data.data_v_l, phasing)

    combined_data.comb_p.plot1(title=title, fig_name=pic_name,fig_show =0,clim_value=[0,1],phase_correction=combined_data.comb_v.BnPEST)
    #combined_data.comb_p.plot1(title=title, fig_name=pic_name,fig_show =0)
    #combined_data.comb_v.plot1(title=title, fig_name=pic_name,fig_show =0,clim_value=[0,1])
    #combined_data.comb_v.plot1(title=title, fig_name=pic_name,fig_show =0)
    combined_data.comb_p_only.B1 = combined_data.comb_p.B1 - combined_data.comb_v.B1
    combined_data.comb_p_only.B2 = combined_data.comb_p.B2 - combined_data.comb_v.B2
    combined_data.comb_p_only.B3 = combined_data.comb_p.B3 - combined_data.comb_v.B3
    combined_data.comb_p_only.Bn = combined_data.comb_p.Bn - combined_data.comb_v.Bn
    combined_data.comb_p_only.BMn = combined_data.comb_p.BMn - combined_data.comb_v.BMn
    
    combined_data.comb_p_only.get_PEST()
    #combined_data.comb_p_only.plot1(title=title, fig_name=pic_name,fig_show =0,clim_value=[0,1],phase_correction=combined_data.comb_v.BnPEST)

    return combined_data

def plot_difference(combined_data_list):
    serial1=1
    serial2=2
    for serial2 in range(2,num.max(combined_data_list.keys())+1):
        new_data = copy.deepcopy(combined_data_list[serial2].comb_p)
        #new_data.B1 = combined_data_list[serial2].comb_p.B1/combined_data_list[serial1].comb_p.B1
        #new_data.B2 = combined_data_list[serial2].comb_p.B2/combined_data_list[serial1].comb_p.B2
        #new_data.B3 = combined_data_list[serial2].comb_p.B3/combined_data_list[serial1].comb_p.B3
        #new_data.Bn = combined_data_list[serial2].comb_p.Bn/combined_data_list[serial1].comb_p.Bn
        new_data.BnPEST = num.abs(combined_data_list[serial2].comb_p.BnPEST)/num.abs(combined_data_list[serial2-1].comb_p.BnPEST)
        #new_data.get_PEST()
        print new_data.BnPEST
        print num.mean(new_data.BnPEST)
        print 'starting plot1'
        new_data.plot1(title='wtf', fig_name='/scratch/haskeysr/temp/adjacent_'+str(serial2)+'.png',fig_show =0,clim_value=[0,4])
#        new_data.plot1(title=str(serial2), fig_name=str(serial2)+'.png',fig_show =0,clim=None)



ROTE_value = 0
q95_target = 3.5
project_name = 'project1_new_eq'
project_dir = '/scratch/haskeysr/mars/'+ project_name + '/'

name = project_dir + '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
project_dict = pickle.load(open(name))
Bn_values = num.arange(1,3,0.1)#1,3,0.1)
#Bn_values = num.array([1,2])

results={}

serial = 1
for i in Bn_values:
    if PEST==1:
        pic_name = '/scratch/haskeysr/temp/phase_vaccorr4_V2_plot_'+str(ROTE_value)+'_'+'q95_'+str(q95_target)+'_BnLi_'+str(i)+'_PEST_Python.png'
    else:
        pic_name = '/scratch/haskeysr/temp/phase_vaccorr4_V2_plot_'+str(ROTE_value)+'_'+'q95_'+str(q95_target)+'_BnLi_'+str(i)+'.png'
    print pic_name
    current_serial = find_serial(project_dict, i,q95_target)
    mars_dir = project_dict['sims'][current_serial]['dir_dict']['mars_plasma_dir']
    mars_base_dir = project_dict['sims'][current_serial]['dir_dict']['mars_dir']
    os.chdir(mars_dir)
    if PEST == 1:
        os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F_EQAC')
        os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F')
        os.system('ln -sf ../../cheaserun_PEST/RMZM_F RMZM_F_PEST')
        os.system('ln -sf PROFEQ.OUT PROFEQ_PEST')
    else:
        pass
    
    Bn_Div_Li_real = project_dict['sims'][current_serial]['BETAN']/ project_dict['sims'][current_serial]['LI']
    #title_command = "title('|Bn| G/kA,%ddeg,q95:%.2f,Bn/Li:%.2f ')\n"%(ROTE_value, project_dict['sims'][current_serial]['Q95'], Bn_Div_Li_real)
    title = "|Bn| G/kA,%ddeg,q95:%.2f,Bn/Li:%.2f"%(ROTE_value, project_dict['sims'][current_serial]['Q95'], Bn_Div_Li_real)
    phasing=ROTE_value
    results[1] = plot_pest(mars_base_dir,pic_name,title,phasing)
    serial+=1
#plot_difference(results)
    #post_func.pest_plot(mars_dir,pic_name,title)

