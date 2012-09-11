'''
Find nearest q95 value in a dictionary and print out its details so that it can be found
'''

import numpy as num
import pickle
ROTE_value= -300

name = '9_project1_new_eq_FEEDI_'+str(ROTE_value)+'_coil_outputs.pickle'
project_dict = pickle.load(open(name))

q_value = 5.247
q_range = 0.1
for jjj in project_dict['sims'].keys():
    if  (q_value - q_range) < project_dict['sims'][jjj]['QMAX'] < (q_value + q_range):
        a = project_dict['sims'][jjj]
        print 'Ser:%d QMAX:%.4f,pmult:%.3f,qmult:%.3f,Bn:%.4f,Bn/Li:%.4f,dcon:%.2f, %.2f, %.2f, %.2f'%(jjj,a['QMAX'], a['PMULT'],a['QMULT'],a['BETAN'],a['BETAN']/a['LI'],a['WTOTN1'],a['WTOTN2'],a['WTOTN3'],a['WWTOTN1'])
