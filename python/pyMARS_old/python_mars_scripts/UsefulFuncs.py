import pickle
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as pt
file_location = '/u/haskeysr/mars/project_new_eq_test/2_project_new_eq_test_setup_directories.pickle'
file_location = '/scratch/haskeysr/mars/project1_new_eq/2_project1_new_eq_setup_directories.pickle'
#file_location = '/u/haskeysr/mars/project1/2_setup_directories.pickle'

def plot_q95_Bn_Div_Li(file_location):
    
    project_dict = pickle.load(open(file_location))
    
    q95 = []
    Bn_Div_Li = []
    qmult = []
    pmult = []

    for i in  project_dict['sims'].keys():
        q95.append(project_dict['sims'][i]['Q95'])
        Bn_Div_Li.append(project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI'])
        qmult.append(project_dict['sims'][i]['QMULT'])
        pmult.append(project_dict['sims'][i]['PMULT'])

    print 'figure section'
    fig = pt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax.plot(qmult,pmult,'.')
    ax.set_xlabel('qmult')
    ax.set_ylabel('pmult')

    ax2.plot(Bn_Div_Li,q95,'.')
    ax2.set_xlabel('B_n/Li')
    ax2.set_ylabel('q95')
    print 'saving figure'
    fig.savefig('temp4B.png')
    print 'figure end'
    
plot_q95_Bn_Div_Li(file_location)
