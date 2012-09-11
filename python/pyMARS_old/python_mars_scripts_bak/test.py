from PythonMARS_funcs import read_stab_results
import os

project_dict = read_stab_results('/u/haskeysr/mars/project2/shot138344/tc_003/efit/stab_setup_results.dat')

good = 0
bad = 0

for i in project_dict.keys():
    name = 'EXPEQ_%d.%.5d_p%d_q%d'%(138344, 2306,int(round(project_dict[i]['PMULT']*1000)),int(round(project_dict[i]['QMULT']*1000)))
    if os.path.exists('/u/haskeysr/mars/project2/shot138344/tc_003/efit/' + name):
        print 'good'
        good +=1
        pass

    else:
        print 'found one : ', name
        bad +=1

print 'good %d, bad %d'%(good, bad)
