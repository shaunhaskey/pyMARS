import os

string_name = '/scratch/haskeysr/corsica_test6/ml_new_'
destination = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/efit/'
tot = 1

for i in range(0,100):
    dir = string_name + str(i) + '/'
    if os.path.exists(dir):
        os.chdir(dir)
        os.system('rsync -av --ignore-existing EXPEQ* ' + destination)
        #os.system('cp -vn EXPEQ* ' + destination)
        file_name = dir + 'stab_setup_results.dat'
        file = open(file_name,'r')
        current_results = file.readlines()
        file.close()

        if tot ==1:
            together = current_results
        else:
            for j in range(3,len(current_results)):
                together.append(current_results[j])
        tot += 1
file2 = open(destination +'stab_setup_results_new.dat','w')
file2.writelines(together)
file2.close()



'''
for i in range(40,76):
    dir = '/u/haskeysr/mars/eq_from_matt/corsica_test4/ml' + str(i) + '/'
    a = read_stab_results(dir + 'stab_setup_results.dat')
    single_p = []
    single_q = []
    for j in a.keys():
        b[j]=a[j]
        q.append(a[j]['QMULT'])
        p.append(a[j]['PMULT'])
        single_p.append(a[j]['PMULT'])
        single_q.append(a[j]['QMULT'])
        tuple_list.append((a[j]['PMULT'],a[j]['QMULT']))
    print ml,i, max(single_p), min(single_p), max(single_q), min(single_q)
sorted(tuple_list,key=itemgetter(0,1))
'''
