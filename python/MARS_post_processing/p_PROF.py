import numpy as np
import matplotlib.pyplot as pt

filename = '/usc-data/task/efit/lao/kinetic/157312_n3RMP_Raffi/EQDSK/p157312.04575_RNB1_T257'
final_names = ['PROFDEN', 'PROFTE', 'PROFTI', 'PROFROT']
search_terms = ['ne','te','ti','omeg']

with file(filename,'r') as handle:lines = handle.readlines()
    
for fname, s_term in zip(final_names, search_terms):
    success = 0
    for i in range(len(lines)):
        if lines[i].find(s_term)>=0: 
            success = 1
            break
    if success:
        n_terms = int(lines[i].split(' ')[0])
        print n_terms, lines[i], fname, s_term
        data = lines[i+1:i+1+n_terms]
        with file(fname,'w') as handle:
            handle.writelines(data)

        

