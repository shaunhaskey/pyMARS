'''
7/11/2012 SH
This reads in several pickle files and outputs what the q95, bn/li and certain pickup output is
It is meant as a starting point for comparing the MARS results with the time varying experimental
results
'''


import numpy as np
import pickle


file_names = ['/home/srh112/NAMP_datafiles/mars/shot146382_single2/shot146382_single2_post_processing.pickle',
              '/home/srh112/NAMP_datafiles/mars/shot146388_single2/shot146388_single2_post_processing.pickle',
              '/home/srh112/NAMP_datafiles/mars/shot146397_3305/shot146397_3305_post_processing.pickle',
              '/home/srh112/NAMP_datafiles/mars/shot146397_3815/shot146397_3815_post_processing.pickle',
              '/home/srh112/NAMP_datafiles/mars/shot146398_3305/shot146398_3305_post_processing.pickle',
              '/home/srh112/NAMP_datafiles/mars/shot146398_3515/shot146398_3515_post_processing.pickle',
              '/home/srh112/NAMP_datafiles/mars/shot146398_3815/shot146398_3815_post_processing.pickle']
#'/home/srh112/NAMP_datafiles/mars/shot146397_3515/shot146397_3515_post_processing.pickle',

for i in file_names:
    a = pickle.load(file(i,'r'))
    for loc_tmp, j in enumerate(a['details']['pickup_coils']['probe']):
        if j.find('66M')!=-1:probe_loc=loc_tmp
    #probe_loc = a['details']['pickup_coils']['probe'].index('66M')
    shot_time =  a['details']['shot_time']
    shot_num = a['details']['shot']
    betaN_li = a['sims'][1]['BETAN']/a['sims'][1]['LI']
    q95 = a['sims'][1]['Q95']
    try: 
        a['sims'][1].keys().index('vacuum_lower_response4')
        upper_lower = 1
    except ValueError:
        upper_lower = 0
    print '====== probe loc %d, shot %d, shot_time %d, q95 %.2f, beta_N/Li %.2f'%(probe_loc,shot_num, shot_time, q95, betaN_li)
    if upper_lower:
        print 'upper, lower'
        vac_quant = a['sims'][1]['vacuum_lower_response4'][probe_loc] + a['sims'][1]['vacuum_upper_response4'][probe_loc]
        plas_quant = a['sims'][1]['plasma_lower_response4'][probe_loc] + a['sims'][1]['plasma_upper_response4'][probe_loc]
        print np.abs(vac_quant)
        print np.abs(plas_quant)
    else:
        print 'single'
        print np.abs(a['sims'][1]['vacuum_response4'][probe_loc])
        print np.abs(a['sims'][1]['plasma_response4'][probe_loc])
