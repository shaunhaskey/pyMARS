import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt

#rot_pickle = pickle.load(file('/home/srh112/ga_mount/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_mars_run.pickle.bak','r'))

rot_pickle = pickle.load(file('/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_processing_PEST.pickle','r'))

rot_pickle = pickle.load(file('/home/srh112/ga_mount/mars/single_run_through_test_142614_V2/single_run_through_test_142614_V2_post_mars_run.pickle','r'))
rote_list = []
total_res = []
for i in rot_pickle['sims'].keys():
    rote_list.append(rot_pickle['sims'][i]['MARS_settings']['<<ROTE>>'])
    total_res.append(rot_pickle['sims'][1]['responses']['total_resonant_response_upper_integral']+rot_pickle['sims'][1]['responses']['total_resonant_response_lower_integral'])

res_list = []
#for i in res_pickle['sims'].keys():
#    res_list.append(res_pickle['sims'][i]['MARS_settings']['<<ETA>>'])
#    total_res.append(res_pickle['sims'][1]['responses']['total_resonant_response_upper_integral']+res_pickle['sims'][1]['responses']['total_resonant_response_lower_integral'])
fig, ax = pt.subplots()
ax.plot(rote_list, total_res, 'o')
fig.canvas.draw(); fig.show()
