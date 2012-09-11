import numpy as num
import data as pydata

data_list = num.loadtxt('/u/hansonjm/var/data/n2ams2011/plas_resp_terms_LISL.txt', comments='#',skiprows=19)
q95_results = []
betan_results = []
li_results = []
shot_list = []
start_time_list = []
end_time_list = []
for i in range(0,data_list.shape[0]):
    current_shot = data_list[i,0]
    start_time = data_list[i,1]
    end_time = data_list[i,2]
    beta_N = pydata.Data('betan',int(current_shot))
    li = pydata.Data('li',int(current_shot))
    q95 = pydata.Data('q95',int(current_shot))

    start_loc = num.argmin(num.abs(beta_N.x-start_time))
    end_loc = num.argmin(num.abs(beta_N.x-end_time))
    mean_betan = num.mean(beta_N.y[start_loc:end_loc+1])
    mean_li = num.mean(li.y[start_loc:end_loc+1])
    mean_q95 = num.mean(q95.y[start_loc:end_loc+1])
    print current_shot, start_time, end_time, mean_betan, mean_li, mean_q95
    q95_results.append(mean_q95)
    betan_results.append(mean_betan)
    li_results.append(mean_li)
    shot_list.append(current_shot)
    start_time_list.append(start_time)
    end_time_list.append(end_time)

num.savetxt('test.txt',num.transpose(num.array([shot_list,start_time_list,end_time_list,betan_results,li_results,q95_results])),fmt='%.3f')
