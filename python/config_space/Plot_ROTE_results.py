import pickle
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate


var_name = 'ROTE'
var_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
var_list = num.arange(0,1.55,0.05)
var_list = num.arange(0.8,1,0.01) #ROTE3

probe  = [ 'dBp_upper - 67A', 'dBp_mid - 66M', 'dBp_lower - 67B', 'dBr_ext - ESL', 'dBr_mid - ISL','dBr_upper - UISL','dBr_lower  - LISL']

project_name = 'project_vary_ROTE3'
passes = 0
fails = 0
calc = 'plasma_response2'


all_values = []

for iii in range(0,len(probe)):
    current_coil_values = []
    for jjj in range(0,len(var_list)):
        var_value = var_list[jjj]
        name = '9_' + project_name + '_' + var_name + '_' + str(var_value) + '_coil_outputs.pickle'
        print name
        project_dict = pickle.load(open(name))
        try:
            for i in project_dict['sims'].keys():
                current_coil_values.append(project_dict['sims'][i][calc][iii])
                passes+=1
        except:
            fails+=1
            print 'FAIL'
    print current_coil_values
    all_values.append(current_coil_values)

print 'pass : %d, fails : %d'%(passes, fails)

print var_list
print all_values[0]

for i in range(0,len(all_values)):
    fig = pt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax.plot(var_list, num.abs(all_values[i]), 'b.-')
    ax.set_xlabel('ROTE_scaling')
    ax.set_ylabel('abs(B)')
    ax.set_title(probe[i])
    ax2.plot(var_list, num.angle(all_values[i]), 'b.-')
    ax2.set_xlabel('ROTE_scaling')
    ax2.set_ylabel('phase (B)')
    fig.canvas.draw()
    fig.show()
