import pickle
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate

#name = '9_coil_outputs2.pickle'
#name = '9_output_interp.pickle'

#name = 'test.pickle'

#name = '9_coil_outputs_TESTnewInterp.pickle'

name = '9_project1_new_eq_FEEDI_0_coil_outputs.pickle'
name = '9_project1_new_eq_FEEDI_-240_coil_outputs.pickle'

#name = '9_benchmark2_coil_outputs.pickle'


project_dict = pickle.load(open(name))

probe  = [ 'dBp_upper - 67A', 'dBp_mid - 66M', 'dBp_lower - 67B', 'dBr_ext - ESL', 'dBr_mid - ISL','dBr_upper - UISL','dBr_lower  - LISL']

calc = 'plasma_response4'
list_images = []
IFEED = []

q95_list = []
Bn_Div_Li_list = []
coil1 = []
passes = 0
fails = 0
NW_list = []
for i in project_dict['sims'].keys():
    if project_dict['sims'][i]['IFEED'][0]>= project_dict['sims'][i]['NW']:
        print "Possible error IFEED value is greater than or equal to NW :" + str(project_dict['sims'][i]['NW']) + ', IFEED' + str(project_dict['sims'][i]['IFEED'][0])
        IFEED.append(project_dict['sims'][i]['NW']-1)
    else:
        IFEED.append(project_dict['sims'][i]['IFEED'][0])
    q95_list.append(project_dict['sims'][i]['Q95'])
    Bn_Div_Li_list.append(project_dict['sims'][i]['BETAN']/project_dict['sims'][i]['LI'])
    passes+=1
    NW_list.append(project_dict['sims'][i]['NW'])
print 'pass : %d, fails : %d'%(passes, fails)

q95_array = num.array(q95_list)
Bn_Div_Li_array = num.array(Bn_Div_Li_list)
IFEED_array = num.array(IFEED)
NW_array = num.array(NW_list)

newfuncB1 = interpolate.Rbf(q95_array,Bn_Div_Li_array, IFEED_array,function='linear')
newfuncNW = interpolate.Rbf(q95_array,Bn_Div_Li_array, NW_array,function='linear')

q95min = 2.15
q95max =7#6#3.7
Bn_Div_Li_min = 0.75 # 0.36
Bn_Div_Li_max = 3#2.5 #2.7 

xnew, ynew = num.mgrid[q95min:q95max:30j, Bn_Div_Li_min:Bn_Div_Li_max:30j]



newvalsIFEED = newfuncB1(xnew,ynew)
newvalsNW = newfuncNW(xnew,ynew)

fig = pt.figure()
ax = fig.add_subplot(111)
list_images.append(ax.imshow(newvalsIFEED,extent=[q95min,q95max,Bn_Div_Li_min,Bn_Div_Li_max],origin='lower'))
    #        image.set_clim([0, 2.5])
fig.colorbar(list_images[-1])
ax.plot(q95_array, Bn_Div_Li_array,'k,')
ax.set_xlabel('q95')
ax.set_ylabel('Bn/Li')
fig.canvas.draw()
fig.show()



fig = pt.figure()
ax = fig.add_subplot(111)
list_images.append(ax.imshow(newvalsNW,extent=[q95min,q95max,Bn_Div_Li_min,Bn_Div_Li_max],origin='lower'))
    #        image.set_clim([0, 2.5])
fig.colorbar(list_images[-1])
ax.plot(q95_array, Bn_Div_Li_array,'k,')
ax.set_xlabel('q95')
ax.set_ylabel('Bn/Li')
fig.canvas.draw()
fig.show()


