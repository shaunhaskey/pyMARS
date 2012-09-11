'''
Not entirely sure what this script does...
appears to be very bad interpolation????
'''

import pickle
import numpy as num
import matplotlib.pyplot as pt
import scipy.interpolate as interpolate

list = pickle.load(open('/home/srh112/NAMP_datafiles/RESULTS_Plasma_r_2.34_z_0.0'))

q95_array = num.array(list[0])
Bn_Div_Li_array = num.array(list[1])

R_val = num.array(list[2])
Z_val = num.array(list[3])
B1_val = num.array(list[4])
B2_val = num.array(list[5])
B3_val = num.array(list[6])

newfuncB1 = interpolate.Rbf(q95_array,Bn_Div_Li_array, num.abs(B1_val),function='cubic')
newfuncB2 = interpolate.Rbf(q95_array,Bn_Div_Li_array, num.abs(B2_val),function='cubic')
newfuncB3 = interpolate.Rbf(q95_array,Bn_Div_Li_array, num.abs(B3_val),function='cubic')

#newfuncB1 = interpolate.interp2d(q95_array,Bn_Div_Li_array, num.abs(B1_val),kind='linear')

q95min = 2.15
q95max =3.7
Bn_Div_Li_min = 0.75 # 0.36
Bn_Div_Li_max = 2.5 #2.7 

xnew, ynew = num.mgrid[q95min:q95max:200j, Bn_Div_Li_min:Bn_Div_Li_max:200j]
newvalsB1 = newfuncB1(xnew,ynew)
newvalsB2 = newfuncB2(xnew,ynew)
newvalsB3 = newfuncB3(xnew,ynew)

B1_err = []; B2_err = []; B3_err = []

#Print Check for errors!
for i in range(0,len(q95_array)):
    if q95_array[i]<q95max and q95_array[i]>q95min:
        if Bn_Div_Li_array[i]<Bn_Div_Li_max and Bn_Div_Li_array[i]>Bn_Div_Li_min:
            B1_err.append((newfuncB1(q95_array[i],Bn_Div_Li_array[i])-num.abs(B1_val[i]))/num.abs(B1_val[i])*100)
            B2_err.append((newfuncB2(q95_array[i],Bn_Div_Li_array[i])-num.abs(B2_val[i]))/num.abs(B2_val[i])*100)
            B3_err.append((newfuncB3(q95_array[i],Bn_Div_Li_array[i])-num.abs(B3_val[i]))/num.abs(B3_val[i])*100)

fig = pt.figure()
ax = fig.add_subplot(111)
ax.plot(B1_err, 'b.')
ax.plot(B2_err, 'r.')
ax.plot(B3_err, 'k.')
#ax.plot(q95_array, num.abs(B3_val),'k.')
fig.canvas.draw()
fig.show()


print num.min(newvalsB1), num.min(num.abs(B1_val)), num.max(newvalsB1),num.max(num.abs(B1_val))
print num.min(newvalsB2), num.min(num.abs(B2_val)), num.max(newvalsB2),num.max(num.abs(B2_val))
print num.min(newvalsB3), num.min(num.abs(B3_val)), num.max(newvalsB3),num.max(num.abs(B3_val))

#fig = pt.figure()
#ax = fig.add_subplot(111)
#ax.plot(q95_array, Bn_Div_Li_array,'.')
#ax.plot(q95_array, num.abs(B1_val),'r.')
#ax.plot(q95_array, num.abs(B2_val),'b.')
#ax.plot(q95_array, num.abs(B3_val),'k.')
#fig.canvas.draw()
#fig.show()
list_new_data = [newvalsB1, newvalsB2, newvalsB3]

for i in range(0,len(list_new_data)):
    fig = pt.figure()
    ax = fig.add_subplot(111)
    image = ax.imshow(list_new_data[i],extent=[q95min,q95max,Bn_Div_Li_min,Bn_Div_Li_max],origin='lower')
    fig.colorbar(image)
    ax.plot(q95_array, Bn_Div_Li_array,'k,')
    ax.set_xlabel('q95')
    ax.set_ylabel('Bn/Li')
    ax.set_title('B'+str(i+1))
    fig.canvas.draw()
    fig.show()
    
