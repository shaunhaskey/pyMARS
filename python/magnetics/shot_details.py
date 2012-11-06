import matplotlib.pyplot as pt
import numpy as np
import data

relevant_data = ['betan', 'q95', 'LI']
shots = [146392, 146398, 146397, 146388, 146382, 148765]
start_times= [3000,3000,3000,3000,2387,2392]
end_times = [4123,3906,4258,4454,4862,5288]
answers = {}
for loc, shot in enumerate(shots):
    answers[shot]={}
    answers[shot]['x']=np.array(range(start_times[loc],end_times[loc],100))
    for i in relevant_data:
        pickup_data = data.Data(i, shot)
        answers[shot][i] = np.interp(answers[shot]['x'], np.array(pickup_data.x).flatten(), np.array(pickup_data.y).flatten())
fig, ax = pt.subplots()
styles = ['x-','.-','o-','x--','.--','o--']
for i in answers.keys():
    x_axis = answers[i]['q95']
    y_axis = answers[i]['betan']/answers[i]['LI']
    ax.plot(x_axis, y_axis, styles[shots.index(i)], label ='%d %d-%dms'%(i,start_times[shots.index(i)],end_times[shots.index(i)]))
ax.legend(loc='best')
ax.set_xlim([2,5])
ax.set_ylim([1.5,4])
ax.set_xlabel('q95')
ax.set_ylabel('Beta_n/L_i')
ax.set_title('Experimental dataset')
fig.canvas.draw(); fig.show()

'''











q95_const = [3.75, 3.77, 3.6]
Bn = [2.02, 2.22, 2.1]

fig, ax = pt.subplots()
ax.plot(q95_const, Bn, 'x', label='const shots')

Bn_range = np.arange(2,2.3,0.1)
q95_range = Bn_range*0+3.3

ax.plot(q95_range, Bn_range, '.-', label='Bn ramp1 146398')

Bn_range = np.arange(2,2.4,0.1)
q95_range = Bn_range*0+3.75

ax.plot(q95_range, Bn_range, '.-', label='Bn ramp2 148765')

q95_range = np.linspace(4,3.5,10)
Bn_range = np.linspace(2.1,1.9,10)

ax.plot(q95_range, Bn_range, '.-', label='q95,Bn ramp2 146382')
leg = ax.legend(loc='best',fancybox=True)
ax.set_xlabel('q95')
ax.set_ylabel('beta_n')

ax.set_xlim(2,7)
ax.set_ylim(1,5)
fig.canvas.draw(); fig.show()
'''
