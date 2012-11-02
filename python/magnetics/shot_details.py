import matplotlib.pyplot as pt
import numpy as np
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
