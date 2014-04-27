import numpy as np
import matplotlib.pyplot as pt
import os

base_dir = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/efit/'
a = os.listdir(base_dir)
times = []
temps = []
res = []
rots = []
for i in  a: 
    tmp = np.loadtxt(base_dir+i + '/PROFTE', skiprows = 1)
    tmp2 = np.loadtxt(base_dir+i + '/PROFROT', skiprows = 1)
    times.append(int(i))
    temps.append(tmp[0,1])
    rots.append(tmp2[0,1])
    res.append(1.e-7/((3200/temps[-1])**-1.5))
    print tmp[0,:]
fig, ax = pt.subplots(nrows = 3, sharex = True)
ax[0].plot(times, temps, 'o')
ax[1].plot(times, res, 'o')
ax[1].plot(times, np.array(res)/10., 'x')
ax[2].plot(times, rots, 'o')
print 'res'
print '[' + ','.join(['{:.3e}'.format(i) for i in res]) + ']'
print 'rots'
print '[' + ','.join(['{:.3e}'.format(i) for i in rots]) + ']'
print 'times'
print '[' + ','.join(['{}'.format(i) for i in a]) + ']'


fig.canvas.draw(); fig.show()
