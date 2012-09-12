import numpy as np
import matplotlib.pyplot as pt
a = file('/u/haskeysr/surfmn_tmp/SURF146382.03230.ph000.mpm/surfmn.out.idl3d','r').readlines()
tmp = a[0].rstrip('\n').split(" ")
unfinished=1
while unfinished:
    try:
        tmp.remove("")
        print 'removed something'
    except ValueError:
        print 'finished?'
        unfinished = 0
print tmp
nst = int(tmp[0])
nfpts = int(tmp[1])
irpt = int(tmp[2])
iradvar = int(tmp[3])
khand = int(tmp[4])
gfile = tmp[5]
imax = nst -1
jmax = 2*nfpts
kmax = nfpts

ms = np.arange(-nfpts, nfpts+1,1)
ns = np.arange(0,nst+1,1)

rvals = np.fromstring(a[1],dtype=float,sep=" ")
qvals = np.fromstring(a[2],dtype=float,sep=" ")


adat = np.zeros((nst, 2*nfpts+1, nfpts+1),dtype=float)
line_num = 3
for i in range(0,nst):
    for j in range(0,2*nfpts+1):
        tmp = np.fromstring(a[line_num],dtype=float,sep=" ")
        adat[i,j,:]=tmp[:]
        line_num += 1
qlvals = np.fromstring(a[line_num],dtype=float,sep=" ")
n=2
zdat = adat[:,:,n]

fig, ax = pt.subplots()
color_plot = ax.pcolor(ms,rvals,zdat,cmap='hot')
color_plot.set_clim([0,1.6])
ax.plot(qlvals*n,rvals,'b--')
ax.set_xlim([-30,30])
ax.set_ylim([0,1])
pt.colorbar(color_plot, ax = ax)
fig.canvas.draw()
fig.show()

