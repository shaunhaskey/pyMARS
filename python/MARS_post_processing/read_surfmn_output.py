import numpy as np
import matplotlib.pyplot as pt

def extract_surfmn_data(filename, n):
    a = file(filename,'r').readlines()
    tmp = a[0].rstrip('\n').split(" ")
    unfinished=1
    while unfinished:
        try:
            tmp.remove("")
            #print 'removed something'
        except ValueError:
            #print 'finished?'
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
    zdat = adat[:,:,n]
    xdat = np.tile(ms,(zdat.shape[0],1)).transpose()
    ydat = np.tile(rvals, (zdat.shape[1],1))
    zdat = zdat.transpose()

    return qlvals, xdat, ydat, zdat
n = 2
qlvals, xdat, ydat, zdat = extract_surfmn_data('/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/SURF146382.03230.ph000.pmz/surfmn.out.idl3d', n)
#xdat = np.tile(ms,(zdat.shape[0],1)).transpose()
#ydat = np.tile(rvals, (zdat.shape[1],1))
#zdat = zdat.transpose()
import h5py

tmp_file = h5py.File('/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/mars_files/RUNrfa.vac/spectral_info_pmz.h5')
stored_data = tmp_file.get('1')
zdat1 = stored_data[0][0]; xdat1 = stored_data[0][1]; ydat1 = stored_data[0][2]

fig, ax = pt.subplots()
#color_plot = ax.pcolor(ms,rvals,zdat,cmap='hot')
color_plot = ax.pcolor(xdat,ydat,zdat,cmap='hot')
color_plot.set_clim([0,1.6])
ax.plot(qlvals*n,ydat[1,:],'b--')
ax.set_xlim([-30,30])
ax.set_ylim([0,1])
pt.colorbar(color_plot, ax = ax)
fig.canvas.draw()
fig.show()

