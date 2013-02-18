import matplotlib.pyplot as pt
import RZfuncs
import numpy as np

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

def extract_surfmn_data_m(filename, m, r_val):
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
    print 'nst', nst, ns.shape
    rvals = np.fromstring(a[1],dtype=float,sep=" ")
    qvals = np.fromstring(a[2],dtype=float,sep=" ")
    print rvals.shape
    m_loc = np.argmin(np.abs(ms-m))
    print 'm_loc', m,  m_loc, ms[m_loc]
    r_loc = np.argmin(np.abs(rvals-r_val)),
    print 'r_loc', r_val, r_loc, rvals[r_loc]

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

    print adat[r_loc,m_loc,0:11]
    return adat, adat[r_loc,m_loc,:], ns

fig, ax = pt.subplots(nrows = 3)

N = 6; n = 2
I = np.array([1.,-1.,0.,1,-1.,0.])

I0EXP,fft_freq_1, values_1 = RZfuncs.I0EXP_calc_real(n,I, produce_plot=1, plot_axes = ax)

#fig.canvas.draw(); fig.show()



m = -10; r_val = 0.95
adat, relevant_n, ns = extract_surfmn_data_m('/home/srh112/Desktop/Test_Case/RZPlot_PEST_Test/SURF146382.03230.ph000.pmz/surfmn.out.idl3d', m, r_val)
ax[2].stem(range(0,len(relevant_n.flatten())), relevant_n.flatten())
ax[2].set_xlim([0,11])
for i in ax:
    i.set_xlabel('')
    i.set_ylabel('')
    i.set_title('')
#ax[2].set_xlabel('n')
#ax[2].set_ylabel('Mode Amplitude G/kA')
#ax[2].set_title(r'm=%d, $\sqrt{\psi_N}=%.2f$'%(m, r_val))
fig.canvas.draw(); fig.show()
