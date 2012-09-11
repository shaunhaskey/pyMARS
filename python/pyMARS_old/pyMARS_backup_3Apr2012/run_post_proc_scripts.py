from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as num
import matplotlib.pyplot as pt
import time
import PythonMARS_funcs as pyMARS

N = 6
n = 2
I = num.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)

print I0EXP, 1.0e+3 * 3./num.pi

def diff_comp(one,two):
    diff = (num.real(one)-num.real(two))
    print 'Real ',
    print 'Max diff : %.3e , std_dev : %.3e , mean : %.3e'%(num.max(num.abs(diff)),num.std(num.abs(diff)),num.mean(num.abs(diff)))
    diff = (num.imag(one)-num.imag(two))
    print 'Imag ',
    print 'Max diff  : %.3e , std_dev : %.3e , mean : %.3e'%(num.max(num.abs(diff)),num.std(num.abs(diff)),num.mean(num.abs(diff)))

def diff_real(one,two):
    diff = one-two
    print 'Real ',
    print 'Max diff : %.3e , std_dev : %.3e , mean : %.3e'%(num.max(num.abs(diff)),num.std(num.abs(diff)),num.mean(num.abs(diff)))

def plot_difference(one,two, R, Z, title_text):
    fig = pt.figure()
    ax = fig.add_subplot(111)
    percent_diff = num.abs((one-two))/num.abs(one)*100
    color_plot = ax.pcolor(R,Z,percent_diff)
    pt.colorbar(color_plot, ax=ax)
    color_plot.set_clim([0,1])
    ax.set_title('Percent Difference '+title_text)
    fig.canvas.draw()
    fig.show()
    return color_plot

c = data('/home/srh112/code/pyMARS/test_shot/marsrun/RUNrfa_COILlower.p/')
#c = data('/home/srh112/code/pyMARS/shot146388_single2/qmult1.000/exp1.000/marsrun/RUNrfa.p', I0EXP = I0EXP)
RZ_dir = '/home/srh112/Desktop/MARS-K20110804/TestCase/'
#test RM, ZM

RZplot={}
print 'RM, ZM'
RZplot['RM'] = num.loadtxt(RZ_dir+'RM.real',delimiter=',') + num.loadtxt(RZ_dir+'RM.imag',delimiter=',')*1j
diff_comp(c.RM,RZplot['RM'])
RZplot['ZM'] = num.loadtxt(RZ_dir+'ZM.real',delimiter=',') + num.loadtxt(RZ_dir+'ZM.imag',delimiter=',')*1j
diff_comp(c.ZM,RZplot['ZM'])


#test R, Z
print 'R,Z'
RZplot['R'] = num.loadtxt(RZ_dir+'R.txt',delimiter=',')
diff_real(c.R,RZplot['R'])
RZplot['Z'] = num.loadtxt(RZ_dir+'Z.txt',delimiter=',')
diff_real(c.Z,RZplot['Z'])

#test dRdchi, dZdchi
print 'dRdchi, dZdchi'
RZplot['dRdchi'] = num.loadtxt(RZ_dir+'dRdchi.txt',delimiter=',')
RZplot['dZdchi'] = num.loadtxt(RZ_dir+'dZdchi.txt',delimiter=',')
diff_real(c.dRdchi,RZplot['dRdchi'])
diff_real(c.dZdchi,RZplot['dZdchi'])
#color_plot = plot_difference(RZplot['dRdchi'],c.dRdchi, c.R, c.Z,'dRdchi')
#color_plot = plot_difference(RZplot['dZdchi'],c.dZdchi, c.R, c.Z,'dZdchi')

#test dRds, dZds
print 'dRds, dZds'
RZplot['dRds'] = num.loadtxt(RZ_dir+'dRds.txt',delimiter=',')
RZplot['dZds'] = num.loadtxt(RZ_dir+'dZds.txt',delimiter=',')
diff_real(c.dRds,RZplot['dRds'])
diff_real(c.dZds,RZplot['dZds'])
#color_plot = plot_difference(RZplot['dRds'],c.dRds, c.R, c.Z,'dRds')
#color_plot = plot_difference(RZplot['dZds'],c.dZds, c.R, c.Z,'dZds')


#test jacobian
print 'jacobian'
RZplot['jacobian'] = num.loadtxt(RZ_dir+'jacobian.txt',delimiter=',')
diff_real(c.jacobian,RZplot['jacobian'])
#color_plot = plot_difference(RZplot['jacobian'],c.jacobian, c.R, c.Z,'jacobian')

#test BM1,2,3
print 'BM1,2,3'
RZplot['BM1'] = num.loadtxt(RZ_dir+'BM1.real',delimiter=',') + num.loadtxt(RZ_dir+'BM1.imag',delimiter=',')*1j
RZplot['BM2'] = num.loadtxt(RZ_dir+'BM2.real',delimiter=',') + num.loadtxt(RZ_dir+'BM2.imag',delimiter=',')*1j
RZplot['BM3'] = num.loadtxt(RZ_dir+'BM3.real',delimiter=',') + num.loadtxt(RZ_dir+'BM3.imag',delimiter=',')*1j
diff_comp(c.BM1,RZplot['BM1'])
diff_comp(c.BM2,RZplot['BM2'])
diff_comp(c.BM3,RZplot['BM3'])
#color_plot = plot_difference(num.real(RZplot['BM1']),num.real(c.BM1), c.R, c.Z,'real(BM1)')
#color_plot = plot_difference(num.real(RZplot['BM2']),num.real(c.BM2), c.R, c.Z,'real(BM2)')
#color_plot = plot_difference(num.real(RZplot['BM3']),num.real(c.BM3), c.R, c.Z,'real(BM3)')

#test B1,2,3
print 'B1,2,3'
RZplot['B1'] = num.loadtxt(RZ_dir+'B1.real',delimiter=',') + num.loadtxt(RZ_dir+'B1.imag',delimiter=',')*1j
RZplot['B2'] = num.loadtxt(RZ_dir+'B2.real',delimiter=',') + num.loadtxt(RZ_dir+'B2.imag',delimiter=',')*1j
RZplot['B3'] = num.loadtxt(RZ_dir+'B3.real',delimiter=',') + num.loadtxt(RZ_dir+'B3.imag',delimiter=',')*1j
RZplot['Bn'] = num.loadtxt(RZ_dir+'Bn.real',delimiter=',') + num.loadtxt(RZ_dir+'Bn.imag',delimiter=',')*1j
diff_comp(c.B1,RZplot['B1'])
diff_comp(c.B2,RZplot['B2'])
diff_comp(c.B3,RZplot['B3'])
diff_comp(c.Bn,RZplot['Bn'])

#color_plot = plot_difference(num.real(RZplot['B1']),num.real(c.B1), c.R, c.Z,'real(B1)')
#color_plot = plot_difference(num.real(RZplot['B2']),num.real(c.B2), c.R, c.Z,'real(B2)')
#color_plot = plot_difference(num.real(RZplot['B3']),num.real(c.B3), c.R, c.Z,'real(B3)')
#color_plot = plot_difference(num.real(RZplot['Bn']),num.real(c.Bn), c.R, c.Z,'real(Bn)')

#color_plot = plot_difference(num.imag(RZplot['B1']),num.imag(c.B1), c.R, c.Z,'imag(B1)')
#color_plot = plot_difference(num.imag(RZplot['B2']),num.imag(c.B2), c.R, c.Z,'imag(B2)')
#color_plot = plot_difference(num.imag(RZplot['B3']),num.imag(c.B3), c.R, c.Z,'imag(B3)')
#color_plot = plot_difference(num.imag(RZplot['Bn']),num.imag(c.Bn), c.R, c.Z,'imag(Bn)')

#test Br,Bz,Bphi
print 'Br,Bz,Bphi'
RZplot['Br'] = num.loadtxt(RZ_dir+'Br.real',delimiter=',') + num.loadtxt(RZ_dir+'Br.imag',delimiter=',')*1j
RZplot['Bz'] = num.loadtxt(RZ_dir+'Bz.real',delimiter=',') + num.loadtxt(RZ_dir+'Bz.imag',delimiter=',')*1j
RZplot['Bphi'] = num.loadtxt(RZ_dir+'Bphi.real',delimiter=',') + num.loadtxt(RZ_dir+'Bphi.imag',delimiter=',')*1j
diff_comp(c.Br,RZplot['Br'])
diff_comp(c.Bz,RZplot['Bz'])
diff_comp(c.Bphi,RZplot['Bphi'])
#color_plot = plot_difference(num.real(RZplot['Br']),num.real(c.Br), c.R, c.Z,'real(Br)')
#color_plot = plot_difference(num.real(RZplot['Bz']),num.real(c.Bz), c.R, c.Z,'real(Bz)')
#color_plot = plot_difference(num.real(RZplot['Bphi']),num.real(c.Bphi), c.R, c.Z,'real(Bphi)')

#color_plot = plot_difference(num.imag(RZplot['Br']),num.imag(c.Br), c.R, c.Z,'imag(Br)')
#color_plot = plot_difference(num.imag(RZplot['Bz']),num.imag(c.Bz), c.R, c.Z,'imag(Bz)')
#color_plot = plot_difference(num.imag(RZplot['Bphi']),num.imag(c.Bphi), c.R, c.Z,'imag(Bphi)')


c.get_VPLASMA()
print 'VM1,2,3'
RZplot['VM1'] = num.loadtxt(RZ_dir+'VM1.real',delimiter=',') + num.loadtxt(RZ_dir+'VM1.imag',delimiter=',')*1j
RZplot['VM2'] = num.loadtxt(RZ_dir+'VM2.real',delimiter=',') + num.loadtxt(RZ_dir+'VM2.imag',delimiter=',')*1j
RZplot['VM3'] = num.loadtxt(RZ_dir+'VM3.real',delimiter=',') + num.loadtxt(RZ_dir+'VM3.imag',delimiter=',')*1j
diff_comp(c.VM1,RZplot['VM1'])
diff_comp(c.VM2,RZplot['VM2'])
diff_comp(c.VM3,RZplot['VM3'])
fig = pt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.plot(c.VM1.flatten())
ax1.plot(RZplot['VM1'].flatten())
ax2.plot(c.VM2.flatten())
ax2.plot(RZplot['VM2'].flatten())
ax3.plot(c.VM3.flatten())
ax3.plot(RZplot['VM3'].flatten())
fig.canvas.draw()
fig.show()

print 'V1,2,3'
RZplot['V1'] = num.loadtxt(RZ_dir+'V1.real',delimiter=',') + num.loadtxt(RZ_dir+'V1.imag',delimiter=',')*1j
RZplot['V2'] = num.loadtxt(RZ_dir+'V2.real',delimiter=',') + num.loadtxt(RZ_dir+'V2.imag',delimiter=',')*1j
RZplot['V3'] = num.loadtxt(RZ_dir+'V3.real',delimiter=',') + num.loadtxt(RZ_dir+'V3.imag',delimiter=',')*1j
diff_comp(c.V1,RZplot['V1'])
diff_comp(c.V2,RZplot['V2'])
diff_comp(c.V3,RZplot['V3'])
#color_plot = plot_difference(num.real(RZplot['V1']),num.real(c.V1), c.R[0:181,:], c.Z[0:181,:],'real(V1)')
#color_plot = plot_difference(num.real(RZplot['V2']),num.real(c.V2), c.R[0:181,:], c.Z[0:181,:],'real(V2)')
#color_plot = plot_difference(num.real(RZplot['V3']),num.real(c.V3), c.R[0:181,:], c.Z[0:181,:],'real(V3)')
#color_plot = plot_difference(num.real(RZplot['Vn']),num.real(c.Vn), c.R[0:181,:], c.Z[0:181,:],'real(Vn)')

#color_plot = plot_difference(num.imag(RZplot['V1']),num.imag(c.V1), c.R[0:181,:], c.Z[0:181,:],'imag(V1)')
#color_plot = plot_difference(num.imag(RZplot['V2']),num.imag(c.V2), c.R[0:181,:], c.Z[0:181,:],'imag(V2)')
#color_plot = plot_difference(num.imag(RZplot['V3']),num.imag(c.V3), c.R[0:181,:], c.Z[0:181,:],'imag(V3)')
#color_plot = plot_difference(num.imag(RZplot['Vn']),num.imag(c.Vn), c.R[0:181,:], c.Z[0:181,:],'imag(Vn)')

print 'Vr,Vz,Vphi'
RZplot['Vr'] = num.loadtxt(RZ_dir+'Vr.real',delimiter=',') + num.loadtxt(RZ_dir+'Vr.imag',delimiter=',')*1j
RZplot['Vz'] = num.loadtxt(RZ_dir+'Vz.real',delimiter=',') + num.loadtxt(RZ_dir+'Vz.imag',delimiter=',')*1j
RZplot['Vphi'] = num.loadtxt(RZ_dir+'Vphi.real',delimiter=',') + num.loadtxt(RZ_dir+'Vphi.imag',delimiter=',')*1j
diff_comp(c.Vr,RZplot['Vr'])
diff_comp(c.Vz,RZplot['Vz'])
diff_comp(c.Vphi,RZplot['Vphi'])


print 'DPSIDS, T'
RZplot['DPSIDS'] = num.loadtxt(RZ_dir+'DPSIDS.txt',delimiter=',')
diff_real(c.DPSIDS,RZplot['DPSIDS'])
RZplot['T'] = num.loadtxt(RZ_dir+'T.txt',delimiter=',')
diff_real(c.T,RZplot['T'])

#color_plot = plot_difference(num.real(RZplot['Vr']),num.real(c.Vr), c.R[0:181,:], c.Z[0:181,:],'real(Vr)')
#color_plot = plot_difference(num.real(RZplot['Vz']),num.real(c.Vz), c.R[0:181,:], c.Z[0:181,:],'real(Vz)')
#color_plot = plot_difference(num.real(RZplot['Vphi']),num.real(c.Vphi), c.R[0:181,:], c.Z[0:181,:],'real(Vphi)')

#color_plot = plot_difference(num.imag(RZplot['Vr']),num.imag(c.Vr), c.R[0:181,:], c.Z[0:181,:],'imag(Vr)')
#color_plot = plot_difference(num.imag(RZplot['Vz']),num.imag(c.Vz), c.R[0:181,:], c.Z[0:181,:],'imag(Vz)')
#color_plot = plot_difference(num.imag(RZplot['Vphi']),num.imag(c.Vphi), c.R[0:181,:], c.Z[0:181,:],'imag(Vphi)')

def normal_plot(plot_quantity,R,Z,title_text):
    fig = pt.figure()
    ax = fig.add_subplot(111)
    color_plot = ax.pcolor(R,Z,plot_quantity)
    pt.colorbar(color_plot, ax=ax)
    color_plot.set_clim([0,0.3])
    ax.set_title('Percent Difference '+title_text)
    fig.canvas.draw()
    fig.show()
    return color_plot


color_plot = normal_plot(num.abs(c.Vn), c.R[0:181,:], c.Z[0:181,:],'abs(Vn)')
color_plot = normal_plot(num.abs(c.Vr), c.R[0:181,:], c.Z[0:181,:],'abs(Vr)')
