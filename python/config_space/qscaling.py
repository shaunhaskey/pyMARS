'''
This file plots the q profile scaling that happens in corsica. Why Does Matt do it this way?
Is there a better way to do it without modifying the core q values so much?k

'''
import numpy as num
import matplotlib.pyplot as pt

psibar = num.linspace(0,1,100)
qedge = 5.0
qmin = 1.15
c = qmin
b = 0
a = qedge-qmin
qprofile = a * psibar**2 + b * psibar + c

fig = pt.figure()
ax = fig.add_subplot(111)
ax.plot(psibar,qprofile,label='original')

qd = 2.
qminp = 0. 
qexp1 = 1.8
plot_styles = ['k--','b--','y--','c--']
qmult = [0.5,1,2,3]

for i in range(0,len(qmult)):
    qprofile_new = ((1/qprofile+qprofile)/qd+qminp)*(1+qmult[i]*psibar**qexp1)
    print min(qprofile_new)
    ax.plot(psibar,qprofile_new,plot_styles[i], label='qmult='+str(qmult[i]))
    #qprofile_new = ((1/qprofile+qprofile)/qd+qminp)*(1+qmult)*psibar**qexp1)
ax.legend(loc=2)
ax.set_xlabel('psibar')
ax.set_ylabel('q')
ax.set_title('qscaling function ((1/q+q)/qd+qminp)*(1+qmult[i]*psibar**qexp1)')
fig.canvas.draw()
fig.show()
