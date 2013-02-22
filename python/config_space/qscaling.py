'''
This file plots the q profile scaling that happens in corsica. Why Does Matt do it this way?
Is there a better way to do it without modifying the core q values so much?k

'''
import numpy as num
import matplotlib.pyplot as pt
import pickle
based_on_single = 0
if based_on_single:

    psibar = num.linspace(0,1,100)
    qedge = 5.0
    qmin = 1.15
    c = qmin
    b = 0
    a = qedge-qmin
    qprofile = a * psibar**2 + b * psibar + c

    dataq = num.loadtxt(open('PROFEQ.OUT','r'))
    psibar = dataq[:,0]
    qprofile = dataq[:,1]
    pprofile = dataq[:,3]

    fig = pt.figure()
    ax = fig.add_subplot(111)
    fig, ax = pt.subplots(nrows = 2, sharex = 1)
    #ax[0].plot(psibar,qprofile,label='original')

    qd = 2.
    qminp = 0. 
    qexp1 = 1.8
    #plot_styles = ['k--','b--','y--','c--']
    qmult = [0.6,0.8,1.1,1.5,2]

    for i in range(0,len(qmult)):
        qprofile_new = ((1/qprofile+qprofile)/qd+qminp)*(1+qmult[i]*psibar**qexp1)
        print min(qprofile_new)
        ax[0].plot(psibar,qprofile_new, label='qmult='+str(qmult[i]))
        #qprofile_new = ((1/qprofile+qprofile)/qd+qminp)*(1+qmult)*psibar**qexp1)
    #ax[0].legend(loc=2)
    ax[0].set_ylabel('q')
    ax[0].set_ylim([0,7.5])
    ax[0].grid(b=True)
    #ax[0].set_title('qscaling function ((1/q+q)/qd+qminp)*(1+qmult[i]*psibar**qexp1)')

    pmult = [0.5,0.8,1.1,1.5,2]
    for i in range(0,len(pmult)):
        pprofile_new = pprofile*pmult[i]
        print min(qprofile_new)
        ax[1].plot(psibar,pprofile_new, label='pmult='+str(pmult[i]))
        #qprofile_new = ((1/qprofile+qprofile)/qd+qminp)*(1+qmult)*psibar**qexp1)
    #ax[1].legend(loc='best')
    ax[1].set_xlabel(r'$\sqrt{\psi_N}$', fontsize = 14)
    ax[1].set_ylabel('Pressure')

    ax[1].grid(b=True)

    fig.canvas.draw()
    fig.show()

q_scan_fname = '/u/haskeysr/mars/detailed_q95_scan_n2_lower_BetaN/detailed_q95_scan_n2_lower_BetaN_post_processing_PEST.pickle'
q_scan_file = file(q_scan_fname,'r')
q_scan_dict = pickle.load(q_scan_file)
q_scan_file.close()

q_95_values = []
serials = q_scan_dict['sims'].keys()
for i in serials:
    q_95_values.append(q_scan_dict['sims'][i]['Q95'])

fig = pt.figure()
ax = fig.add_subplot(111)
fig, ax = pt.subplots(nrows = 2, sharex = 1)
plot_values = range(3,6,1)
for i in plot_values:
    tmp = np.argmin(np.abs(np.array(q_95_values)-i))
    profeq_loc = q_scan_dict['sims'][i]['Q95'])
    dataq = num.loadtxt(q_scan_dict['sims'][tmp]['dir_dict']['mars_upper_plasma_dir']+'/PROFEQ.OUT')

    psibar = dataq[:,0]
    qprofile = dataq[:,1]
    pprofile = dataq[:,3]
    ax[0].plot(dataq[:,0], dataq[:,1], label=r'$q_{95}=%.2f$'%(q_scan_dict['sims'][tmp]['Q95']))
    

fig.canvas.draw()
fig.show()
