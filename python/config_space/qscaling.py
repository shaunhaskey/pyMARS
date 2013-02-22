'''
This file plots the q profile scaling that happens in corsica. Why Does Matt do it this way?
Is there a better way to do it without modifying the core q values so much?k

'''
import numpy as np
import matplotlib.pyplot as pt
import pickle
based_on_single = 0
if based_on_single:

    psibar = np.linspace(0,1,100)
    qedge = 5.0
    qmin = 1.15
    c = qmin
    b = 0
    a = qedge-qmin
    qprofile = a * psibar**2 + b * psibar + c

    dataq = np.loadtxt(open('PROFEQ.OUT','r'))
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

plot_styles = ['-','--','-.']
plot_values = range(3,6,1)
for j,i in enumerate(plot_values):
    tmp = np.argmin(np.abs(np.array(q_95_values)-i))
    cur_serial = serials[tmp]
    dataq = np.loadtxt(q_scan_dict['sims'][cur_serial]['dir_dict']['mars_upper_plasma_dir']+'/PROFEQ.OUT')

    psibar = dataq[:,0]
    qprofile = dataq[:,1]
    pprofile = dataq[:,3]
    ax[0].plot(dataq[:,0], dataq[:,1], plot_styles[j],label=r'$q_{95}=%.1f$'%(q_scan_dict['sims'][cur_serial]['Q95']))
    print q_scan_dict['sims'][cur_serial]['Q95'],q_scan_dict['sims'][cur_serial]['BETAN'],q_scan_dict['sims'][cur_serial]['LI'],q_scan_dict['sims'][cur_serial]['BETAN']/q_scan_dict['sims'][cur_serial]['LI']
ax[0].set_ylabel('q', fontsize=12)
ax[0].legend(loc='best')


beta_scan_fname = '/u/haskeysr/mars/equal_spacingV2/equal_spacingV2_post_processing_PEST.pickle'
beta_scan_file = file(beta_scan_fname,'r')
beta_scan_dict = pickle.load(beta_scan_file)
beta_scan_file.close()

q_95_values = []
betaN_values = []
li_values = []
qmult_values = []; pmult_values=[]
serials = beta_scan_dict['sims'].keys()
for i in serials:
    q_95_values.append(beta_scan_dict['sims'][i]['Q95'])
    betaN_values.append(beta_scan_dict['sims'][i]['BETAN'])
    li_values.append(beta_scan_dict['sims'][i]['LI'])
    qmult_values.append(beta_scan_dict['sims'][i]['QMULT'])
    pmult_values.append(beta_scan_dict['sims'][i]['PMULT'])
bn_li = np.array(betaN_values)/np.array(li_values)
plot_styles = ['-','--','-.']
plot_values = np.array([2.,3.,4.])
q_value = 3.
for i in range(0,len(plot_values)):
    tmp = np.argmin((np.array(q_95_values)-q_value)**2+(bn_li-plot_values[i])**2)
    cur_serial = serials[tmp]
    dataq = np.loadtxt(beta_scan_dict['sims'][cur_serial]['dir_dict']['mars_upper_plasma_dir']+'/PROFEQ.OUT')
    ax[1].plot(dataq[:,0], dataq[:,3], plot_styles[i],label=r'$\beta_{N}/L_i=%.1f$'%(beta_scan_dict['sims'][cur_serial]['BETAN']/beta_scan_dict['sims'][cur_serial]['LI']))
    print beta_scan_dict['sims'][cur_serial]['BETAN']/beta_scan_dict['sims'][cur_serial]['LI'], beta_scan_dict['sims'][cur_serial]['Q95']
ax[1].set_ylabel('Pressure (a.u.)', fontsize = 12)
ax[1].legend(loc='best')
ax[1].set_xlabel(r'$\sqrt{\Psi_N}$',fontsize=14)

fig.canvas.draw()
fig.show()
