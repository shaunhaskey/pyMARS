import numpy as np
import matplotlib.pyplot as pt
import pyMARS.PythonMARS_funcs as py_funcs
profte = np.loadtxt('/home/srh112/NAMP_datafiles/mars/yueqiang_standing_wave_pp/efit/PROFTE', skiprows = 1)
profte = np.loadtxt('/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan/efit/1615/PROFTE', skiprows = 1)
fig, ax = pt.subplots(nrows = 2, sharex = True)
ax[0].plot(profte[:,0], profte[:,1])

#If in keV, convert to eV
if profte[0,1]<200:
    profte[:,1] *= 1000
Te_center = profte[1,1]


# def spitz_eta_func(Te, Z = 1, e = -1.602176565*10**(-19), m = 9.10938291*10**(-31), coul_log = 15, e_0 = 8.85418782*10**(-12), K = 1.38*10**(-23), chen_H_approx = False):
#     '''Te is in eV
#     SRH : 11March2014
#     '''
#     if chen_H_approx:
#         return 5.2*10**(-5) * Z * coul_log/(Te)**1.5
#     else:
#         return np.pi * Z * e**2 * np.sqrt(m) * coul_log/((4.*np.pi*e_0)**2*(K * Te*11600)**1.5)

# def lundquist(eta, L = 1.6, va=4.e6):
#     '''
#     Gives the lundquist number which is a dimensionless quantity
#     The ratio between the Alfven wave crossing timescale to the resistive diffusion timescale

#     eta in [Ohm m ] = [V m A-1]
#     SRH : 11Mar2014
#     '''

#     mu_0 = 4.*np.pi*10**(-7) #[v s A-1 m-1]
#     return mu_0 * L * va / eta

# def eta_from_lundquist(lundquist, L = 1.6):
#     '''
#     Gives the lundquist number which is a dimensionless quantity
#     The ratio between the Alfven wave crossing timescale to the resistive diffusion timescale

#     eta in [Ohm m ] = [V m A-1]
#     SRH : 11Mar2014
#     '''

#     mu_0 = 4.*np.pi*10**(-7) #[v s A-1 m-1]
#     va = 4.e6 #[m s-1]
#     return mu_0 * L * va / lundquist


spitz_eta = py_funcs.spitz_eta_func(Te_center)
lundquist = py_funcs.lundquist_calc(spitz_eta)
eta_return = py_funcs.eta_from_lundquist(lundquist)
print '{:.2e}, {:.2e}'.format(spitz_eta, lundquist)
print '{:.2e}'.format(1./lundquist)
print '{:.2e}'.format(eta_return)

ax[1].plot(profte[:,0], py_funcs.spitz_eta_func(1000. * profte[:,1]/profte[0,1]*2))
ax[1].set_yscale('log')
fig.canvas.draw(); fig.show()
