import numpy as np
import matplotlib.pyplot as pt
import MDSplus as MDS

a = np.loadtxt('/home/srh112/NAMP_datafiles/mars/single_run_through_test_142614_V2/efit/PROFTE', skiprows = 1)
s = a[:,0]
Te = a[:,1]

R_0 = 1.e-7
def calc_resistance(R_0, R0EXP = 1.69, A=2.8):
    alpha = R_0/(Te[0]**(-1.5))
    Resistivity = alpha * Te[1:]**(-1.5)
    #Area pi r^2
    r = R0EXP/A
    A = (s[1:]*r)**2*np.pi - (s[:-1]*r)**2*np.pi
    L = 2*R0EXP*np.pi
    R_zones = Resistivity*L/A
    return 1./(np.sum(1./R_zones))

R_0_vals = np.linspace(1.e-10,1.e-6,1000)
vals = []
R0EXP = 1.69
for i in R_0_vals:
    vals.append(calc_resistance(i, R0EXP = R0EXP))
V_l = 1.
I_p = 1.4e6
resistance = V_l / I_p
fig, ax = pt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)
err = np.abs(np.array(vals) - resistance)
ax.plot(R_0_vals, err,'.')
fig.canvas.draw();fig.show()
resistivity_0 = R_0_vals[np.argmin(err)]

mu0 = 4.*np.pi*10**(-7)
va = 4.55e6
A = 2.8
eta_0 = resistivity_0 /(mu0*va*R0EXP/A**2)
print resistivity_0, eta_0
