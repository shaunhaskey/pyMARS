import numpy as num
import os, time
import PythonMARS_funcs as pyMARS
import results_class as res
import RZfuncs

N = 6; n = 2; I = num.array([1.,-1.,0.,1,-1.,0.])

#print 'phi_location %.2f deg'%(phi_location)
template_dir = '/u/haskeysr/mars/templates/PROBE_G_TEMPLATE/'
base_run_dir = '/u/haskeysr/PROBE_G_RUNS/'
project_name = 'phi_scan/'
run_dir = base_run_dir + project_name
print run_dir
os.system('mkdir '+run_dir)
os.system('cp -r ' + template_dir +'*  ' + run_dir)

print 'go to new directory'
os.chdir(run_dir + 'PROBE_G')

probe_g_template = file('probe_g.in', 'r')
probe_g_template_txt = probe_g_template.read()
probe_g_template.close()

diiid = file('diiid.in', 'r')
diiid_txt = diiid.read()
diiid.close()

#a, b = coil_responses6(1,1,1,1,1,1,Navg=120,default=1)

probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad']
# probe type 1: poloidal field, 2: radial field
probe_type   = num.array([     1,     1,     1,     0,     0,     0,     0, 1,0])
# Poloidal geometry
Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1.])
Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0.])
tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0.])*2*num.pi/360  #DTOR # poloidal inclination
lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05])  # Length of probe

probe_name = 'UISL'
k = probe.index(probe_name)
#Generate interpolation points
Rprobek, Zprobek = pyMARS.pickup_interp_points(Rprobe[k], Zprobe[k], lprobe[k], tprobe[k], probe_type[k], 800)



#Generate the points string and modify the .in file
r_flattened = Rprobek.flatten()
z_flattened = Zprobek.flatten()

phi_flattened = num.linspace(0,360,num=800)
r_flattened = phi_flattened * 0 + num.average(Rprobek.flatten())
z_flattened = phi_flattened * 0 + num.average(Zprobek.flatten())

print 'r',r_flattened
print 'z', z_flattened
print 'phi', phi_flattened

points_string = ''
print len(r_flattened)
for i in range(0,len(r_flattened)):
    points_string+='%.3f   %.3f   %.3f\n'%(r_flattened[i], phi_flattened[i], z_flattened[i])

changes = {'<<npts>>' : str(len(r_flattened)),
          '<<points>>' : points_string}

for tmp_key in changes.keys():
    probe_g_template_txt = probe_g_template_txt.replace(tmp_key, changes[tmp_key])
probe_g_template = file('probe_g.in', 'w')
probe_g_template.write(probe_g_template_txt)
probe_g_template.close()

diiid_changes = {'<<upper>>': '1000 -1000 0 1000 -1000 0',
                '<<lower>>': '1000 -1000 0 1000 -1000 0'}

for tmp_key in diiid_changes:
    diiid_txt = diiid_txt.replace(tmp_key, diiid_changes[tmp_key])

diiid = file('diiid.in', 'w')
diiid.write(diiid_txt)
diiid.close()



#run probe_g
os.system('./probe_g')


#Read the output file
results = num.loadtxt('probe_gb.out', skiprows=8)
B_R = results[:,4]
B_phi =results[:,3]
B_Z= results[:,5]
phi_out = results[:,0]
R_out = results[:,1]
Z_out = results[:,2]
'''
print 'get the answer from MARS'
I0EXP = RZfuncs.I0EXP_calc(N,n,I)

base_dir = '/u/haskeysr/mars/grid_check10/qmult1.000/exp1.000/marsrun/'
Nchi=513

#plas_run = res.data(base_dir + 'RUNrfa.p', I0EXP = I0EXP, Nchi=Nchi)
vac_run = res.data(base_dir + 'RUNrfa.vac', I0EXP = I0EXP, Nchi=Nchi)


grid_r = vac_run.R*vac_run.R0EXP
grid_z = vac_run.Z*vac_run.R0EXP
'''
import matplotlib.pyplot as pt
fig = pt.figure()
ax = fig.add_subplot(111)
ax.plot(phi_flattened, B_R*10000., 'r-')
ax.plot(phi_flattened, B_Z*10000., 'k-')

fig.canvas.draw()
fig.show()
'''

Brprobek, Bzprobek = pyMARS.pickup_field_interpolation(grid_r, grid_z, vac_run.Br, vac_run.Bz, vac_run.Bphi, num.array(Rprobek), num.array(Zprobek))
import matplotlib.pyplot as pt
fig = pt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.plot(Rprobek, num.abs(Brprobek),  'b-')
ax2.plot(Rprobek, num.abs(B_R*10000), 'b--')

ax.plot(Zprobek, num.abs(Bzprobek), 'k-')
ax2.plot(Zprobek, num.abs(B_Z*10000), 'k--')
fig.canvas.draw()
fig.show()


fig = pt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax.plot(Rprobek, R_out,  'b-')
ax2.plot(Zprobek, Z_out, 'b--')

fig.canvas.draw()
fig.show()

'''
