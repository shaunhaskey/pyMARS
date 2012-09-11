import results_class
import PythonMARS_funcs as funcs
dir1 = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult1.980/exp0.430/marsrun/RUNrfa_COILupper.p/'
new_data = results_class.data(dir1,Nchi=240)

r_array, z_array = funcs.post_mars_r_z(dir1)
Br = funcs.extractB(dir1,'Br')
Bz = funcs.extractB(dir1,'Bz')
Bphi = funcs.extractB(dir1,'Bphi')
