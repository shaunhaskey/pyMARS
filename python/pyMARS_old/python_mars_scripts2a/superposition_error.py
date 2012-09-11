import numpy as num
import scipy.interpolate as interp
import matplotlib.pyplot as pt
import matplotlib.mlab as mlab
from scipy.interpolate import *
import os
from RZfuncs import *

directory_upper = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult0.540/exp0.670/marsrun/RUNrfa_COILupper.p/'
directory_lower = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult0.540/exp0.670/marsrun/RUNrfa_COILlower.p/'
phasing = -240
directory_single = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult0.540/exp0.670/marsrun/RUNrfa_FEEDI-240.p/'

R_comb, Z_comb, B1_comb, B2_comb, B3_comb, Bn_comb, BMn_comb = combine_data(directory_upper, directory_lower, phasing)

R_sing, Z_sing, B1_sing, B2_sing, B3_sing, Bn_sing, BMn_sing = extract_data_temp(directory_single)

B1_error = num.abs(B1_sing[1:-2,:]-B1_comb[1:-2,:])/num.abs(B1_sing[1:-2,:])*100.
B2_error = num.abs(B2_sing[1:-2,:]-B2_comb[1:-2,:])/num.abs(B2_sing[1:-2,:])*100.
B3_error = num.abs(B3_sing[1:-2,:]-B3_comb[1:-2,:])/num.abs(B3_sing[1:-2,:])*100.
Bn_error = num.abs(Bn_sing[1:-2,:]-Bn_comb[1:-2,:])/num.abs(Bn_sing[1:-2,:])*100.
BMn_error = num.abs(BMn_sing[1:-2,:]-BMn_comb[1:-2,:])/num.abs(BMn_sing[1:-2,:])*100.

print 'B1 error percent - mean: %.4f, std_dev: %.4f, min: %.4f, max: %.4f'%(num.mean(B1_error),num.std(B1_error),num.min(B1_error),num.max(B1_error))
print 'B2 error percent - mean: %.4f, std_dev: %.4f, min: %.4f, max: %.4f'%(num.mean(B2_error),num.std(B2_error),num.min(B2_error),num.max(B2_error))
print 'B3 error percent - mean: %.4f, std_dev: %.4f, min: %.4f, max: %.4f'%(num.mean(B3_error),num.std(B3_error),num.min(B3_error),num.max(B3_error))
print 'Bn error percent - mean: %.4f, std_dev: %.4f, min: %.4f, max: %.4f'%(num.mean(Bn_error),num.std(Bn_error),num.min(Bn_error),num.max(Bn_error))
print 'BMn error percent - mean: %.4f, std_dev: %.4f, min: %.4f, max: %.4f'%(num.mean(BMn_error),num.std(BMn_error),num.min(BMn_error),num.max(BMn_error))


plot_error(R_comb,Z_comb,B1_error)
