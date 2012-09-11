from  results_class import *
from RZfuncs import I0EXP_calc
import numpy as num
import matplotlib.pyplot as pt

import PythonMARS_funcs as pyMARS

N = 6
n = 2
I = num.array([1.,-1.,0.,1,-1.,0.])
I0EXP = I0EXP_calc(N,n,I)


c = data('/home/srh112/code/pyMARS/test_shot/marsrun/RUNrfa_COILlower.p', I0EXP = I0EXP)

c.get_VPLASMA()

