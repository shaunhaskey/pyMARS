import numpy as np
count = 0
a = (np.random.rand(201,513)-0.5)*1.e-15 + 1j*(np.random.rand(201,513)-0.5)*1.e-15
b = (np.random.rand(513,59)-0.5)*1.e-15 + 1j*(np.random.rand(513,59)-0.5)*1.e-15
for i in range(500):
    c = np.dot(a,b)
    count+=np.sum(np.isnan(c))
print 'number of nans:', count
