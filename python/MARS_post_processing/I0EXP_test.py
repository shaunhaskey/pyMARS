import numpy as np
import matplotlib.pyplot as pt
I = np.array([1.,-1.,0.,1,-1.,0.])
#I = np.array([1.,-0.5,-0.5,1,-0.5,-0.5])
phi_list = []
current_list = []
current_list_perfect = []
discrete = 1000
for coil_num in range(0,6):
    phi_tmp = np.linspace(np.pi/3*coil_num, (coil_num+1)*np.pi/3,discrete)
    current_tmp = (phi_tmp*0+1) * I[coil_num]
    for i in range(0,len(phi_tmp)-1):
        phi_list.append(phi_tmp[i])
        current_list.append(current_tmp[i])
        current_list_perfect.append(np.cos(2.*phi_tmp[i]))
phi_array = np.array(phi_list)
current_array = np.array(current_list)
current_array_perfect = np.array(current_list_perfect)
current_fft = np.fft.fft(current_array)
current_fft_perfect = np.fft.fft(current_array_perfect)
current_fft_freq = np.fft.fftfreq(len(current_fft),d=(phi_list[1]-phi_list[0])/(np.pi*2))
n2_loc = np.argmin(np.abs(current_fft_freq - 2))
fig, ax = pt.subplots()
ax.plot(phi_list, current_list, 'o')
ax.plot(phi_list, current_list_perfect, 'o')
fig.canvas.draw(); fig.show()

fig, ax = pt.subplots()
ax.plot(current_fft_freq, 2.*np.abs(current_fft)/len(current_fft), 'bo')
ax.plot(current_fft_freq, 2.*np.abs(current_fft_perfect)/len(current_fft), 'ko')
ax.set_xlim([-5,5])
fig.canvas.draw(); fig.show()
#print np.abs(current_fft_perfect[n2_loc])/np.abs(current_fft[n2_loc]), np.abs(current_fft[n2_loc])/np.abs(current_fft_perfect[n2_loc])
print 'dft approach perfect, stepped : ', 2.*current_fft_perfect[n2_loc]/len(current_fft), 2.*current_fft[n2_loc]/len(current_fft)
print 'abs value : ', 2.*np.abs(current_fft_perfect[n2_loc]/len(current_fft)), 2.*np.abs(current_fft[n2_loc]/len(current_fft))

n = 2.
bounds = np.linspace(0, 2.*np.pi, len(I)+1)
complex_sum = 0
for i, I_curr in enumerate(I):
    tmp = 1./(2.*np.pi)*I_curr*1./(-1j*n)*(np.exp(-1j*n*bounds[i+1])-np.exp(-1j*n*bounds[i]))
    #print tmp
    complex_sum += tmp

print 'complex sum', complex_sum, np.abs(complex_sum)
print 3.**1.5/(2.*np.pi), 3./(2.*np.pi)

complex_sum = 0
for i, I_curr in enumerate(I):
    tmp = 1./(2.*np.pi)*I_curr*1./(-1j*n)*(np.exp(-1j*n*bounds[i+1])-np.exp(-1j*n*bounds[i]))
    #print tmp
    complex_sum += tmp
print 'complex sum', complex_sum, np.abs(complex_sum),2.*np.abs(complex_sum)
print 3.**1.5/(2.*np.pi), 3./(2.*np.pi)


a2 = 0
for i, I_curr in enumerate(I):
    tmp = I_curr*1./(np.pi*n)*(np.sin(n*bounds[i+1])-np.sin(n*bounds[i]))
    #print tmp
    a2 += tmp
print '========='
print 'a2:', a2

b2 = 0
for i, I_curr in enumerate(I):
    tmp = I_curr *1./(np.pi*n)*(np.cos(n*bounds[i])-np.cos(n*bounds[i+1]))
    #print tmp
    b2 += tmp
print '========='
print 'b2:', b2

print 'sqrt(a^2 +b^2) : ', np.sqrt(a2**2+b2**2)
print '3^1.5, 3 :', 3.**1.5/(2.*np.pi), 3./(2.*np.pi)
