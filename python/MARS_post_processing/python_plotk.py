import matplotlib.pyplot as pt
import numpy as np
file_handle = file('bnisldo','r')
file_lines = file_handle.readlines()
output_dict = {}
output_dict['REAL']={}
output_dict['IMAG']={}
for i in range(3,len(file_lines)):
    current_line = file_lines[i]
    #print current_line
    if current_line=='\n':
        print i, 'blank'
    elif current_line.find('REAL')!=(-1):
        component = 'REAL'
        mode = int(current_line.split('=')[1])
        output_dict[component][mode] = []
        print component, mode
    elif current_line.find('IMAG')!=(-1):
        component = 'IMAG'
        mode = int(current_line.split('=')[1])
        output_dict[component][mode] = []
        print component, mode
    else:
        line_list = filter(None,current_line.split(' '))
        for j in line_list:
            output_dict[component][mode].append(float(j))
m = np.array(output_dict['REAL'].keys())
m.sort()
start = 1
mode_amps = np.ones((m.shape[0], len(output_dict['REAL'][m[0]])),dtype='complex')
for i in range(0,m.shape[0]):
    mode_amps[i,:] = (np.array(output_dict['REAL'][m[i]])+1j*(np.array(output_dict['IMAG'][m[i]])))
fig, ax = pt.subplots()
ax.imshow(np.abs(mode_amps))
mode_amps *= 2.*np.pi

fig.canvas.draw(); fig.show()
s = np.loadtxt('PROFEQ.OUT')[:,0]
print s.shape
m *= -1
fig, ax = pt.subplots()
color_plot = ax.pcolor(m, np.sqrt(s[1:-1]),np.abs(mode_amps).transpose(),cmap='hot')
pt.colorbar(color_plot, ax = ax)
fig.canvas.draw(); fig.show()

import h5py
fig_tmp, ax_tmp = pt.subplots(nrows = 2)
#file_name = surfmn_file#'spectral_info.h5'
tmp_file = h5py.File('spectral_info.h5')
stored_data = tmp_file.get('1')
zdat = stored_data[0][0]; xdat = stored_data[0][1]; ydat = stored_data[0][2]
image1 = ax_tmp[0].pcolor(xdat,ydat,zdat,cmap = 'hot')
color_plot = ax_tmp[1].pcolor(m, (s[1:-1]),np.abs(mode_amps).transpose(),cmap='hot')
ax_tmp[0].set_title('Matt routine')
ax_tmp[1].set_title('SH routine')
pt.colorbar(color_plot, ax = ax_tmp[1])
pt.colorbar(image1, ax = ax_tmp[0])
fig_tmp.canvas.draw(); fig_tmp.show()
