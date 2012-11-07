'''
This bit of code will read in the reflectometer data - either from the MDSPlus database
on DIII-D computers, or from a .pickle file, or hdf5 file. It will then interpolate the
data onto two regular grids - one of density vs time and one of radius vs time. This allows
The Fourier decomposition of the signal as a function of time, to find the 10Hz component
in the signal. Also calculates signal to noise, and can display a MARS output along with the
reflectometer output. 

SH : Sept 7 2012
'''


try:
    import data
except ImportError:
    print 'cant import data - currently not on DIII-D computer system'
import time, pickle
import matplotlib.pyplot as pt
import numpy as np
import scipy.interpolate as interp
import h5py

I_coil_freq = 10.
#shot = 146398; start_time = 3200; end_time = 3620
#start_time = 3010; end_time = 3980
#start_time = 3010; end_time = 3970

shot = 146392; start_time = 2990; end_time = 4700
shot = 146397; start_time = 3030; end_time = 4800
#shot = 146398; start_time = 3200; end_time = 3620

#shot = 146398; start_time = 3050; end_time = 3950

#shot = 146398; start_time = 3350; end_time = 3650 #around 3500
#shot = 146398; start_time = 3650; end_time = 3950 #around 3800
#shot = 146398; start_time = 3150; end_time = 3450 #around 3800


#shot = 146400; start_time = 2800; end_time = 4300
#shot = 138340; start_time = 2800; end_time = 4300
#shot = 146392; start_time = 3010; end_time = 3950
#shot = 146392; start_time = 3010; end_time = 3950
#shot = 146392; start_time = 3010; end_time = 3650
#shot = 146397; start_time = 3030; end_time = 3950

reg_grid_pts = 100
rad_thresh = 208
include_MARS = 1
perform_interp = 1
density_interp_range = [0.1e19, 6.e19]
radius_interp_range = [2.1, 2.35]
skip_am = 1
#period = 5.

load_pickle = 0
load_hdf5 = 1
save_pickle = 0
include_n = 0
only_save = 0

#for MARS run
#for GA comps
if shot == 146397:
    base_dir = '/home/srh112/mars/shot146397_3515/qmult1.000/exp1.000/marsrun/'
    base_dir = '/home/srh112/mars/shot146397_3815/qmult1.000/exp1.000/marsrun/'
    base_dir = '/home/srh112/mars/shot146397_3305/qmult1.000/exp1.000/marsrun/'
elif shot == 146398:
    print 'using new non kinetic shot 146398 MARS run'
    base_dir = '/home/srh112/code/pyMARS/other_scripts/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/'
    base_dir = '/u/haskeysr/mars/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/'
    base_dir = '/home/srh112/mars/shot146398_3515/qmult1.000/exp1.000/marsrun/'
else:
    base_dir = '/home/srh112/code/pyMARS/other_scripts/shot146398_ul_june2012/qmult1.000/exp1.000/marsrun/'
#for sh_laptop

#plots
plot_profiles = 1
grid_plot = 1
plot_interp_example = 0
single_clr_plot = 1

def monoton(input_dens, input_r):
    for i in range(0,len(input_r)-1):
        if (input_r[i+1]-input_r[i])>0 or (input_r[i+1]-input_r[i])>0 :
            j = 3
            while ((input_r[j+i]-input_r[i])>0) and (j<len(input_r)):
                j+=3
            shift = 0
            change_start_loc = np.max([i -shift, 0])
            change_end_loc = np.min([j+i+shift+1, len(input_r)])
            old_input_r = input_r[change_start_loc:change_end_loc]
            #print input_r[i], input_r[j+i], len(input_r[i+1:j+i])
            input_r[change_start_loc:change_end_loc] = np.linspace(input_r[change_start_loc],input_r[change_end_loc],len(old_input_r))
            #input_dens[i+1:j+i]=np.interp(input_r[i+1:j+i], old_input_r, input_dens[i+1:j+i])
    if np.min(input_dens[1:]-input_dens[0:-1])<0:
        print '---->ERROR input_dens not monotonic - what to do?'
    if np.max(input_r[1:]-input_r[0:-1])>0:
        print '---->ERROR input_r not monotonic - what to do?'
    return input_dens, input_r

#go from p(r) to r(p) by fitting a radial profile function at each time step
def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def check_monotonic(input_dens, input_r):
    if np.min(input_dens[1:]-input_dens[0:-1])<0:
        print 'E',
        return True
    if np.max(input_r[1:]-input_r[0:-1])>0:
        print 'E',
        input_dens, input_r = monoton(input_dens, input_r)
        return True

#make it a multiple of the I-coil frequency
print 'old end_time :', end_time
end_time = start_time + (((end_time - start_time)/int(1./10.*1000.))*int(1./10.*1000.))
print 'new end_time :', end_time

if load_pickle == 1:
    tmp_filename = file('displacement_data%s.pickle'%(shot),'r')
    stored_data = pickle.load(tmp_filename)
    tmp_filename.close()
    n_time, n_x1, n_data = stored_data['n']
    I_coil_x, I_coil_y = stored_data['I_coil']
    boundary_x, boundary_y = stored_data['boundary']
elif load_hdf5 ==1:
    tmp_filename = '/home/srh112/NAMP_datafiles/hdf5testfile2.h5'
    tmp_file = h5py.File(tmp_filename,'r')
    stored_data = tmp_file.get(str(shot))
    n_data = stored_data[0][0]
    n_time = stored_data[0][1]
    n_r = stored_data[0][2]
    n_rho = stored_data[0][3]
    tmp_file.close()
    tmp_filename = file('/home/srh112/NAMP_datafiles/displacement_data%s.pickle'%(shot),'r') #tmp hack to make it work
    stored_data = pickle.load(tmp_filename)
    tmp_filename.close()
    I_coil_x, I_coil_y = stored_data['I_coil']
    boundary_x, boundary_y = stored_data['boundary']
else:
    channel = 'reflect_3dr'
    n = data.Data([channel,'d3d'],shot,save_xext=1)
    n_time = n.xext[0]
    n_r = n.xext[1]
    n_data = n.y
    I_coil = data.Data('IU30',shot)
    I_coil_x = I_coil.x[0]
    I_coil_y = I_coil.y
    #get boundary location data
    try:
        boundary = data.Data(['RMIDOUT','efit03'],shot)
        boundary_x = boundary.x[0]; boundary_y = boundary.y
        print 'Got RMIDOUT from efit03'
    except:
        try:
            boundary = data.Data(['RMIDOUT','efit02'],shot)
            boundary_x = boundary.x[0];boundary_y = boundary.y
            print 'Got RMIDOUT from efit02'
        except:
            try:
                boundary = data.Data(['RMIDOUT','efit01'],shot)
                boundary_x = boundary.x[0];boundary_y = boundary.y
                print 'Got RMIDOUT from efit01'
            except:
                print 'Failed to get RMIDOUT'
                boundary = None; boundary_x = None; boundary_y = None
    del n, I_coil, boundary

if save_pickle ==1:
    if include_n ==1:
        save_data  = {'n':(n_time, n_x1, n_data), 'boundary':(boundary_x, boundary_y), 'I_coil':(I_coil_x, I_coil_y)}
    else:
        save_data  = {'boundary':(boundary_x, boundary_y), 'I_coil':(I_coil_x, I_coil_y)}
    tmp_filename = file('displacement_data%s.pickle'%(shot), 'w')
    pickle.dump(save_data, tmp_filename)
    tmp_filename.close()
    print 'data pickled'
    if only_save ==1:
        exit('finished')

def time_slice(x_data, y_data, time_bounds):
    tmp_min = np.argmin(np.abs((x_data - time_bounds[0])))
    tmp_max = np.argmin(np.abs((x_data - time_bounds[1])))
    return x_data[tmp_min:tmp_max], y_data[tmp_min:tmp_max]

#matrix positions for times of interest - note this throws away a sample of each side
start_time = np.max([start_time, n_time[1]])
end_time = np.min([end_time, n_time[-2]])
start_loc = np.argmin(np.abs(n_time-start_time))
end_loc = np.argmin(np.abs(n_time-end_time))

if boundary_x != None:
    boundary_x, boundary_y = time_slice(boundary_x, boundary_y, [start_time, end_time])
    print boundary_x[0], boundary_x[-1], start_time, end_time
    #I_coil_x, I_coil_y = time_slice(I_coil_x, I_coil_y, [start_time, end_time])
    #print I_coil_x[0], I_coil_x[-1], start_time, end_time

#time domain plots
fig_time, ax_time = pt.subplots(nrows=3, sharex=1)
fig2 = pt.figure()

#frequency domain plots - constant density lines
fig_freq = pt.figure()
ax_freq = []
ax_freq.append(fig_freq.add_subplot(311))
ax_freq.append(fig_freq.add_subplot(312, sharex=ax_freq[0]))
ax_freq.append(fig_freq.add_subplot(325))
ax_freq.append(fig_freq.add_subplot(326, sharex=ax_freq[2], sharey = ax_freq[2]))

#frequency domain plots 2 - constant radius lines
fig_freq2 = pt.figure()
ax_freq2 = []
ax_freq2.append(fig_freq2.add_subplot(311))
ax_freq2.append(fig_freq2.add_subplot(312, sharex=ax_freq2[0]))
ax_freq2.append(fig_freq2.add_subplot(325))
ax_freq2.append(fig_freq2.add_subplot(326, sharex=ax_freq2[2], sharey = ax_freq2[2]))

#create a regular density, time and radius grid to interpolate onto
plot_densities = np.linspace(density_interp_range[0], density_interp_range[1],reg_grid_pts)
plot_radius = np.linspace(radius_interp_range[0],radius_interp_range[1],reg_grid_pts)
period = np.min(n_time[1:]-n_time[:-1])
time_base = np.arange(n_time[start_loc+1],n_time[end_loc-2], period)

#matrices to contain the interpolated data
const_dens = np.zeros((len(plot_densities),end_loc-start_loc),dtype=float)
const_rad = np.zeros((len(plot_radius),end_loc-start_loc),dtype=float)


if plot_profiles == 1:
    tmp_fig, tmp_ax = pt.subplots(nrows = 2, sharex=1, sharey = 1)

non_monoton = 0; non_monoton_fixed = 0

#perform the interpolation onto a regular density and radius grid
for i in range(start_loc,end_loc):
    input_dens = np.flipud(n_data[:,i]) #np.flipud(n_data[:,i])
    input_r = np.flipud(n_r[:,i]) #np.flipud(n_r[:,i])

    #narrow the data down to what is needed
    a1 = np.argmin(np.abs(input_dens-np.min(plot_densities)))
    a2 = np.argmin(np.abs(input_r-np.min(plot_radius)))
    b1 = np.argmin(np.abs(input_dens-np.max(plot_densities)))
    b2 = np.argmin(np.abs(input_r-np.max(plot_radius)))
    max_loc = np.min([np.max([a2, b1])+2, len(input_dens)])
    min_loc = np.max([np.min([b2, a1])-2, 0])
    input_dens = input_dens[min_loc:max_loc:skip_am]
    input_r = input_r[min_loc:max_loc:skip_am]

    if plot_profiles == 1:
        tmp_ax[0].plot(input_r, input_dens)
    #smooth(input_r, window_len=10,window='hanning')

    if check_monotonic(input_dens, input_r):
        input_dens, input_r = monoton(input_dens, input_r)
        non_monoton +=  1
    smooth(input_r, window_len=5,window='hanning')

    if check_monotonic(input_dens, input_r):
        print 'failed again....'
        input_dens, input_r = monoton(input_dens, input_r)
        non_monoton_fixed += 1

    if perform_interp == 1:
        #create constant density vs time lines
        #f = interp.interp1d(input_dens, input_r)
        #const_dens[:,i-start_loc]=f(plot_densities)
        #if num.min(input_dens[1:]-input_dens[:-1])<0:
        if np.all(np.diff(input_dens) > 0):
            const_dens[:,i-start_loc] = np.interp(plot_densities, input_dens, input_r)
        else:
            raise Exception('xp not monotonic')

        #create constant radius vs time lines
        #f = interp.interp1d(np.flipud(input_r),np.flipud(input_dens))
        #if np.max(input_r[1:]-input_r[:-1])>0:
        if np.all(np.diff(np.flipud(input_r)) >= 0):
        #const_rad[:,i-start_loc]=f(plot_radius)
            const_rad[:,i-start_loc] = np.interp(plot_radius, np.flipud(input_r), np.flipud(input_dens))
        else:
            raise Exception('xp input r not monotonic')
    if plot_profiles == 1:
        tmp_ax[1].plot(input_r, input_dens)

if plot_interp_example==1:
    fig, ax  = pt.subplots()
    i = int((start_loc+end_loc)/2)
    ax.plot(n_r[:,i],n_data[:,i],'x', label='data points')
    ax.plot(plot_radius, const_rad[:,i-start_loc],'x', label='data points')
    ax.legend(loc='best')
    ax.set_xlabel('R(m)')
    ax.set_ylabel('Density')
    ax.set_title('Interpolation check')
    fig.canvas.draw()
    fig.show()

#upsample
#perform the interpolation onto a regular time axis
print 'Interpolating onto a regular time axis'
const_dens_reg_time = np.zeros((const_rad.shape[0],len(time_base)),dtype=float)
const_rad_reg_time = const_dens_reg_time * 1.
for i in range(0,const_rad.shape[0]):
     const_dens_reg_time[i,:] = np.interp(time_base, n_time[start_loc:end_loc], const_dens[i,:])
     const_rad_reg_time[i,:] = np.interp(time_base, n_time[start_loc:end_loc], const_rad[i,:])
const_dens = const_dens_reg_time
const_rad = const_rad_reg_time

#Interpolate the I-coil signal and find its amp @ I_coil_freqHz
fft_freqs = np.fft.fftfreq(len(time_base),(time_base[1]-time_base[0])/1000.)
loc_10Hz = np.argmin(np.abs(fft_freqs[0:len(fft_freqs)/2] - I_coil_freq))

if I_coil_x != None:
    f = interp.interp1d(I_coil_x, I_coil_y)
    I_coil_interp = f(time_base)
    fft_values = np.fft.fft(I_coil_interp)/len(time_base)/1000 #->kA
    I_coil_current = (2.*np.abs(fft_values[loc_10Hz]))
if boundary_x != None:
    pass
    #f = interp.interp1d(boundary_x, boundary_y)
    #boundary_interp = f(time_base)
print 'Finished interpolating onto a regular time axis'

#downsample - ?? decimate or resample??
#const_dens, time_base = sig.resample(const_dens, 200, t=n_time[start_loc:end_loc, axis=1)
#const_rad, time_base = sig.resample(const_dens, 200, t=n_time[start_loc:end_loc, axis=1)

#print non_monoton, end_loc-start_loc
#print non_monoton_fixed, end_loc-start_loc

if plot_profiles == 1:
    tmp_ax[1].set_xlabel('R (m)')
    tmp_ax[0].set_ylabel('density')
    tmp_ax[1].set_ylabel('density')
    tmp_ax[0].set_title('shot:%d, %d - %d ms'%(shot, start_time, end_time))
    tmp_ax[1].set_title('monoton shot:%d, %d - %d ms'%(shot, start_time, end_time))
    tmp_fig.canvas.draw(); tmp_fig.show()

#plot the gridded data and raw data as pcolor plots
if grid_plot == 1:
    lim1 = [0,6.4e19]
    clr_fig, clr_ax = pt.subplots(nrows=3, sharex=1)
    clr_ax[0].set_title('grid_data and raw data')
    clr_plot = clr_ax[0].pcolor(time_base[::10], plot_densities, const_dens[:,::10], cmap = 'spectral')
    pt.colorbar(clr_plot, ax=clr_ax[0])
    #clr_plot.set_clim(lim1)
    clr_plot = clr_ax[1].pcolor(time_base[::10], plot_radius, const_rad[:,::10], cmap = 'jet')
    pt.colorbar(clr_plot, ax=clr_ax[1])
    clr_ax[1].set_xlim([start_time,end_time])
    clr_ax[1].set_ylim([2.2,2.35])
    clr_plot.set_clim([0,3.2e19])
    clr_plot = clr_ax[2].pcolor(n_time[start_loc:end_loc:10], n_r[:,start_loc:end_loc:10], n_data[:,start_loc:end_loc:10], cmap = 'spectral')
    pt.colorbar(clr_plot, ax=clr_ax[2])
    clr_fig.canvas.draw(); clr_fig.show()
    #clr_plot.set_clim(lim1)

#create the single plot of the gridded data
if single_clr_plot == 1:
    tmp_fig, tmp_ax = pt.subplots(nrows = 2, sharex = 1)
    clr_plot1 = tmp_ax[0].pcolor(time_base[::10], plot_radius, const_rad[:,::10], cmap = 'jet')
    clr_plot2 = tmp_ax[1].pcolor(time_base[::10], plot_densities, const_dens[:,::10], cmap = 'jet')

    tmp_ax[0].set_xlim([start_time,end_time])
    tmp_ax[0].set_ylim([2.2,2.35])
    clr_plot1.set_clim([0,3.2e19])
    clr_plot2.set_clim([2.2,2.35])
    if boundary_x != None:
        tmp_ax[0].plot(boundary_x, boundary_y, 'k')
    tmp_ax[0].set_xlabel('time ms')
    tmp_ax[0].set_ylabel('radius m')
    tmp_ax[0].set_title('Shot %d'%(shot,))
    cbar = pt.colorbar(clr_plot1, ax=tmp_ax[0])
    cbar = pt.colorbar(clr_plot2, ax=tmp_ax[1])
    tmp_fig.canvas.draw(); tmp_fig.show()
    #clr_ax[2].plot(I_coil_x, I_coil_y,'k')


#plot the constant radius contours in density vs time space
colormap = pt.cm.jet_r
fig_new, ax_new = pt.subplots(nrows=2)
ax_new[0].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, int((n_data.shape[0])/3)+1)])
ax_new[1].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, int((n_data.shape[0])/3)+1)])
for i in range(0,n_data.shape[0], 3):
    ax_new[0].plot(n_time[start_loc:end_loc], n_data[i,start_loc:end_loc],'-')
for i in range(0,const_rad.shape[0], 3):
    ax_new[1].plot(time_base, const_rad[i,:])
ax_new[0].set_title('original data - assumes regular grid, not usually true!')
ax_new[1].set_title('constant radius data grid')
fig_new.canvas.draw()
fig_new.show()



#how often to skip lines on the line plots to avoid overcrowding
plot_skip = 1; plot_skip_2 = 3

num_plots = int(const_dens.shape[0]/plot_skip)+1
num_plots2 = int(const_dens.shape[0]/plot_skip_2)+1

#setup the cycle color maps on the line plots with many lines
colormap = pt.cm.jet_r
colormap2 = pt.cm.jet
ax_time[0].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, int((n_data.shape[0])/3)+1)])
ax_time[2].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots2)])

ax_freq[2].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
ax_freq2[2].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
ax_freq[3].set_color_cycle([colormap2(i) for i in np.linspace(0, 0.9, num_plots/3)])
ax_freq2[3].set_color_cycle([colormap2(i) for i in np.linspace(0, 0.9, num_plots/3)])

#lists to hold the various answers in
#dens_list = []; amp_list_n = []; SNR_list_n = []; phase_list_n = []
#radius_list = [];amp_list = [];phase_list = [];SNR_list = []

#perform fft on the data
print 'performing fft analysis'
def perform_fft_calc(const_dens, const_rad, I_coil_interp, I_coil_freq, time_base, start_pos, end_pos):
    period = time_base[1]-time_base[0]
    fft_length = end_pos - start_pos
    answer_dict = {}
    const_dens_fft = np.fft.fft(const_dens[:,start_pos:end_pos],axis=1)/fft_length*100 #cm
    const_rad_fft = np.fft.fft(const_rad[:,start_pos:end_pos],axis=1)/fft_length#*100
    I_coil_fft = np.fft.fft(I_coil_interp[start_pos:end_pos])/fft_length/1000. #->kA
    freq_list = np.fft.fftfreq(const_rad_fft.shape[1],period/1000.)
    loc_10Hz = np.argmin(np.abs(freq_list - I_coil_freq))
    I_coil_current = (2.*np.abs(I_coil_fft[loc_10Hz]))

    #extract useful data
    answer_dict['radius_list'] = np.abs(const_dens_fft[:,0]).flatten()
    answer_dict['dens_list'] = np.abs(const_rad_fft[:,0]).flatten()

    answer_dict['amp_list'] = np.abs(const_dens_fft[:,loc_10Hz]).flatten()*2./I_coil_current
    answer_dict['amp_list_n'] = np.abs(const_rad_fft[:,loc_10Hz]).flatten()*2./I_coil_current

    answer_dict['phase_list'] = np.angle(const_dens_fft[:,loc_10Hz]/I_coil_fft[loc_10Hz],deg=True).flatten()
    answer_dict['phase_list_n'] = np.angle(const_rad_fft[:,loc_10Hz],deg=True).flatten()

    answer_dict['signal_power'] = (2.*np.abs(const_dens_fft[:,loc_10Hz]))**2
    answer_dict['noise_power'] = (np.sum(np.abs(const_dens_fft[:,1:])**2,axis=1))
    answer_dict['signal_power_n'] = (2.*np.abs(const_rad_fft[:,loc_10Hz]))**2
    answer_dict['noise_power_n'] = (np.sum(np.abs(const_rad_fft[:,1:])**2,axis=1))

    answer_dict['SNR_list'] = answer_dict['signal_power']/answer_dict['noise_power']
    answer_dict['SNR_list_n'] = answer_dict['signal_power_n']/answer_dict['noise_power_n']
    answer_dict['const_dens_fft'] = const_dens_fft/I_coil_fft[loc_10Hz]
    answer_dict['const_rad_fft'] = const_rad_fft/I_coil_fft[loc_10Hz]
    return answer_dict


def step_through(cycles, overlap, I_coil_freq, time_base):
    tmp_fig, tmp_ax = pt.subplots(nrows = 4, sharex = 1)
    tmp_time = 1./I_coil_freq * cycles * 1000.
    increment = np.argmin(np.abs(time_base - (time_base[0]+tmp_time)))
    start_pos = 0
    end_pos = start_pos + increment
    start_pos_increment = np.max([int(increment*overlap),1])
    plot_list_rad = []; plot_list_amp = []; plot_list_SNR = []; label_list = []
    plot_list_power = []; plot_list_phase = []
    while end_pos < len(time_base):
        print start_pos, end_pos, time_base[start_pos], time_base[end_pos]
        answer_dict = perform_fft_calc(const_dens, const_rad, I_coil_interp, I_coil_freq, time_base, start_pos, end_pos)
        plot_list_rad.append(answer_dict['radius_list'])
        plot_list_amp.append(answer_dict['amp_list'])
        plot_list_SNR.append(answer_dict['SNR_list'])
        plot_list_power.append(answer_dict['signal_power'])
        plot_list_phase.append(answer_dict['phase_list'])
        label_list.append('%d-%d'%(time_base[start_pos], time_base[end_pos]))
        #tmp_ax[0].plot(answer_dict['radius_list'], answer_dict['amp_list'],'.-', label='%d-%d'%(time_base[start_pos], time_base[end_pos]))
        #tmp_ax[1].plot(answer_dict['radius_list'], answer_dict['SNR_list'],'.-', label='%d-%d'%(time_base[start_pos], time_base[end_pos]))
        start_pos += start_pos_increment
        end_pos = start_pos + increment
    colormap2 = pt.cm.jet
    for j in tmp_ax:
        j.set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(plot_list_rad))])
    for i, j in enumerate(plot_list_rad):
        tmp_ax[0].plot(j, plot_list_amp[i],'-',label = label_list[i])
        tmp_ax[1].plot(j, plot_list_phase[i],'.',label = label_list[i])
        tmp_ax[2].plot(j, plot_list_SNR[i],'-',label = label_list[i])
        tmp_ax[3].plot(j, plot_list_power[i],'-',label = label_list[i])
    #tmp_ax[0].legend(loc='best')    
    tmp_ax[0].legend(loc='best')
    tmp_fig.canvas.draw(); tmp_fig.show()
step_through(5, 0.3, 10, time_base)    

#if cycles == None:
#    end_pos = const_dens.shape[1]
#else:
#    tmp_time = 1./I_coil_freq * cycles
#    end_pos = np.max(np.argmin(np.abs(time_base - (time_base[start_pos]+tmp_time))),len(time_base))

answer_dict = perform_fft_calc(const_dens, const_rad, I_coil_interp, I_coil_freq, time_base, 0, len(time_base))

radius_list = answer_dict['radius_list']
dens_list = answer_dict['dens_list']

amp_list = answer_dict['amp_list']
amp_list_n = answer_dict['amp_list_n']

phase_list = answer_dict['phase_list']
phase_list_n = answer_dict['phase_list_n']

signal_power = answer_dict['signal_power']
noise_power = answer_dict['noise_power']
signal_power_n = answer_dict['signal_power_n']
noise_power_n = answer_dict['noise_power_n']

SNR_list = answer_dict['SNR_list']
SNR_list_n = answer_dict['SNR_list_n']

const_dens_fft = answer_dict['const_dens_fft']
const_rad_fft = answer_dict['const_rad_fft']


rad_thresh_loc = np.argmin(np.abs(radius_list-rad_thresh))
print 'plotting quantities'

ax_freq[2].plot(np.transpose(fft_freqs[0:len(fft_freqs)/2]),  np.transpose(np.abs(const_dens_fft[:rad_thresh_loc,0:len(fft_freqs)/2])),'-')
ax_freq[3].plot(np.transpose(fft_freqs[0:len(fft_freqs)/2]),  np.transpose(np.abs(const_dens_fft[rad_thresh_loc:,0:len(fft_freqs)/2])),'-')
ax_freq2[2].plot(np.transpose(fft_freqs[0:len(fft_freqs)/2]),  np.transpose(np.abs(const_rad_fft[:,0:len(fft_freqs)/2])),'-')
#ax_freq2[3].plot(np.transpose(fft_freqs[0:len(fft_freqs)/2]),  np.transpose(np.abs(const_dens_fft[rad_thresh_loc:,0:len(fft_freqs)/2])),'-') #,label='%.1e'%(plot_densities[i]))

ax_time[2].plot(np.transpose(time_base), np.transpose(const_dens[::plot_skip_2,:]),'-',label='%.1e'%(plot_densities[i]))
ax_time[0].plot(np.transpose(time_base), np.transpose(const_rad[::plot_skip_2,:]))


fig_freq.canvas.draw();fig_freq.show()
fig_freq2.canvas.draw();fig_freq.show()
'''

if i%plot_skip == 0 and radius_list[-1]>rad_thresh:
    ax_freq[2].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values[0:len(fft_freqs)/2]),'-',label='%.1e'%(plot_densities[i]))
elif i%plot_skip == 0:
    ax_freq[3].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values[0:len(fft_freqs)/2]),'-',label='%.1e'%(plot_densities[i]))

if i%plot_skip == 0 and plot_radius[i]*100.>rad_thresh:
    ax_freq2[2].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values_n[0:len(fft_freqs)/2]),'-')
elif i%plot_skip == 0:
    ax_freq2[3].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values_n[0:len(fft_freqs)/2]),'-')


for i in range(0,const_dens.shape[0]):
    #plot lines of constant density in radius vs time space
    if i%plot_skip_2 == 0:
        ax_time[2].plot(n_time[start_loc:end_loc], const_dens[i,:],'-',label='%.1e'%(plot_densities[i]))
        ax_time[0].plot(n_time[start_loc:end_loc], const_rad[i,:])

    #interpolate constant density signal with new time base
    #and fft the resulting signal
    f = interp.interp1d(n_time[start_loc:end_loc], const_dens[i,:])
    fft_values = np.fft.fft(f(time_base))/len(time_base)*100./I_coil_current #->cm/kA

    #interpolate constant radius signal with new time base
    #and fft the resulting signal
    f = interp.interp1d(n_time[start_loc:end_loc], const_rad[i,:])
    fft_values_n = np.fft.fft(f(time_base))/len(time_base)*100./I_coil_current #->cm/kA

    #Record the mean radius of each constant density signal
    radius_list.append(np.abs(fft_values[0])*I_coil_current) #because this offset isn't dependent on I-coil current
    dens_list.append(np.abs(fft_values_n[0])*I_coil_current) #because this offset isn't dependent on I-coil current

    #Record the amplitude @ I-coil frequency
    amp_list.append(np.abs(fft_values[loc_10Hz])*2.)
    amp_list_n.append(np.abs(fft_values_n[loc_10Hz])*2.)

    #Record the phase @ I-coil frequency, need to take I-coil into account
    phase_list.append(np.angle(fft_values[loc_10Hz], deg=True))
    phase_list_n.append(np.angle(fft_values_n[loc_10Hz], deg=True))

    #Record signal to noise of the I-coil frequency signal excluding the mean value
    #SNR_list.append((amp_list[i])**2/(np.sum(np.abs(fft_values)**2)-(amp_list[i])**2-(radius_list[i]/I_coil_current)**2))
    #SNR_list_n.append((amp_list_n[i])**2/(np.sum(np.abs(fft_values_n)**2)-(amp_list_n[i])**2-(dens_list[i]/I_coil_current)**2))

    SNR_list.append((amp_list[i]/2.)**2/(np.sum(np.abs(fft_values[1:len(fft_values)-1])**2)-(amp_list[i]/2.)**2))
    SNR_list_n.append((amp_list_n[i]/2.)**2/(np.sum(np.abs(fft_values_n[1:len(fft_values_n)-1])**2)-(amp_list_n[i]/2.)**2))
    #SNR_list_n.append((amp_list_n[i])**2/(np.sum(np.abs(fft_values_n)**2)-(amp_list_n[i])**2-(dens_list[i]/I_coil_current)**2))

    #plot fft of each constant density signal
    if i%plot_skip == 0 and radius_list[-1]>rad_thresh:
        ax_freq[2].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values[0:len(fft_freqs)/2]),'-',label='%.1e'%(plot_densities[i]))
    elif i%plot_skip == 0:
        ax_freq[3].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values[0:len(fft_freqs)/2]),'-',label='%.1e'%(plot_densities[i]))

    if i%plot_skip == 0 and plot_radius[i]*100.>rad_thresh:
        ax_freq2[2].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values_n[0:len(fft_freqs)/2]),'-')
    elif i%plot_skip == 0:
        ax_freq2[3].plot(fft_freqs[0:len(fft_freqs)/2], 2*np.abs(fft_values_n[0:len(fft_freqs)/2]),'-')

'''
ax_freq[2].text(10,0.4,'SNR good, R > %dcm'%(rad_thresh))
ax_freq[3].text(10,0.4,'SNR bad, R < %dcm'%(rad_thresh))
if boundary_x != None:
    ax_time[2].plot(boundary_x,boundary_y,'k-')
ax_time[2].set_ylim([2.1,2.35])
#plot I-coil freq pickup amp as a function of radius and SNR of that signal
ax_freq[0].plot(radius_list, amp_list, 'x-',label='10Hz reflect')


ax_freq[0].set_title('shot:%d time:%d->%d, dens %.1e -> %.1e'%(shot, start_time, end_time,min(plot_densities),max(plot_densities)))

boundary_list = [np.max(boundary_y)*100,np.min(boundary_y)*100,np.mean(boundary_y)*100]
v_line_list = boundary_list
v_line_list.append(rad_thresh)

ax_freq2[0].plot(plot_radius*100, amp_list_n, 'x-',label='10Hz reflect')
ax6_2_n=ax_freq2[0].twinx()



ax6_2=ax_freq[0].twinx()
ax7_2=ax_freq[1].twinx()

fig_delta, ax_delta = pt.subplots()
ax_delta.plot(plot_radius*100, np.array(amp_list_n)/np.array(dens_list),'x-')
ax_delta.set_xlabel('R (cm)')
ax_delta.set_ylabel('10Hz delta n / n') 
ax_delta.grid()
ax_freq2[0].set_title('shot:%d time:%d->%d, rad %.1e -> %.1e'%(shot, start_time, end_time,min(plot_radius),max(plot_radius)))
ax_delta.set_title('shot:%d time:%d->%d, 10Hz delta n/n vs radius'%(shot, start_time, end_time))
ax_delta.text(229,0.015, 'boundary')
ax_delta.set_ylim([0,0.065])
ax_delta2=ax_delta.twinx()

#plot the reflectometer density vs radius on some plots
for i in [start_loc+10, int((start_loc+end_loc)/2), end_loc-10]:
    input_dens = np.flipud(n_data[:,i])
    input_r = np.flipud(n_r[:,i])
    ax6_2.plot(np.array(input_r)*100, input_dens, label='density profile')
    ax6_2_n.plot(np.array(input_r)*100, input_dens, label='density profile')
    ax7_2.plot(np.array(input_r)*100, input_dens, label='density profile')
    ax_delta2.plot(np.array(input_r)*100, input_dens, label='density profile')

fig_delta.canvas.draw(); fig_delta.show()

#plot to show the comparison between the two ways of calculating the
fig_tmp10, ax_tmp10 = pt.subplots(nrows = 4, sharex = 1)
#for i in [start_loc+10, int((start_loc+end_loc)/2), end_loc-10]:
for i in [int((start_loc+end_loc)/2)]:
    input_dens = np.flipud(n_data[:,i])
    input_r = np.flipud(n_r[:,i])
    dr_drho = np.diff(input_r)/np.diff(input_dens)
    new_r = (input_r[1:]+input_r[:-1])/2.
    dr_drho_interp = np.interp(plot_radius,new_r[::-1],dr_drho[::-1])
    ax_tmp10[0].plot(plot_radius, amp_list_n)
    ax_tmp10[1].plot(input_r, input_dens)
    ax_tmp10[2].plot(new_r,dr_drho)
    ax_tmp10[2].plot(plot_radius, dr_drho_interp,'x-')
    ax_tmp10[3].plot(plot_radius,np.abs(dr_drho_interp*amp_list_n)*100,label=str(i))

ax_tmp10[0].set_ylabel(r'$d\rho$')
ax_tmp10[1].set_ylabel(r'$\rho$')
ax_tmp10[2].set_ylabel(r'$dr/d\rho$')
ax_tmp10[3].set_ylabel(r'$dr$')
ax_tmp10[3].plot(radius_list/100., amp_list, 'x-', label='dr_1')
ax_tmp10[3].legend(loc='best')
fig_tmp10.canvas.draw();fig_tmp10.show()

#Overlay the 
fig_tmp11, ax_tmp11 = pt.subplots()
tmp_xaxis = np.arange(start_time, end_time, 10)/1000.
tmp_data = np.ones((len(amp_list),len(tmp_xaxis)),dtype=float)
for i in range(0,len(amp_list)):
    tmp_data[i,:] = amp_list[i] * np.sin(10.*np.pi*2*tmp_xaxis)
clr_tmp11 = ax_tmp11.pcolor(tmp_xaxis,plot_radius,tmp_data, rasterized=True)
pt.colorbar(clr_tmp11,ax=ax_tmp11)
fig_tmp11.canvas.draw();fig_tmp11.show()


if single_clr_plot == 1:
    #tmp_fig, tmp_ax = pt.subplots(nrows = 2, sharex = 1)
    tmp_fig, tmp_ax = pt.subplots()
    tmp_ax = [tmp_ax]
    clr_plot1 = tmp_ax[0].pcolor(time_base[::10], plot_radius, const_rad[:,::10], cmap = 'jet')
    clr_plot1.set_rasterized(True)
    #clr_plot2 = tmp_ax[1].pcolor(time_base[::10], plot_densities, const_dens[:,::10], cmap = 'jet')
    #clr_plot2.set_rasterized(True)

    tmp_ax[0].set_xlim([start_time,end_time])
    tmp_ax[0].set_ylim([2.1,2.35])
    clr_plot1.set_clim([0,5e19])
    #clr_plot2.set_clim([2.2,2.35])
    if boundary_x != None:
        tmp_ax[0].plot(boundary_x, boundary_y, 'k--', linewidth=5)
    tmp_ax[0].set_xlabel('Time (ms)', fontsize = 15)
    tmp_ax[0].set_ylabel('Radius (m)', fontsize = 15)
    tmp_ax[0].set_title('Shot %d'%(shot,), fontsize = 15)
    
    new_vals_tmp = np.arange(210.,233.,1.)/100.
    new_vals_amp = np.interp(new_vals_tmp, plot_radius,amp_list/100.)
    new_vals_phase = np.interp(new_vals_tmp, plot_radius,phase_list)

    for i in range(0,len(new_vals_amp)):
        tmp_data = new_vals_amp[i] * np.cos(10.*np.pi*2*time_base[::10]/1000.+phase_list[i]/180.*np.pi)*4. + new_vals_tmp[i]
        tmp_ax[0].plot(time_base[::10],tmp_data,'k-')
    cbar = pt.colorbar(clr_plot1, ax=tmp_ax[0])
    cbar.set_label(r'Density   $(m^{-3})$', fontsize = 15)
    #cbar = pt.colorbar(clr_plot2, ax=tmp_ax[1])
    tmp_fig.canvas.draw(); tmp_fig.show()
    tmp_fig.savefig('/home/srh112/Desktop/testing.pdf')
    #clr_ax[2].plot(I_coil_x, I_coil_y,'k')



ax6_2.set_ylabel('density profile')
ax7_2.set_ylabel('density profiles')
ax_freq[1].plot(radius_list, SNR_list, 'o-')

for i in range(0,len(ax_freq)):
    ax_freq[i].grid()
    ax_freq2[i].grid()

ax_freq2[1].plot(plot_radius*100, SNR_list_n, 'o-')

#slice I-coil signal for plot
ax_time[0].set_ylabel('Density')
ax_time[0].set_title('x shot:%d time:%d->%d interp data'%(shot,start_time, end_time))
ax_time[1].set_ylabel('IU30')
ax_time[1].plot(I_coil_x,I_coil_y)
ax_time[2].set_ylabel('R (m)')
ax_time[2].set_xlabel('Time (ms)')
ax_time[0].set_xlim([start_time, end_time])
ax_freq[1].set_xlabel('R (cm)')
ax_freq[1].set_ylabel('%.1fHz SNR'%(I_coil_freq))
ax_freq[0].set_ylabel('disp mag @ %.1fHz (cm/kA)'%(I_coil_freq))
ax_freq[0].set_title('shot:%d time:%d->%d, dens %.1e -> %.1e'%(shot,start_time, end_time,min(plot_densities),max(plot_densities)))
ax_freq[0].set_ylim([0,0.4])

ax_freq[2].set_xlabel('Freq (Hz)')
ax_freq[2].set_ylabel('FFT Amp (cm/kA)')
ax_freq[2].set_ylim([0,0.6]); ax_freq[2].set_xlim([3,30])

ax_freq[3].set_xlabel('Freq (Hz)')
ax_freq[3].set_xlim([3,30]); ax_freq[3].set_ylim([0,0.6])
ax_freq2[3].set_ylim([0,1.e20]); ax_freq2[2].set_xlim([3,30])

v_line_axes = [ax_freq[0],ax_freq[1],ax_freq2[0],ax_freq2[1],ax_delta]
for tmp_ax in v_line_axes:
    tmp_ax.vlines(boundary_list,tmp_ax.get_ylim()[0],tmp_ax.get_ylim()[1])

fig_time.canvas.draw(); fig_time.show()
fig_freq.canvas.draw(); fig_freq.show()
fig_freq2.canvas.draw(); fig_freq2.show()


if include_MARS == 1:
    from  results_class import *
    from RZfuncs import I0EXP_calc
    import numpy as np
    import matplotlib.pyplot as pt
    import time
    import PythonMARS_funcs as pyMARS

    def extract_data(base_dir, I0EXP, ul=0, Nchi=513, get_VPLASMA=0):
        if ul==0:
            c = data(base_dir + 'RUNrfa.p', I0EXP = I0EXP, Nchi=Nchi)
            d = data(base_dir + 'RUNrfa.vac', I0EXP = I0EXP, Nchi=Nchi)
            if get_VPLASMA:
                c.get_VPLASMA()
                d.get_VPLASMA()
            return (c,d)
        else:
            a = data(base_dir + 'RUN_rfa_lower.p', I0EXP = I0EXP, Nchi=Nchi)
            b = data(base_dir + 'RUN_rfa_lower.vac', I0EXP = I0EXP, Nchi=Nchi)
            c = data(base_dir + 'RUN_rfa_upper.p', I0EXP = I0EXP, Nchi=Nchi)
            d = data(base_dir + 'RUN_rfa_upper.vac', I0EXP = I0EXP, Nchi=Nchi)
            if get_VPLASMA:
                a.get_VPLASMA()
                b.get_VPLASMA()
                c.get_VPLASMA()
                d.get_VPLASMA()
            return (a,b,c,d)


    def combine_fields(input_data, attr_name, theta = 0, field_type='plas'):
        print 'combining property : ', attr_name
        if len(input_data)==2:
            if field_type=='plas':
                answer = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
            elif field_type == 'total':
                answer = getattr(input_data[0], attr_name)
            elif field_type == 'vac':
                answer = getattr(input_data[1], attr_name)
            print 'already combined, ', field_type
        elif len(input_data) == 4:
            if field_type=='plas':
                lower_data = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
                upper_data = getattr(input_data[2], attr_name) - getattr(input_data[3], attr_name)
            elif field_type=='total':
                lower_data = getattr(input_data[0], attr_name)
                upper_data = getattr(input_data[2], attr_name)
            elif field_type=='vac':
                lower_data = getattr(input_data[1], attr_name)
                upper_data = getattr(input_data[3], attr_name)
            answer = upper_data + lower_data*(np.cos(theta)+1j*np.sin(theta))
            print 'combine %s, theta : %.2f'%(field_type, theta)
        return answer

    N = 6; n = 2; I = np.array([1.,-1.,0.,1,-1.,0.])
    I0EXP = I0EXP_calc(N,n,I)
    print I0EXP, 1.0e+3 * 3./np.pi

    ul = 1;
    plot_field = 'Vn'; field_type = 'plas'
    Nchi=513
    run_data = extract_data(base_dir, I0EXP, ul=ul, Nchi=Nchi,get_VPLASMA=1)
    #fig, ax = pt.subplots()

    for theta_deg in [0]:
        print '===== %d ====='%(theta_deg)
        theta = float(theta_deg)/180*np.pi;

        plot_quantity = combine_fields(run_data, plot_field, theta=theta, field_type=field_type)

        grid_r = run_data[0].R*run_data[0].R0EXP
        grid_z = run_data[0].Z*run_data[0].R0EXP

        plas_r = grid_r[0:plot_quantity.shape[0],:]
        plas_z = grid_z[0:plot_quantity.shape[0],:]
        R_values=np.linspace(run_data[0].R0EXP, np.max(plas_r),10000)
        Z_values=R_values * 0
        Vn_values = scipy_griddata((plas_r.flatten(),plas_z.flatten()), plot_quantity.flatten(), (R_values.flatten(),Z_values.flatten()))
        ax_freq[0].plot(R_values*100, np.abs(Vn_values)*100,'k-', label='MARS-Vn')

    plot_field = 'Vr'; field_type = 'plas'
    for theta_deg in [0]:
        print '===== %d ====='%(theta_deg)
        theta = float(theta_deg)/180*np.pi;

        plot_quantity = combine_fields(run_data, plot_field, theta=theta, field_type=field_type)

        grid_r = run_data[0].R*run_data[0].R0EXP
        grid_z = run_data[0].Z*run_data[0].R0EXP

        plas_r = grid_r[0:plot_quantity.shape[0],:]
        plas_z = grid_z[0:plot_quantity.shape[0],:]
        R_values=np.linspace(run_data[0].R0EXP, np.max(plas_r),10000)
        Z_values=R_values * 0

        Vr_values = scipy_griddata((plas_r.flatten(),plas_z.flatten()), plot_quantity.flatten(), (R_values.flatten(),Z_values.flatten()))
        ax_freq[0].plot(R_values*100, np.abs(Vr_values)*100,'r.', label='MARS-Vr')


    ax_freq[0].set_ylim([0,0.4])
    ax_freq[0].legend(loc='lower left')
    fig_freq.canvas.draw()
    fig_freq.show()



fig_tmp, ax_tmp = pt.subplots(nrows = 2, sharex=1)
ax_tmp[0].plot(radius_list, amp_list, 'bx-',label='10Hz reflect')
ax_tmp[1].plot(radius_list, phase_list, 'bx-',label='10Hz reflect')

if include_MARS == 1:
    ax_tmp[0].plot(R_values*100, np.abs(Vn_values)*100,'k-', label='MARS-Vn')
    ax_tmp[1].plot(R_values*100, np.angle(Vn_values,deg=True),'k-', label='MARS-Vn')
fig_tmp.canvas.draw(); fig_tmp.show()


