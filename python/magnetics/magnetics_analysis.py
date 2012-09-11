import numpy as num
import scipy
import data
import matplotlib.pyplot as pt
from scipy.interpolate import interp1d

def get_alpha(array, shot, time, number_of_samples, n_list):
    F, fft_list, output_list = array_details(array['pickup_names'], shot, time, number_of_samples)
    alpha = calculate_alpha(array['phi'], n_list, F)
    print alpha
    print num.angle(alpha[2],deg=True), num.abs(alpha[2])
    return alpha


def hilbert_plots(shot, time, I_coil_name, pickup_name, number_of_samples, freq_band):
    I_coil_output = interpolate_signal(I_coil_name, shot, time)[0:number_of_samples]
    I_coil_time = time[0:number_of_samples]
    hilb_phase, hilb_amp = hilbert_trans2(I_coil_output, I_coil_time/1000., freq_band)
    plot_stuff(I_coil_time, I_coil_output, hilb_phase, hilb_amp)
    
    pickup_output = interpolate_signal(pickup_name, shot, time)[0:number_of_samples]
    pickup_time = time[0:number_of_samples]
    hilb_phase_pick, hilb_amp_pick = hilbert_trans2(pickup_output, pickup_time/1000., freq_band)
    plot_stuff(pickup_time, pickup_output, hilb_phase_pick, hilb_amp_pick)

    phase_diff = hilb_phase_pick-hilb_phase
    for i in range(0,len(phase_diff)):
        if phase_diff[i] < -180:
            phase_diff[i] += 360
        if phase_diff[i] > 180:
            phase_diff[i] -= 360
    plot_stuff(pickup_time, pickup_output, phase_diff, hilb_amp/hilb_amp_pick)

def plot_stuff(time, output, phase, amp):
    fig = pt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax.plot(time, phase)
    ax2.plot(time,output)
    ax3.plot(time,amp)
    ax3.set_ylim(0,num.max(amp)*1.1)
    fig.canvas.draw()
    fig.show()


def calculate_alpha(phi, n_list, F):
    gamma = num.zeros((len(phi), len(n_list)),dtype=complex)
    for i in range(0, len(n_list)):
        for j in range(0, len(phi)):
            gamma[j,i] = num.exp(1j*n_list[i]*phi[j]/180.*num.pi)
    gamma_inv = num.linalg.pinv(gamma)
    alpha = num.dot(gamma_inv, F)
    return alpha

def hilbert_trans2(signal, time, applied_frequency):
    sample_period = (time[-1]-time[0])/len(time)
    sample_rate = 1./sample_period
    #print sample_period, sample_rate/1.e6,'Mhz', len(time)
    
    freq = num.fft.fftfreq(len(signal),d=sample_period)
    freq1 = num.argmin(num.abs(freq-applied_frequency[0]*0.3))
    freq2 = num.argmin(num.abs(freq-applied_frequency[1]*2))
    
    fft_window=num.hanning(freq2-freq1)
    
    mask = num.zeros(len(signal),dtype=complex)
    mask[freq1:freq2] = fft_window*0.+1.
    
    signal_fft = num.fft.fft(signal)
    signal_ifft = num.fft.ifft(signal_fft*mask)
    phase = (num.arctan2(signal_ifft.imag, signal_ifft.real))*180./num.pi
    amp = num.abs(signal_ifft)*2
    for i in range(0,len(phase)):
        if phase[i] < -180:
            phase[i] += 360
        if phase[i] > 180:
            phase[i] -= 360
    return phase, amp

def interpolate_signal(pickup_name, shot, time):
    print pickup_name
    pickup_data = data.Data(pickup_name, shot)
    pickup_data = pickup_data.xslice((0, time[0]*0.9, num.min([time[-1]*1.1, pickup_data.x[0][-1]])))
    f = interp1d(pickup_data.x[0], pickup_data.y)
    output = f(time)
    return output

def array_details(pickup_name, shot, time, number_of_samples):
    fig = pt.figure()
    ax = fig.add_subplot(111)
    F = num.zeros((len(pickup_name), 1),dtype=complex)
    output_list = []
    fft_list = []
    for tmp in range(0, len(pickup_name)):
        output_list.append(interpolate_signal(pickup_name[tmp], shot, time)[0:number_of_samples])
        #print time.shape, output_list[-1].shape, 1./(time[1]-time[0]),'kHz'
        fft_list.append(num.fft.fft(output_list[-1]))
        #print num.max(num.abs(fft_list[-1])), num.abs(fft_list[-1])
        ax.plot(freq_list, num.abs(fft_list[-1]), 'o', label = pickup_name[tmp])
        F[tmp,0]=fft_list[-1][freq_position]
        ax.legend()
        fig.canvas.draw()
        fig.show()
    return F, fft_list, output_list

MPI66M_array = {}
MPI66M_array['pickup_names'] = ['MPI66M067','MPI66M097','MPI66M127','MPI66M157','MPI66M247','MPI66M277','MPI66M307','MPI66M322','MPI66M340']
MPI66M_array['phi'] = [67.5,97.4,127.9,157.6,246.4,277.5,307.,317.4,339.7]

UISL_array = {}
UISL_array['pickup_names'] = ['UISL1','UISL2','UISL3','UISL4','UISL5','UISL6']
UISL_array['phi'] = [17.8, 73., 133., 198.6, 253., 312.3]

LISL_array = {}
LISL_array['pickup_names'] = ['LISL1','LISL2','LISL3','LISL4','LISL5','LISL6']
LISL_array['phi'] = [18.7, 73., 133., 198.6, 253., 313.2]

MISL_array = {}
MISL_array['pickup_names'] = ['MISL1','MISL2','MISL3','MISL4','MISL5','MISL6']
MISL_array['phi'] = [17.2, 72.4, 132.6, 197.6, 252.4, 312.3]

IU_array = {}
IU_array['pickup_names'] = ['IU30','IU90','IU150','IU210','IU270','IU330']
IU_array['phi'] = [30.,90.,150.,210.,270.,330.]

IL_array = {}
IL_array['pickup_names'] = ['IL30','IL90','IL150','IL210','IL270','IL330']
IL_array['phi'] = [30.,90.,150.,210.,270.,330.]

n_list = [0, 1,2,3]

shot = 146382; start_time = 2500; end_time = 6500;
number_of_samples = 1500; desired_freq = 10; sample_rate = 1000

freq_list = num.fft.fftfreq(number_of_samples, 1./sample_rate)
freq_position = num.argmin(num.abs(freq_list-desired_freq))
time = num.arange(start_time,end_time, (1./sample_rate)*1000, dtype=float)

UISL_array['alpha'] = get_alpha(UISL_array, shot, time, number_of_samples, n_list)
LISL_array['alpha'] = get_alpha(LISL_array, shot, time, number_of_samples, n_list)
MISL_array['alpha'] = get_alpha(MISL_array, shot, time, number_of_samples, n_list)
IU_array['alpha'] = get_alpha(IU_array, shot, time, number_of_samples, n_list)
IL_array['alpha'] = get_alpha(IL_array, shot, time, number_of_samples, n_list)
MPI66M_array['alpha'] = get_alpha(MPI66M_array, shot, time, number_of_samples, n_list)

number_of_samples=1000
hilbert_plots(shot, time, 'IU30', 'UISL1', number_of_samples, [5.,20.])
