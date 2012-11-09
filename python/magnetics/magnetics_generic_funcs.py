#contains several generic functions for calculating couplings
#and phasings between i-coil and pickup arrays


import h5py, copy
import numpy as num
import scipy.signal as signal
import scipy.signal as scipy_signal
try:
    import data
except:
    print 'unable to import data'
from scipy.interpolate import interp1d
from scipy.misc import comb
import matplotlib.pyplot as pt

def normalize(b, a):
    """Normalize polynomial representation of a transfer function.

    If values of b are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.
    """
    #This function has been copied out of scipy with the 'while' part below
    #commented out - hopefully this makes some difference to the badly
    #conditioned filter co-eff problem
    b,a = map(num.atleast_1d,(b,a))
    if len(a.shape) != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if len(b.shape) > 2:
        raise ValueError("Numerator polynomial must be rank-1 or rank-2 array.")
    if len(b.shape) == 1:
        b = num.asarray([b],b.dtype.char)
    while a[0] == 0.0 and len(a) > 1:
        a = a[1:]
    outb = b * (1.0) / a[0]
    outa = a * (1.0) / a[0]
    if num.allclose(outb[:,0], 0, rtol=1e-14, atol=1e-14):
        print "Badly conditioned filter coefficients (numerator): the results may be meaningless"
        #while num.allclose(outb[:,0], 0, rtol=1e-14,atol=1e-14) and (outb.shape[-1] > 1):
        #    outb = outb[:,1:]
    if outb.shape[0] == 1:
        outb = outb[0]
    return outb, outa


def bilinear(b, a, fs=1.0):
    """Return a digital filter from an analog filter using the bilinear transform.

    The bilinear transform substitutes ``(z-1) / (z+1``) for ``s``.
    """
    #This function has been copied out of scipy

    fs =float(fs)
    a,b = map(num.atleast_1d,(a,b))
    D = len(a) - 1
    N = len(b) - 1
    artype = float
    M = max([N,D])
    Np = M
    Dp = M
    bprime = num.zeros(Np+1,artype)
    aprime = num.zeros(Dp+1,artype)
    for j in range(Np+1):
        val = 0.0
        for i in range(N+1):
            for k in range(i+1):
                for l in range(M-i+1):
                    if k+l == j:
                        val += comb(i,k)*comb(M-i,l)*b[N-i]*pow(2*fs,i)*(-1)**k
        bprime[j] = num.real(val)
    for j in range(Dp+1):
        val = 0.0
        for i in range(D+1):
            for k in range(i+1):
                for l in range(M-i+1):
                    if k+l == j:
                        val += comb(i,k)*comb(M-i,l)*a[D-i]*pow(2*fs,i)*(-1)**k
        aprime[j] = num.real(val)
    #return aprime, bprime
    return normalize(bprime, aprime)


def hilbert_trans2(signal, time, applied_frequency):
    sample_period = (time[-1]-time[0])/len(time)
    sample_rate = 1./sample_period
    print 'hilbert :', sample_period, sample_rate/1.e3,'kHz', len(time)
    
    freq = num.fft.fftfreq(len(signal),d=sample_period)
    freq1 = num.argmin(num.abs(freq-applied_frequency[0]))
    freq2 = num.argmin(num.abs(freq-applied_frequency[1]))
    
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

def plot_stuff(time, output, phase, amp):
    fig = pt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax.plot(time, phase)
    ax.set_ylim([0,370])
    ax2.plot(time,output)
    ax3.plot(time,amp)
    ax3.set_ylim(0,num.max(amp)*1.1)
    fig.canvas.draw()
    fig.show()


def interpolate_signal(pickup_name, shot, time, existing_signals, remove_mean = 0, remove_trend = 0):
    '''Interpolate a signal up to the new time axis
    used to change the sampling to something regular
    '''
    #substituting of pickup names
    print pickup_name, shot
    replacement_values = {'iu30':['SPA1BCOM',500.], 'iu210':['SPA1BCOM',500.],
                          'iu90':['SPA2BCOM',500.], 'iu270':['SPA2BCOM',500.],
                          'iu150':['SPA3BCOM',500.], 'iu330':['SPA3BCOM',500.],
                          'il30':['SPA1BCOM',500.], 'il210':['SPA1BCOM',500.],
                          'il90':['SPA2BCOM',500.], 'il270':['SPA2BCOM',500.],
                          'il150':['SPA3BCOM',500.], 'il330':['SPA3BCOM',500.],}
    
    #iu90 and iu150 were the issues originally
    #above substitutions are from subsList2.txt
    if pickup_name.lower() in replacement_values.keys():
        correction = replacement_values[pickup_name.lower()][1]
        pickup_name = replacement_values[pickup_name.lower()][0]
        print '!! coil_name corrected to %s with corr: %.2f'%(pickup_name,correction)
    else:
        correction = 1.

    #Check to see if we already have the data in existing signals
    #if not, then get the data and put in existing signals
    if (shot in existing_signals.keys()) and (pickup_name in existing_signals[shot].keys()):
        pickup_data = existing_signals[shot][pickup_name]
        print 'getting the data from archive'
    else:
        pickup_data = data.Data(pickup_name, shot)
        pickup_data = (pickup_data.x[0], pickup_data.y)
        if shot not in existing_signals.keys():
            existing_signals[shot] = {}
        f = interp1d(pickup_data[0], pickup_data[1])
        time_tmp = num.arange(pickup_data[0][1],pickup_data[0][-2],1./1000.*1000.,dtype=float)
        output_tmp = f(time_tmp)
        existing_signals[shot][pickup_name] = (time_tmp, output_tmp)

    #interpolate the signal to the timebase we want
    f = interp1d(pickup_data[0], pickup_data[1]*correction)
    output = f(time)

    #remove mean and trend if you want
    if remove_mean == 1:
        output = output - num.mean(output)
    if remove_trend == 1:
        f = num.polyfit(time, output, 1)
        output = output - num.polyval(f, time)
    return output

def get_hdf5values(f, sensor_name, coil_name, nd, debug=0):
    g = f.get(sensor_name.lower()).get(coil_name.lower())
    ng = len(g.items())
    if ng % nd == 0:
        vers = num.round(ng/nd)
        print 'Dataset has multiple of %d items - continuing, vers = %d'%(nd,vers)
    else:
        vers = num.round(ng/nd)
        print 'ERROR, Dataset does not have multiple of %d items - continuing, vers = %d'%(nd,vers)
    #number of poles and number of zeros
    nz = g.get('nz.'+str(vers)).value[0]
    np = g.get('np.'+str(vers)).value[0]
    #print 'n_zeros, n_poles : ',nz,np
    if nz > 0:
        sz = g.get('ReZeros.'+str(vers)).value+g.get('ImZeros.'+str(vers)).value*1j
    else:
        sz = []
    sp = g.get('RePoles.'+str(vers)).value+g.get('ImPoles.'+str(vers)).value*1j
    sk = g.get('gain.'+str(vers)).value
    Afit = g.get('a.'+str(vers)).value
    Bfit = g.get('b.'+str(vers)).value
    sk = sk*1e-7
    if debug==1:
        print 'sp:', ['%.2e %.2ei'%(num.real(i_tmp),num.imag(i_tmp)) for i_tmp in sp]
        print 'sz:', ['%.2e %.2ei'%(num.real(i_tmp),num.imag(i_tmp)) for i_tmp in sz]
        print 'sk:', ['%.2e'%(i_tmp) for i_tmp in sk]
    return sk, sp, sz, Afit, Bfit, np, nz

def sfft(ref_signal, signal, time, length, fs=1000., phase_ax=None, amp_ax=None, label=None,i_coil_freq = 10., window='boxcar'):
    complex_list=[]; time_values = []; ref_list = []
    length=int(length)
    window = getattr(scipy_signal, window)
    window_func = window(length)
    for i in range(0,len(signal)-length):
        sfft = 2.*10000.*num.fft.fft(signal[i:i+length]*window_func)/length
        sfft_ref = 2.*num.fft.fft(ref_signal[i:i+length]*window_func)/length
        sfft_freqs = num.fft.fftfreq(len(sfft),1./fs)
        index = num.argmin(num.abs(sfft_freqs-i_coil_freq))
        #print index, sfft_freqs[index], num.max(num.abs(sfft_ref)),num.abs(sfft_ref[index])
        complex_list.append(sfft[index])#/sfft_ref[index])
        time_values.append(num.mean(time[i:i+length]))
        ref_list.append(sfft_ref[index])
    complex_array = num.array(complex_list)
    phase_array = num.angle(complex_array/num.array(ref_list),deg=True)
    amp_array = num.abs(complex_array)
    if phase_ax!= None:
        phase_ax.plot(time_values, phase_array, '-',label=label,linewidth=1)
    if amp_ax!= None:
        amp_ax.plot(time_values, num.abs(amp_array)*1000., '-',label=label, linewidth=1)
    return complex_array, phase_array, amp_array

def hilbert_func(I_coil_output, pickup_output, time, freq_band, ax=None, label=None):
    hilb_phase, hilb_amp = hilbert_trans2(I_coil_output, time/1000., freq_band)
    hilb_phase_pick, hilb_amp_pick = hilbert_trans2(pickup_output, time/1000., freq_band)
    #plot_stuff(time, I_coil_output, hilb_phase, hilb_amp)
    #plot_stuff(time, pickup_output, hilb_phase_pick, hilb_amp_pick)

    phase_diff = hilb_phase_pick-hilb_phase
    for i in range(0,len(phase_diff)):
        while phase_diff[i]< (0) or phase_diff[i]>360:
            if phase_diff[i] < 0:
                phase_diff[i] += 360
            if phase_diff[i] > 360:
                phase_diff[i] -= 360

    if ax!= None:
        ax.plot(time, phase_diff,label=label, linewidth=3.5)
    #plot_stuff(time, pickup_output, phase_diff, hilb_amp_pick/hilb_amp)

def svd_scan(n_range, input_dict, title, include_plot = 0):
    '''Run a scan through a whole range of basis 'harmonics'
    for the svd analysis to see what effect including more functions has
    '''
    n_scan_input = []; n_scan_input_std = []
    n_scan_input_list = []; n_scan_input_resid = []
    for i in range(n_range[0],n_range[1]):
        tmp_n_list = range(0,i)
        tmp, residual = calculate_alpha(input_dict['phi'],tmp_n_list,input_dict['plasma_results_list'])
        n_scan_input.append(tmp[2])
        n_scan_input_resid.append(residual)
        n_scan_input_list.append(tmp)
        n_scan_input_std.append(num.std(num.abs(tmp)))
    if include_plot:
        tmp_fig, tmp_ax = pt.subplots(nrows=2,ncols=2, sharex=1)
        tmp_ax = tmp_ax.flatten()
        colormap = pt.cm.jet_r
        for i in tmp_ax:
            i.grid()
            i.set_color_cycle([colormap(i) for i in num.linspace(0, 0.9, len(n_scan_input_list))])
        for i in range(0,len(n_scan_input_list)):
            tmp_ax[2].plot(num.abs(n_scan_input_list[i]),'o-')
            tmp_ax[1].plot(n_range[0]+i,n_scan_input_std[i],'o')
            tmp_ax[0].plot(n_range[0]+i,num.abs(n_scan_input[i]),'o')
            tmp_ax[3].plot(n_range[0]+i,n_scan_input_resid[i],'o')
        tmp_ax[0].set_title(title)
        tmp_ax[1].set_title('std dev of alpha')
        tmp_ax[2].set_title('alpha')
        tmp_ax[3].set_title('residual')
        tmp_ax[3].set_xlabel('max n')
        tmp_fig.canvas.draw(); tmp_fig.show()

def print_results(sensor_dict, icoil_dict, shot, start_time, end_time):
    '''Print out the results
    '''
    for loc, i in enumerate(['total','vac','plasma']):
        answer = sensor_dict[i+'_alpha']*1.e4/(icoil_dict['total_alpha']/1.e3)
        print '============= %d,%d-%dms, %s, %s ==================='%(shot, start_time, end_time, i, sensor_dict['name'])
        tmp = num.abs(sensor_dict[i+'_alpha'])
        print '%-10s |'%('n'), ['| %-10d |'%(j) for j in range(0,len(tmp))]
        print '%-10s |'%('sensor_amp'), ['| %-10.2e |'%(tmp[j]) for j in range(0,len(tmp))]
        tmp = num.angle(sensor_dict[i+'_alpha'],deg=True)
        print '%-10s |'%('sensor_deg'), ['| %-10.2e |'%(tmp[j]) for j in range(0,len(tmp))]
        tmp = num.abs(icoil_dict['total_alpha'])
        print '%-10s |'%('icoil_amp'), ['| %-10.2e |'%(tmp[j]) for j in range(0,len(tmp))]
        tmp = num.angle(icoil_dict['total_alpha'],deg=True)
        print '%-10s |'%('icoil_deg'), ['| %-10.2e |'%(tmp[j]) for j in range(0,len(tmp))]
        tmp = num.abs(answer)
        print '%-10s |'%('rel_amp'), ['| %-10.2e |'%(tmp[j]) for j in range(0,len(tmp))]
        tmp = num.angle(answer,deg=True)
        print '%-10s |'%('rel_deg'), ['| %-10.2e |'%(tmp[j]) for j in range(0,len(tmp))]

def return_trans_func(f, sensor_name, coil_name, sample_rate, nd=16, debug=0):
    try:
        sk, sp, sz, Afit, Bfit, np, nz = get_hdf5values(f, sensor_name, coil_name, nd, debug=debug)
        print 'obtained hdf data'
        data_found = 1
    except:
        print 'failed to get data - coupling not recorded because its too weak?'
        data_found = 2
        return 0, 0, 0

    #convert zero-pole-gain representation to polynomial representation
    b_s, a_s = scipy_signal.zpk2tf(sz,sp,sk)
    if debug==1:
        print 'b_s:', ['%.2e'%(i_tmp) for i_tmp in b_s]
        print 'a_s:', ['%.2e'%(i_tmp) for i_tmp in a_s]
        print 'zpk2tf finished'

    #convert from s-domain -> z-domain using bilinear transform
    #note this uses a modified version of the scipy function
    #b_z, a_z = scipy_signal.bilinear(b_s, a_s,fs=sample_rate)
    b_z, a_z = bilinear(b_s, a_s, fs=sample_rate)
    if debug==1:
        print 'b_z:', ['%.2e'%(i_tmp) for i_tmp in b_z]
        print 'a_z:', ['%.2e'%(i_tmp) for i_tmp in a_z]
        print 'bz and az found'
    return a_z, b_z, 1


def return_coupling(a_z, b_z, shot, time,coil_name, existing_signals):
    '''
    get the signal, and return the coupling to the pickup
    '''
    I_coil_current = interpolate_signal(coil_name, shot, time, existing_signals)
    I_coil_RMS = num.sqrt(num.average((num.abs(I_coil_current))**2))
    print 'obtained I-coil signal'

    #calculate the signal that is coupled to the I_coil
    new_signal = scipy_signal.lfilter(b_z, a_z, I_coil_current)
    return new_signal, I_coil_RMS


def fft_contribs(plasma_component, vacuum_contributions, sample_rate, time, I_coil_freq, time_ax = None, freq_ax= None, sensor_name = '', plot_all = None, window = 'boxcar', plot_filtered_signal = 0):
    '''
    with the list of vacuum_contributions subtract each one from the 
    plasma_component list one at a time at each step record the fft 
    value at the perturbation frequency
    '''
    fft_list = []
    window = getattr(scipy_signal, window)
    window_func = window(len(plasma_component))
    fft_freqs = num.fft.fftfreq(len(plasma_component), 1./sample_rate)
    fft_list.append(2.*10000.*num.fft.fft(plasma_component*window_func)/len(plasma_component)) #in Gauss
    for i in range(0, len(vacuum_contributions)):
        plasma_component = plasma_component - vacuum_contributions[i]
        fft_list.append(2.*10000.*num.fft.fft(plasma_component*window_func)/len(plasma_component))
        if time_ax != None and plot_all == 1:
            time_ax.plot(time, plasma_component, '-', label='Plasma Component')
            #freq_ax.plot(fft_freqs, num.abs(fft_list[-1]), '-o', label='plasma_comp')
            freq_ax.stem(fft_freqs, num.abs(fft_list[-1]), linefmt='k--', label='plasma_comp')
    if time_ax != None:
        pos_freqs = fft_freqs.shape[0]/2
        freq_ax.plot(fft_freqs[0:pos_freqs], num.abs(fft_list[-1][0:pos_freqs]), 'b-', label='plasma_comp')
        time_ax.plot(time, plasma_component, label=sensor_name + ' Pickup - Plasma Component')
        freq_ax.grid(); time_ax.grid()
        if plot_filtered_signal == 1:
            fs = 1./((time[1]-time[0])/1000.)
            cutoff = 100.
            b_butter,a_butter = scipy_signal.butter(6,cutoff/fs)
            time_ax.plot(time, scipy_signal.lfilter(b_butter, a_butter, plasma_component))
    sra_answer = sra_analysis(time, I_coil_freq, plasma_component, time_ax)
    #multiply sra_answer by 10000 to brinag change it to G
    return fft_freqs, fft_list, plasma_component, sra_answer*10000.

def extract_data(inp_dict, shot, time, f, coil_name_list, sample_rate, i_coil_freq, existing_signals, plotting=1,ax_ylim=[0,20], window='boxcar', remove_mean = 1, remove_trend = 1, plot_all = 0):
    '''
    extract the data, interpolate it onto 'time', 
    remove coupling to the coils listed in coil_name_list
    pass the function the existing signals so that you don't constantly get data you don't need
    '''
    ncols = 3; nrows = len(inp_dict['pickup_names'])/ncols
    if plotting==1:
        if len(inp_dict['pickup_names'])%ncols!=0: nrows+=1
        fig_time, ax_time = pt.subplots(nrows = nrows, ncols = ncols, sharex = 1, sharey = 1)
        ax_time = ax_time.flatten()
        fig_freq, ax_freq = pt.subplots(nrows = nrows, ncols = ncols, sharex = 1, sharey = 1)
        ax_freq = ax_freq.flatten()

    inp_dict['plasma_results_list'] = []
    inp_dict['plasma_results_list_sra'] = []
    inp_dict['total_results_list'] = []
    inp_dict['vac_results_list'] = []
    inp_dict['signals'] = []
    inp_dict['comp_signals'] = []

    #cycle through the sensors in inp_dictionary
    for ax_loc, sensor_name in enumerate(inp_dict['pickup_names']):
        #sensor_signal = interpolate_signal(sensor_name, shot, time, existing_signals, remove_mean = 1, remove_trend = 1)
        sensor_signal = interpolate_signal(str(sensor_name), shot, time, existing_signals, remove_mean = remove_mean, remove_trend = remove_trend)
        inp_dict['signals'].append(sensor_signal)
        vacuum_contributions = []; vacuum_contrib_list = []
        #cycle through all i-coils to get the couplings
        for coil_name in coil_name_list:
            print '='*10, coil_name, '-', sensor_name, '='*10
            #get transfer functions in az bz form
            a_z, b_z, success = return_trans_func(f, sensor_name, coil_name, sample_rate, nd = 16, debug = 0)
            #only perform this step if a transfer function exists for sensor coil pair
            if success == 1:
                #calculate coupling in time domain
                new_signal, I_coil_RMS = return_coupling(a_z, b_z, shot, time, coil_name, existing_signals)
                pickup_RMS = num.sqrt(num.average((num.abs(sensor_signal))**2))

                #append the coupling in the time domain to a list
                vacuum_contributions.append(new_signal)
                vacuum_contrib_list.append(coil_name)
                transfer_RMS = num.sqrt(num.average((num.abs(new_signal))**2))
                print 'I_coil: %e, pickup: %e,  transfer: %e' %(I_coil_RMS, pickup_RMS, transfer_RMS)
        if plotting == 1:
            #fig, ax = pt.subplots(nrows=2)
            #record the fft at each stage of subtracting out the coupling
            
            print ax_loc, len(ax_freq)
            fft_freqs, fft_list, plasma_component, sra_answer = fft_contribs(sensor_signal, vacuum_contributions, sample_rate, time, i_coil_freq, time_ax=ax_time[ax_loc], freq_ax=ax_freq[ax_loc], sensor_name = sensor_name, plot_all = plot_all, window=window)
            #fft_freqs, fft_list, plasma_component, sra_answer = fft_contribs(sensor_signal, vacuum_contributions, sample_rate, time, i_coil_freq, time_ax=ax[0], freq_ax=ax[1], sensor_name = sensor_name, window=window, plot_all = plot_all)
            ax_freq[ax_loc].set_xlim([0, 40])
            if ax_ylim!=None:
                ax_freq[ax_loc].set_ylim(ax_ylim)
        else:
            #record the fft at each stage of subtracting out the coupling
            #returning the plasma_component is probably unnecessary
            fft_freqs, fft_list, plasma_component, sra_answer = fft_contribs(sensor_signal, vacuum_contributions, sample_rate, time, i_coil_freq, sensor_name = sensor_name, window = window, plot_all = plot_all)
        #record the answers at the i_coil_frequency for plasma, total and vac cases,
        #total is fft_list[0], plasma is fft_list[-1] and vac is the difference between them
        freq_loc = num.argmin(num.abs(fft_freqs-i_coil_freq))
        inp_dict['plasma_results_list'].append(fft_list[-1][freq_loc])
        inp_dict['plasma_results_list_sra'].append(sra_answer)
        inp_dict['total_results_list'].append(fft_list[0][freq_loc])
        inp_dict['comp_signals'].append(plasma_component)
        #vacuum is the difference between the two
        #need to double check that this is correct
        inp_dict['vac_results_list'].append(inp_dict['total_results_list'][-1]-inp_dict['plasma_results_list'][-1])
        #freq_band = [8.,12.]
        #length = 1./10*3*sample_rate
        #I_coil_current = interpolate_signal('iu30', shot, time)
        #complex_array, phase_array, amp_array = sfft(I_coil_current, plasma_component, time, length, fs=sample_rate, ax = overall_fft_ax[1], label=sensor_name)
        #complex_array, phase_array, amp_array = sfft(I_coil_current, sensor_signal, time, length, fs=sample_rate, ax = overall_fft_ax[0], label=sensor_name)
        #complex_array, phase_array, amp_array = sfft(I_coil_current, sensor_signal-plasma_component, time, length, fs=sample_rate, ax = overall_fft_ax[2], label=sensor_name)
        #hilbert_func(I_coil_current, plasma_component, time, freq_band, ax = overall_ax[1], label=sensor_name)
        #hilbert_func(I_coil_current, sensor_signal, time, freq_band, ax = overall_ax[0], label=sensor_name)
        #hilbert_func(I_coil_current, sensor_signal-plasma_component, time, freq_band, ax = overall_ax[2], label=sensor_name)
    if plotting==1:
        fig_time.canvas.draw(); fig_time.show()
        fig_freq.canvas.draw(); fig_freq.show()
    return inp_dict

def sra_analysis(time, I_coil_freq, plasma_component, time_ax):
    '''
    sinusoidal regression analysis, to find the amplitude and phase of the
    I_coil_frequency in the plasma component of the signal
    '''
    tmp = time/1000.*I_coil_freq*2.*num.pi
    sin_basis = num.sin(tmp)
    cos_basis = num.cos(tmp)
    sin_basis = sin_basis.reshape(sin_basis.shape[0],1)
    cos_basis = cos_basis.reshape(cos_basis.shape[0],1)
    sin_ans = num.linalg.lstsq(sin_basis, plasma_component)
    cos_ans = num.linalg.lstsq(cos_basis, plasma_component)
    sra_amp = num.sqrt(cos_ans[0]**2+sin_ans[0]**2)
    sra_phase = num.arctan2(cos_ans[0],sin_ans[0])-num.pi/2.
    sra_answer = sra_amp * (num.cos(sra_phase)+1j*num.sin(sra_phase))
    if time_ax != None:
        time_ax.plot(time, sra_answer*num.exp(1j*tmp), 'o')
    return sra_answer

def get_alpha(array, shot, time, number_of_samples, n_list):
    F, fft_list, output_list = array_details(array['pickup_names'], shot, time, number_of_samples)
    alpha, residual = calculate_alpha(array['phi'], n_list, F)
    print alpha
    print num.angle(alpha[2],deg=True), num.abs(alpha[2])
    return alpha, residual

def calculate_alpha(phi, n_list, F):
    #construct gamma
    gamma = num.zeros((len(phi), len(n_list)),dtype=complex)
    for i in range(0, len(n_list)):
        for j in range(0, len(phi)):
            gamma[j,i] = num.exp(1j*n_list[i]*phi[j]/180.*num.pi)
    num.set_printoptions(precision=3)
    print gamma
    #calculate alpha using pseudo inverse of gamma
    gamma_inv = num.linalg.pinv(gamma)
    alpha = num.dot(gamma_inv, F)

    print 'condition of gamma_inv: %.2f, gamma: %.2f'%(num.linalg.cond(gamma_inv), num.linalg.cond(gamma))
    #print 'alpha max: %.2e, min: %.2e, ratio:%.2f'%(num.max(num.abs(alpha)),num.min(num.abs(alpha)),num.max(num.abs(alpha))/num.min(num.abs(alpha)))
    #print 'condition of alpha: %.2f'%(num.linalg.cond(alpha))

    if num.sum(num.abs(num.array(F)))==0.:
        print 'F is too small'
        residual = 10
    else:
        print 'reconstructed', num.dot(gamma,alpha)
        diff = num.array(F)-num.dot(gamma,alpha)
        residual = num.sum(num.abs(num.array(F)-num.dot(gamma,alpha))/num.abs(F))/len(F)*100.
        print 'abs F', num.abs(num.array(F))
        print 'absolute diff %', num.abs(diff)/num.abs(num.array(F))*100
        print 'angle recon', num.angle(num.dot(gamma,alpha),deg=True)
        print 'angle original', num.angle(num.array(F), deg=True)
    return alpha, residual


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
