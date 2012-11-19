import matplotlib.pyplot as pt
import numpy as np
try:
    import data
except ImportError:
    print 'not on DIII-D data module not available'

import pickle

relevant_data = ['betan', 'q95', 'LI']
shots = [146392, 146398, 146397, 146388, 146382, 148765]
start_times= [3000,3000,3000,3000,2387,2392]
end_times = [4123,3906,4258,4454,4862,5288]
pickle_details = 1
i_coil_details = 0
from_pickle_file = 1
answers = {}

include_shots = [146392,146398,146397]
if from_pickle_file:
    for loc, shot in enumerate(shots):
        if shot in include_shots:
            pickle_details = pickle.load(file('/home/srh112/NAMP_datafiles/%d_details.pickle'%shot))
            answers[shot]={}
            answers[shot]['x']=np.array(range(start_times[loc],end_times[loc],100))
            for i in relevant_data:
                pickup_data_x = pickle_details[i]['x']
                pickup_data_y = pickle_details[i]['y']
                answers[shot][i] = np.interp(answers[shot]['x'], np.array(pickup_data_x).flatten(), np.array(pickup_data_y).flatten())
    fig, ax = pt.subplots()
    styles = ['x-','.-','o-','x--','.--','o--']
    for i in answers.keys():
        x_axis = answers[i]['q95']
        y_axis = answers[i]['betan']/answers[i]['LI']
        ax.plot(x_axis, y_axis, styles[shots.index(i)], label ='%d %d-%dms'%(i,start_times[shots.index(i)],end_times[shots.index(i)]))
    ax.legend(loc='best')
    ax.set_xlim([2,5])
    ax.set_ylim([1.5,4])
    ax.set_xlabel('q95')
    ax.set_ylabel('Beta_n/L_i')
    ax.set_title('Reflectometer Dataset')
    fig.canvas.draw(); fig.show()



else:
    for loc, shot in enumerate(shots):
        answers[shot]={}
        pickle_answers = {}
        answers[shot]['x']=np.array(range(start_times[loc],end_times[loc],100))
        for i in relevant_data:
            pickup_data = data.Data(i, shot)
            pickle_answers[i] = {}
            pickle_answers[i]['x'] = np.array(pickup_data.x).flatten()
            pickle_answers[i]['y'] = np.array(pickup_data.y).flatten()
            answers[shot][i] = np.interp(answers[shot]['x'], np.array(pickup_data.x).flatten(), np.array(pickup_data.y).flatten())
        if pickle_details:
            file_name = '%d_details.pickle'%(shot)
            pickle.dump(pickle_answers, file(file_name,'w'))
    fig, ax = pt.subplots()
    styles = ['x-','.-','o-','x--','.--','o--']
    for i in answers.keys():
        x_axis = answers[i]['q95']
        y_axis = answers[i]['betan']/answers[i]['LI']
        ax.plot(x_axis, y_axis, styles[shots.index(i)], label ='%d %d-%dms'%(i,start_times[shots.index(i)],end_times[shots.index(i)]))
    ax.legend(loc='best')
    ax.set_xlim([2,5])
    ax.set_ylim([1.5,4])
    ax.set_xlabel('q95')
    ax.set_ylabel('Beta_n/L_i')
    ax.set_title('Experimental dataset')
    fig.canvas.draw(); fig.show()



if i_coil_details:
    lower_array_list = ['IL30', 'IL90','IL150','IL210','IL270','IL330','IU30', 'IU90','IU150','IU210','IU270','IU330']
    loc = 0
    for loc in range(0,len(shots)):
        print '============== %d ================'%(shots[loc])
        for j,i in enumerate(lower_array_list):
            shot = shots[loc]
            new_time_axis = np.arange(start_times[loc]+10,end_times[loc]-10,0.5)
            new_time_axis = np.arange(3050,3850,0.5)
            pickup_data = data.Data(i, shot)
            new_data = np.interp(new_time_axis, np.array(pickup_data.x).flatten(), np.array(pickup_data.y).flatten())
            new_data_fft = 2.*np.fft.fft(new_data)/len(new_data)
            fft_freqs = np.fft.fftfreq(len(new_data),d=(new_time_axis[2]-new_time_axis[1])/1000.)
            max_loc = np.argmax(np.abs(new_data_fft[0:len(fft_freqs)/2]))
            if j == 0:
                ref_phase = np.angle(new_data_fft[max_loc],deg=True)
                phase_answer = 0
            else:
                phase_answer = np.angle(new_data_fft[max_loc],deg=True) - ref_phase
                if phase_answer<0.: phase_answer += 360.
                if phase_answer<0.: phase_answer += 360.
                if phase_answer>360.: phase_answer -= 360.

            print 'freq : %8.2fHz, amp : %8.2f, phase : %8.2f, cos(phase): %8.2f'%(fft_freqs[max_loc], np.abs(new_data_fft[max_loc]), phase_answer, np.cos(phase_answer/180.*np.pi))

