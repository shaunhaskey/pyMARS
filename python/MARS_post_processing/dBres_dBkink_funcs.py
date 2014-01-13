import copy
import numpy as np

def extract_q95_Bn2(tmp_dict):
    '''
    extract some various quantities from a standard pyMARS output dictionary
    '''
    q95_list = []; Bn_Li_list = []; time_list = []; Beta_N = [];
    for i in tmp_dict['sims'].keys():
        q95_list.append(tmp_dict['sims'][i]['Q95'])
        time_list.append(tmp_dict['sims'][i]['shot_time'])
        Bn_Li_list.append(tmp_dict['sims'][i]['BETAN']/tmp_dict['sims'][i]['LI'])
        Beta_N.append(tmp_dict['sims'][i]['BETAN'])
    return q95_list, Bn_Li_list, Beta_N, time_list

def extract_q95_Bn(tmp_dict, bn_li = 1):
    '''
    extract some various quantities from a standard pyMARS output dictionary
    '''
    q95_list = []; Bn_Li_list = []; time_list = []
    for i in tmp_dict['sims'].keys():
        q95_list.append(tmp_dict['sims'][i]['Q95'])
        time_list.append(tmp_dict['sims'][i]['shot_time'])
        if bn_li == 1:
            Bn_Li_list.append(tmp_dict['sims'][i]['BETAN']/tmp_dict['sims'][i]['LI'])
        else:
            Bn_Li_list.append(tmp_dict['sims'][i]['BETAN'])
    return q95_list, Bn_Li_list, time_list

def extract_dB_res(tmp_dict):
    '''
    extract dB_res values from the standard pyMARS output dictionary
    Maybe change this in future to output the total also
    '''
    res_vac_list_upper = []; res_vac_list_lower = []
    res_plas_list_upper = []; res_plas_list_lower = []
    for i in tmp_dict['sims'].keys():
        upper_tot_res = np.array(tmp_dict['sims'][i]['responses']['total_resonant_response_upper'])
        lower_tot_res = np.array(tmp_dict['sims'][i]['responses']['total_resonant_response_lower'])
        upper_vac_res = np.array(tmp_dict['sims'][i]['responses']['vacuum_resonant_response_upper'])
        lower_vac_res = np.array(tmp_dict['sims'][i]['responses']['vacuum_resonant_response_lower'])

        res_vac_list_upper.append(upper_vac_res)
        res_vac_list_lower.append(lower_vac_res)
        res_plas_list_upper.append(upper_tot_res - upper_vac_res)
        res_plas_list_lower.append(lower_tot_res - lower_vac_res)
    return res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower


def extract_dB_kink(tmp_dict, s_surface, upper_lower=True):
    '''
    extract dB_kink information from a standard pyMARS output dictionary
    '''
    if upper_lower:
        amps_vac_comp_upper = []; amps_vac_comp_lower = []
        amps_plas_comp_upper = []; amps_plas_comp_lower = []
        amps_tot_comp_upper = []; amps_tot_comp_lower = []
    else:
        amps_vac_comp = [];
        amps_plas_comp = [];
        amps_tot_comp = [];

    mk_list = [];  q_val_list = []; resonant_close = []
    for i in tmp_dict['sims'].keys():
        tmp = tmp_dict['sims'][i]['responses'][str(s_surface)]
        if upper_lower:
            relevant_values_upper_tot = tmp['total_kink_response_upper']
            relevant_values_lower_tot = tmp['total_kink_response_lower']
            relevant_values_upper_vac = tmp['vacuum_kink_response_upper']
            relevant_values_lower_vac = tmp['vacuum_kink_response_lower']
        else:
            relevant_values_tot = tmp['total_kink_response_single']
            relevant_values_vac = tmp['vacuum_kink_response_single']
        mk_list.append(tmp['mk'])
        q_val_list.append(tmp['q_val'])
        resonant_close.append(np.min(np.abs(tmp_dict['sims'][i]['responses']['resonant_response_sq']-s_surface)))
        if upper_lower:
            amps_plas_comp_upper.append(relevant_values_upper_tot-relevant_values_upper_vac)
            amps_plas_comp_lower.append(relevant_values_lower_tot-relevant_values_lower_vac)
            amps_vac_comp_upper.append(relevant_values_upper_vac)
            amps_vac_comp_lower.append(relevant_values_lower_vac)
            amps_tot_comp_upper.append(relevant_values_upper_tot)
            amps_tot_comp_lower.append(relevant_values_lower_tot)
        else:
            amps_plas_comp.append(relevant_values_tot-relevant_values_vac)
            amps_vac_comp.append(relevant_values_vac)
            amps_tot_comp.append(relevant_values_tot)
    if upper_lower:
        return amps_vac_comp_upper, amps_vac_comp_lower, amps_plas_comp_upper, amps_plas_comp_lower, amps_tot_comp_upper, amps_tot_comp_lower, mk_list, q_val_list, resonant_close
    else:
        return amps_vac_comp, amps_plas_comp, amps_tot_comp, mk_list, q_val_list, resonant_close


def apply_phasing(upper, lower, phasing, n, phase_machine_ntor = 1):
    '''
    Appy a phasing between an upper and lower array quantity
    '''
    answer = []
    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
    for i in range(0,len(upper)):
        answer.append(upper[i] + lower[i] * phasor)
    return answer

def calculate_db_kink(mk_list, q_val_list, n, reference, to_be_calculated):
    '''
    Calculate db_kink based on the maximum value
    This function seems to be superceeded by calculate_db_kink2
    '''
    answer = []; mode_list = []; max_loc_list = []
    answer_phase = []
    #answer_phase = []
    for i in range(0,len(reference)):
        allowable_indices = np.array(mk_list[i])>(np.array(q_val_list[i])*(n))
        maximum_val = np.max(np.abs(reference[i])[allowable_indices])
        max_loc = np.argmin(np.abs(np.abs(reference[i]) - maximum_val))
        max_loc_list.append(max_loc)
        mode_list.append(mk_list[i][max_loc])
        answer.append(to_be_calculated[i][max_loc])
        #answer_phase.append(np.angle(to_be_calculated[i][max_loc], deg = True))
    return answer, mode_list, max_loc_list


def calculate_db_kink2(mk_list, q_val_list, n, reference, to_be_calculated, reference_offset = [0,0]):
    '''
    Calculate db_kink based on the maximum value
    '''
    answer = []; mode_list = []; max_loc_list = []
    answer_phase = []
    #answer_phase = []
    print 'starting'
    for i in range(0,len(reference)):
        #allowable_indices = np.array(mk_list[i])>(np.array(q_val_list[i])*(n+0))
        not_allowed_indices = np.array(mk_list[i])<=(np.array(q_val_list[i])*(n+reference_offset[1])+reference_offset[0])
        tmp_reference = reference[i]*1
        tmp_reference[:,not_allowed_indices] = 0

        tmp_phase_loc,tmp_m_loc = np.unravel_index(np.abs(tmp_reference).argmax(), tmp_reference.shape)
        print tmp_phase_loc, tmp_m_loc,q_val_list[i]*(n), mk_list[i][tmp_m_loc], int((mk_list[i][tmp_m_loc] - q_val_list[i]*n))
        maximum_val = tmp_reference[tmp_phase_loc, tmp_m_loc]
        #maximum_val = np.max(np.abs(reference[i])[allowable_indices])
        max_loc = tmp_m_loc
        #max_loc = np.argmin(np.abs(np.abs(reference[i]) - maximum_val))
        max_loc_list.append(max_loc)
        mode_list.append(mk_list[i][max_loc])
        answer.append(to_be_calculated[i][max_loc])
        #answer_phase.append(np.angle(to_be_calculated[i][max_loc], deg = True))
    print 'finishing'
    return answer, mode_list, max_loc_list


def calculate_db_kink_fixed(mk_list, q_val_list, n, to_be_calculated, n_plus):
    '''
    Calculate db_kink based on a fixed harmonic
    '''
    answer = []
    for i in range(0,len(to_be_calculated)):
        fixed_loc = np.min([np.argmin(np.abs(mk_list[i] - q_val_list[i]*n)) + n_plus, len(to_be_calculated[i])-1])
        answer.append(to_be_calculated[i][fixed_loc])
    return answer


def dB_kink_phasing_dependence(q95_list_copy, lower_values_plasma, upper_values_plasma, lower_values_vac, upper_values_vac, lower_values_tot, upper_values_tot, lower_values_vac_fixed, upper_values_vac_fixed, phase_machine_ntor, upper_values_plas_fixed, lower_values_plas_fixed, n, n_phases = 360):
    '''
    Calculate dBkink upper-lower phasing dependence

    SH: 26Feb2013

    '''
    #Work on the phasing as a function of q95
    phasing_array = np.linspace(0,360,n_phases)
    q95_array = np.array(q95_list_copy)

    #create arrays to work with
    rel_lower_vals_plasma = np.array(lower_values_plasma)
    rel_upper_vals_plasma = np.array(upper_values_plasma)
    rel_lower_vals_vac =  np.array(lower_values_vac)
    rel_upper_vals_vac =  np.array(upper_values_vac)
    rel_lower_vals_tot =  np.array(lower_values_tot)
    rel_upper_vals_tot =  np.array(upper_values_tot)

    rel_lower_vals_vac_fixed =  np.array(lower_values_vac_fixed)
    rel_upper_vals_vac_fixed =  np.array(upper_values_vac_fixed)
    rel_lower_vals_plas_fixed =  np.array(lower_values_plas_fixed)
    rel_upper_vals_plas_fixed =  np.array(upper_values_plas_fixed)

    #Create the arrays for the q95, phasing dependence for amplitude and angle
    plot_array_plasma = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_vac = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_tot = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_vac_fixed = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_plasma_fixed = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)

    plot_array_plasma_phase = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_vac_phase = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_tot_phase = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_vac_fixed_phase = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)
    plot_array_plasma_fixed_phase = np.ones((phasing_array.shape[0], q95_array.shape[0]),dtype=float)

    #run through each phasing and calculate the answer for each one
    for i, curr_phase in enumerate(phasing_array):
        phasing = curr_phase/180.*np.pi
        if phase_machine_ntor:
            phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
        else:
            phasor = (np.cos(phasing)+1j*np.sin(phasing))
        plot_array_plasma[i,:] = np.abs(rel_upper_vals_plasma + rel_lower_vals_plasma*phasor)
        plot_array_vac[i,:] = np.abs(rel_upper_vals_vac + rel_lower_vals_vac*phasor)
        plot_array_tot[i,:] = np.abs(rel_upper_vals_tot + rel_lower_vals_tot*phasor)
        plot_array_vac_fixed[i,:] = np.abs(rel_upper_vals_vac_fixed + rel_lower_vals_vac_fixed*phasor)
        plot_array_plasma_fixed[i,:] = np.abs(rel_upper_vals_plas_fixed + rel_lower_vals_plas_fixed*phasor)

        plot_array_plasma_phase[i,:] = np.angle(rel_upper_vals_plasma + rel_lower_vals_plasma*phasor,deg=True)
        plot_array_vac_phase[i,:] = np.angle(rel_upper_vals_vac + rel_lower_vals_vac*phasor,deg=True)
        plot_array_tot_phase[i,:] = np.angle(rel_upper_vals_tot + rel_lower_vals_tot*phasor,deg=True)
        plot_array_vac_fixed_phase[i,:] = np.angle(rel_upper_vals_vac_fixed + rel_lower_vals_vac_fixed*phasor,deg=True)
        plot_array_plasma_fixed_phase[i,:] = np.angle(rel_upper_vals_plas_fixed + rel_lower_vals_plas_fixed*phasor,deg=True)

    return plot_array_plasma, plot_array_vac, plot_array_tot, plot_array_vac_fixed, q95_array, phasing_array, plot_array_plasma_fixed, plot_array_plasma_phase, plot_array_vac_phase, plot_array_vac_fixed_phase, plot_array_plasma_fixed_phase


def dB_res_single_phasing(curr_phase,phase_machine_ntor, n,res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower):
    print 'phase :', curr_phase
    phasing = curr_phase/180.*np.pi
    if phase_machine_ntor:
        phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
    else:
        phasor = (np.cos(phasing)+1j*np.sin(phasing))
    #phasor = (np.cos(curr_phase/180.*np.pi)+1j*np.sin(curr_phase/180.*np.pi))
    tmp_vac_list = []; tmp_plas_list = []
    tmp_vac_list2 = []; tmp_plas_list2 = []
    
    for ii in range(0,len(res_vac_list_upper)):
        #divisor is for calculating the dBres_ave
        divisor = len(res_vac_list_upper[ii])
        tmp_vac_list.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor)))
        tmp_plas_list.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor)))

        tmp_vac_list2.append(tmp_vac_list[-1]/divisor)
        tmp_plas_list2.append(tmp_plas_list[-1]/divisor)
        #tmp_vac_list2.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor))/divisor)
        #tmp_plas_list2.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor))/divisor)
    return tmp_vac_list, tmp_plas_list, tmp_vac_list2, tmp_plas_list2


def dB_res_phasing_dependence(phasing_array, q95_array, res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, phase_machine_ntor, n):
    '''
    Apply the different upper-lower phasings to the upper and lower runs
    Can probably do this section faster... 
    SH: 26Feb2013
    '''
    #Create the arrays to be populated, q95 columns, phasing rows. 
    plot_array_vac_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
    plot_array_plas_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
    plot_array_vac_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
    plot_array_plas_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)

    #Cycle through each phasing and calculate dBres and dBres_ave for each of them
    for i, curr_phase in enumerate(phasing_array):
        plot_array_vac_res[i,:], plot_array_plas_res[i,:], plot_array_vac_res_ave[i,:], plot_array_plas_res_ave[i,:] = dB_res_single_phasing(curr_phase,phase_machine_ntor, n,res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower)
    return plot_array_vac_res, plot_array_plas_res, plot_array_vac_res_ave, plot_array_plas_res_ave

def get_reference(upper, lower, phasing_list, n, phase_machine_ntor = 1):
    '''
    Appy a phasing between an upper and lower array quantity
    '''
    answer = []
    for i in range(0,len(upper)):
        tmp = []
        for phasing in phasing_list:
            if phase_machine_ntor:
                phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
            else:
                phasor = (np.cos(phasing)+1j*np.sin(phasing))
            tmp.append(upper[i] + lower[i] * phasor)
        answer.append(np.array(tmp))
    return answer


def extract_pmult_qmult(project_dict):
    pmult_list = []; qmult_list = []
    for i in project_dict['sims'].keys():
        pmult_list.append(project_dict['sims'][i]['PMULT'])
        qmult_list.append(project_dict['sims'][i]['QMULT'])
    return pmult_list, qmult_list



def no_wall_limit(q95_list, beta_n_list):
    '''
    Returns the maximum item in beta_n_list for each unique item in q95_list
    Useful for returning the no wall limit
    SH 26/12/2012
    '''
    q95 = np.array(q95_list)
    bn = np.array(beta_n_list)
    q95_values = list(set(q95_list))
    q95_values.sort()
    xaxis = []; yaxis = []; yaxis2 = []
    for i in q95_values:
        print i
        xaxis.append(i)
        yaxis.append(np.max(bn[q95==i]))
        if np.sum(q95==i)>=2:
            increment = yaxis[-1] - np.max(bn[(q95==i) & (bn!=yaxis[-1])])
        else:
            increment=0
        yaxis2.append(yaxis[-1]+increment)
    tmp1 = sorted(zip(xaxis,yaxis,yaxis2))
    xaxis = [tmp for (tmp,tmp2,tmp3) in tmp1]
    yaxis = [tmp2 for (tmp,tmp2,tmp3) in tmp1]
    yaxis2 = [tmp3 for (tmp,tmp2,tmp3) in tmp1]

    return xaxis, yaxis, yaxis2