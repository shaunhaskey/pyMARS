import copy
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pt
import scipy.ndimage.filters as scipy_filt

class test1():
    def __init__(self, file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = 5, reference_offset=[2,0], reference_dB_kink='plas',sort_name = 'q95_list', try_many_phasings = True):
        self.project_dict = pickle.load(file(file_name,'r'))
        self.key_list = self.project_dict['sims'].keys()
        self.n = np.abs(self.project_dict['details']['MARS_settings']['<<RNTOR>>'])
        self.s_surface = s_surface
        self.phasing = phasing
        self.phase_machine_ntor = phase_machine_ntor
        self.fixed_harmonic = fixed_harmonic
        self.reference_offset = reference_offset
        self.reference_dB_kink = reference_dB_kink
        self.sort_name = sort_name
        
        self.extract_q95_Bn(bn_li = 1)
        self.extract_eta_rote()
        self.extract_dB_res()

        self.extract_dB_kink()
        #Create the fixed phasing cases (as set by phasing)
        self.amps_vac_comp = apply_phasing(self.amps_vac_comp_upper, self.amps_vac_comp_lower, self.phasing, self.n, phase_machine_ntor = self.phase_machine_ntor)
        self.amps_plas_comp = apply_phasing(self.amps_plas_comp_upper, self.amps_plas_comp_lower, self.phasing, self.n, phase_machine_ntor = self.phase_machine_ntor)
        self.amps_tot_comp = apply_phasing(self.amps_tot_comp_upper, self.amps_tot_comp_lower, self.phasing, self.n, phase_machine_ntor = self.phase_machine_ntor)

        if self.reference_dB_kink=='plas':
            self.reference = get_reference(self.amps_plas_comp_upper, self.amps_plas_comp_lower, np.linspace(0,2.*np.pi,100), self.n, phase_machine_ntor = self.phase_machine_ntor)
        elif self.reference_dB_kink=='tot':
            self.reference = get_reference(self.amps_tot_comp_upper, self.amps_tot_comp_lower, np.linspace(0,2.*np.pi,100), self.n, phase_machine_ntor = self.phase_machine_ntor)

        #Single cases
        self.plot_quantity_vac, self.mode_list, self.max_loc_list = calculate_db_kink2(self.mk_list, self.q_val_list, self.n, self.reference, self.amps_vac_comp, reference_offset = self.reference_offset)
        self.plot_quantity_plas, self.mode_list, self.max_loc_list = calculate_db_kink2(self.mk_list, self.q_val_list, self.n, self.reference, self.amps_plas_comp, reference_offset = self.reference_offset)
        self.plot_quantity_tot, self.mode_list, self.max_loc_list = calculate_db_kink2(self.mk_list, self.q_val_list, self.n, self.reference, self.amps_tot_comp, reference_offset = self.reference_offset)

        self.upper_values_plasma = self.calculate_db_kink2(self.amps_plas_comp_upper)
        self.lower_values_plasma = self.calculate_db_kink2(self.amps_plas_comp_lower)
        self.upper_values_tot = self.calculate_db_kink2(self.amps_tot_comp_upper)
        self.lower_values_tot = self.calculate_db_kink2(self.amps_tot_comp_lower)
        self.upper_values_vac = self.calculate_db_kink2(self.amps_vac_comp_upper)
        self.lower_values_vac = self.calculate_db_kink2(self.amps_vac_comp_lower)

        #Calculate fixed harmonic dBkink based only on vacuum fields, again upper_values.... are 1D array containing the complex amplitude of fixed harmonic
        self.upper_values_vac_fixed = calculate_db_kink_fixed(self.mk_list, self.q_val_list, self.n, self.amps_vac_comp_upper, self.fixed_harmonic)
        self.lower_values_vac_fixed = calculate_db_kink_fixed(self.mk_list, self.q_val_list, self.n, self.amps_vac_comp_lower, self.fixed_harmonic)
        self.upper_values_plas_fixed = calculate_db_kink_fixed(self.mk_list, self.q_val_list, self.n, self.amps_plas_comp_upper, self.fixed_harmonic)
        self.lower_values_plas_fixed = calculate_db_kink_fixed(self.mk_list, self.q_val_list, self.n, self.amps_plas_comp_lower, self.fixed_harmonic)

        self.plot_quantity_vac_phase = np.angle(self.plot_quantity_vac,deg=True).tolist()
        self.plot_quantity_plas_phase = np.angle(self.plot_quantity_plas,deg=True).tolist()
        self.plot_quantity_tot_phase = np.angle(self.plot_quantity_tot,deg=True).tolist()
        self.plot_quantity_vac = np.abs(self.plot_quantity_vac).tolist()
        self.plot_quantity_plas = np.abs(self.plot_quantity_plas).tolist()
        self.plot_quantity_tot = np.abs(self.plot_quantity_tot).tolist()

        self.q95_list_copy = copy.deepcopy(self.q95_list)
        self.Bn_Li_list_copy = copy.deepcopy(self.Bn_Li_list)

        print self.eta_list
        list_of_item_names = ['eta_list', 'rote_list', 'q95_list', 'Bn_Li_list', 'plot_quantity_plas','plot_quantity_vac', 'plot_quantity_tot', 'plot_quantity_plas_phase', 'plot_quantity_vac_phase', 'plot_quantity_tot_phase', 'mode_list', 'time_list', 'key_list', 'resonant_close']
        list_of_items = zip(*[getattr(self,i) for i in list_of_item_names])
        sort_index = list_of_item_names.index(sort_name)
        print sort_index
        tmp = zip(*sorted(list_of_items, key = lambda sort_val:sort_val[sort_index]))
        output_dict2 = {}
        for loc, i in enumerate(list_of_item_names): output_dict2[i+'_arranged'] = tmp[loc]
        for loc, i in enumerate(list_of_item_names): output_dict2[i] = getattr(self,i)

        if try_many_phasings:
            name_list = ['plot_array_plasma', 'plot_array_vac', 'plot_array_tot', 'plot_array_vac_fixed', 'q95_array', 'phasing_array', 'plot_array_plasma_fixed', 'plot_array_plasma_phase', 'plot_array_vac_phase', 'plot_array_vac_fixed_phase', 'plot_array_plasma_fixed_phase']
            tmp1 = dB_kink_phasing_dependence(self.q95_list_copy, self.lower_values_plasma, self.upper_values_plasma, self.lower_values_vac, self.upper_values_vac, self.lower_values_tot, self.upper_values_tot, self.lower_values_vac_fixed, self.upper_values_vac_fixed, self.phase_machine_ntor, self.upper_values_plas_fixed, self.lower_values_plas_fixed, self.n, n_phases = 360)
            for name, var in zip(name_list, tmp1): output_dict2[name]=var
            name_list = ['plot_array_vac_res', 'plot_array_plas_res','plot_array_tot_res', 'plot_array_vac_res_ave', 'plot_array_plas_res_ave','plot_array_tot_res_ave']
            tmp1 = self.dB_res_phasing_dependence(output_dict2['phasing_array'], output_dict2['q95_array'], self.res_vac_list_upper, self.res_vac_list_lower, self.res_plas_list_upper, self.res_plas_list_lower, self.res_tot_list_upper, self.res_tot_list_lower, self.phase_machine_ntor, self.n)
            for name, var in zip(name_list, tmp1): output_dict2[name]=var

        name_list = ['q95_list_copy', 'max_loc_list', 'upper_values_vac_fixed', 'n', 'lower_values_plasma', 'lower_values_vac']
        for name in name_list: output_dict2[name]=getattr(self,name)
        self.output_dict = output_dict2

    def eta_rote_matrix(self, phasing = 0, med_filt_value = 1):
        '''
        function for plotting resistivity vs rotation scans and
        including different phasings SRH: 31Jan2014
        '''
        print len(self.eta_list), len(self.rote_list), len(self.res_vac_list_upper)
        print len(set(self.eta_list))
        print len(set(self.rote_list))
        eta_vals = sorted(set(self.eta_list))
        rote_vals = sorted(set(self.rote_list))
        dB_res_matrix_vac = np.zeros((len(eta_vals),len(rote_vals)),dtype=float)
        dB_res_matrix_plas = +dB_res_matrix_vac; dB_res_matrix_tot = +dB_res_matrix_vac
        dB_kink_matrix = +dB_res_matrix_vac
        phasings = [0,45,90,135,180,225,270,315]
        fig, ax_orig = pt.subplots(ncols = len(phasings)/2, nrows = 2, sharex = True, sharey = True); ax = ax_orig.flatten()
        fig2, ax2_orig = pt.subplots(ncols = len(phasings)/2, nrows = 2, sharex = True, sharey = True); ax2 = ax2_orig.flatten()
        cm_to_inch=0.393701
        fig.set_figwidth(8.48*2*cm_to_inch)
        fig.set_figheight(8.48*1.1*cm_to_inch)
        fig2.set_figwidth(8.48*2*cm_to_inch)
        fig2.set_figheight(8.48*1.1*cm_to_inch)
        for i, phasing in enumerate(phasings):
            tmp_vac_res, tmp_plas_res, tmp_tot_res, tmp_vac_ave, tmp_plas_ave,  tmp_tot_ave = self.dB_res_single_phasing(phasing,self.phase_machine_ntor, self.n,self.res_vac_list_upper, self.res_vac_list_lower, self.res_plas_list_upper, self.res_plas_list_lower, self.res_tot_list_upper, self.res_tot_list_lower)
            
            name_list = ['plot_array_plasma', 'plot_array_vac', 'plot_array_tot', 'plot_array_vac_fixed', 'q95_array', 'phasing_array', 'plot_array_plasma_fixed', 'plot_array_plasma_phase', 'plot_array_vac_phase', 'plot_array_vac_fixed_phase', 'plot_array_plasma_fixed_phase']
            tmp_kink = dB_kink_phasing_dependence(self.q95_list_copy, self.lower_values_plasma, self.upper_values_plasma, self.lower_values_vac, self.upper_values_vac, self.lower_values_tot, self.upper_values_tot, self.lower_values_vac_fixed, self.upper_values_vac_fixed, self.phase_machine_ntor, self.upper_values_plas_fixed, self.lower_values_plas_fixed, self.n, n_phases = 360, phasing_array = [phasing])
            tmp_dB_kink = tmp_kink[0].flatten()
            print tmp_kink[0].shape
            xaxis = dB_res_matrix_tot*0
            yaxis = dB_res_matrix_tot*0
            for eta, rote, list_index in zip(self.eta_list, self.rote_list, range(len(self.eta_list))):
                row = eta_vals.index(eta)
                col = rote_vals.index(rote)
                #plot_array_vac_res[i,:], plot_array_plas_res[i,:], plot_array_tot_res[i,:], plot_array_vac_res_ave[i,:], plot_array_plas_res_ave[i,:], plot_array_tot_res_ave[i,:] 
                dB_res_matrix_vac[row, col] = +tmp_vac_ave[list_index]
                dB_res_matrix_plas[row, col] = +tmp_plas_ave[list_index]
                dB_res_matrix_tot[row, col] = +tmp_tot_ave[list_index]
                dB_kink_matrix[row, col] = +tmp_dB_kink[list_index]
                yaxis[row, col] = +eta
                xaxis[row, col] = +rote
            color_ax = ax[i].pcolormesh(xaxis, yaxis, scipy_filt.median_filter(dB_res_matrix_tot,med_filt_value), cmap='spectral', rasterized= 'True')
            color_ax2 = ax2[i].pcolormesh(xaxis, yaxis, scipy_filt.median_filter(dB_kink_matrix,med_filt_value), cmap='spectral', rasterized= 'True')
            print color_ax.get_clim()
            print color_ax2.get_clim()
            color_ax.set_clim([0,2])
            color_ax2.set_clim([0,1.5])
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            ax2[i].set_xscale('log')
            ax2[i].set_yscale('log')
            ax[i].set_title(r'$\Delta \phi = {}^o$'.format(phasing))
            ax2[i].set_title(r'$\Delta \phi = {}^o$'.format(phasing))
        ax[-1].set_xlim([1.e-4,1e-1])
        ax2[-1].set_xlim([1.e-4,1e-1])
        for i in ax_orig[:,0]:i.set_ylabel('eta')
        for i in ax_orig[-1,:]:i.set_xlabel('rote')
        for i in ax2_orig[:,0]:i.set_ylabel('eta')
        for i in ax2_orig[-1,:]:i.set_xlabel('rote')
        fig.tight_layout(pad=0.01)
        fig2.tight_layout(pad=0.01)
        cbar = pt.colorbar(color_ax, ax = ax.tolist())
        cbar2 = pt.colorbar(color_ax2, ax = ax2.tolist())
        #ax.imshow(new_matrix_tot)
        cbar.set_label(r'$\delta B_{res}$ vac + plasma')
        cbar2.set_label(r'$\delta B_{kink}$ plasma')
        fig.savefig('res_rot_scan_dBres.pdf')
        fig2.savefig('res_rot_scan_dBkink.pdf')
        fig.canvas.draw(); fig.show()
        fig2.canvas.draw(); fig2.show()

    def calculate_db_kink2(self,to_be_calculated):
        '''
        Calculate db_kink based on the maximum value
        '''
        answer = []; self.mode_list = []; self.max_loc_list = []
        print 'starting'
        for i in range(0,len(self.reference)):
            #allowable_indices = np.array(mk_list[i])>(np.array(q_val_list[i])*(n+0))
            not_allowed_indices = np.array(self.mk_list[i])<=(np.array(self.q_val_list[i])*(self.n+self.reference_offset[1])+self.reference_offset[0])
            tmp_reference = self.reference[i]*1
            tmp_reference[:,not_allowed_indices] = 0

            tmp_phase_loc,tmp_m_loc = np.unravel_index(np.abs(tmp_reference).argmax(), tmp_reference.shape)
            print tmp_phase_loc, tmp_m_loc,self.q_val_list[i]*(self.n), self.mk_list[i][tmp_m_loc], int((self.mk_list[i][tmp_m_loc] - self.q_val_list[i]*self.n))
            maximum_val = tmp_reference[tmp_phase_loc, tmp_m_loc]
            #maximum_val = np.max(np.abs(reference[i])[allowable_indices])
            max_loc = tmp_m_loc
            #max_loc = np.argmin(np.abs(np.abs(reference[i]) - maximum_val))
            self.max_loc_list.append(max_loc)
            self.mode_list.append(self.mk_list[i][max_loc])
            answer.append(to_be_calculated[i][max_loc])

        print 'finishing'
        return answer


    def extract_q95_Bn(self, bn_li = 1):
        '''
        extract some various quantities from a standard pyMARS output dictionary
        '''
        self.q95_list = []; self.Bn_Li_list = []; self.time_list = []
        for i in self.project_dict['sims'].keys():
            self.q95_list.append(self.project_dict['sims'][i]['Q95'])
            self.time_list.append(self.project_dict['sims'][i]['shot_time'])
            if bn_li == 1:
                self.Bn_Li_list.append(self.project_dict['sims'][i]['BETAN']/self.project_dict['sims'][i]['LI'])
            else:
                self.Bn_Li_list.append(self.project_dict['sims'][i]['BETAN'])
    def extract_eta_rote(self,):
        '''
        extract some various quantities from a standard pyMARS output dictionary
        '''
        self.eta_list = []; self.rote_list = []
        for i in self.project_dict['sims'].keys():
            self.eta_list.append(self.project_dict['sims'][i]['MARS_settings']['<<ETA>>'])
            self.rote_list.append(self.project_dict['sims'][i]['MARS_settings']['<<ROTE>>'])

    def extract_dB_res(self,):
        '''
        extract dB_res values from the standard pyMARS output dictionary
        Maybe change this in future to output the total also
        '''
        self.res_vac_list_upper = []; self.res_vac_list_lower = []
        self.res_tot_list_upper = []; self.res_tot_list_lower = []
        self.res_plas_list_upper = []; self.res_plas_list_lower = []
        for i in self.project_dict['sims'].keys():
            self.upper_tot_res = np.array(self.project_dict['sims'][i]['responses']['total_resonant_response_upper'])
            self.lower_tot_res = np.array(self.project_dict['sims'][i]['responses']['total_resonant_response_lower'])
            self.upper_vac_res = np.array(self.project_dict['sims'][i]['responses']['vacuum_resonant_response_upper'])
            self.lower_vac_res = np.array(self.project_dict['sims'][i]['responses']['vacuum_resonant_response_lower'])

            self.res_vac_list_upper.append(self.upper_vac_res)
            self.res_vac_list_lower.append(self.lower_vac_res)
            self.res_tot_list_upper.append(self.upper_tot_res)
            self.res_tot_list_lower.append(self.lower_tot_res)
            self.res_plas_list_upper.append(self.upper_tot_res - self.upper_vac_res)
            self.res_plas_list_lower.append(self.lower_tot_res - self.lower_vac_res)

    def extract_dB_kink(self, upper_lower=True):
        '''
        extract dB_kink information from a standard pyMARS output dictionary
        '''
        if upper_lower:
            self.amps_vac_comp_upper = []; self.amps_vac_comp_lower = []
            self.amps_plas_comp_upper = []; self.amps_plas_comp_lower = []
            self.amps_tot_comp_upper = []; self.amps_tot_comp_lower = []
        else:
            self.amps_vac_comp = [];
            self.amps_plas_comp = [];
            self.amps_tot_comp = [];

        self.mk_list = [];  self.q_val_list = []; self.resonant_close = []
        for i in self.project_dict['sims'].keys():
            tmp = self.project_dict['sims'][i]['responses'][str(self.s_surface)]
            if upper_lower:
                self.relevant_values_upper_tot = tmp['total_kink_response_upper']
                self.relevant_values_lower_tot = tmp['total_kink_response_lower']
                self.relevant_values_upper_vac = tmp['vacuum_kink_response_upper']
                self.relevant_values_lower_vac = tmp['vacuum_kink_response_lower']
            else:
                self.relevant_values_tot = tmp['total_kink_response_single']
                self.relevant_values_vac = tmp['vacuum_kink_response_single']
            self.mk_list.append(tmp['mk'])
            self.q_val_list.append(tmp['q_val'])
            self.resonant_close.append(np.min(np.abs(self.project_dict['sims'][i]['responses']['resonant_response_sq']-self.s_surface)))
            if upper_lower:
                self.amps_plas_comp_upper.append(self.relevant_values_upper_tot-self.relevant_values_upper_vac)
                self.amps_plas_comp_lower.append(self.relevant_values_lower_tot-self.relevant_values_lower_vac)
                self.amps_vac_comp_upper.append(self.relevant_values_upper_vac)
                self.amps_vac_comp_lower.append(self.relevant_values_lower_vac)
                self.amps_tot_comp_upper.append(self.relevant_values_upper_tot)
                self.amps_tot_comp_lower.append(self.relevant_values_lower_tot)
            else:
                self.amps_plas_comp.append(self.relevant_values_tot-self.relevant_values_vac)
                self.amps_vac_comp.append(self.relevant_values_vac)
                self.amps_tot_comp.append(self.relevant_values_tot)

    def dB_res_phasing_dependence(self,phasing_array, q95_array, res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, res_tot_list_upper, res_tot_list_lower, phase_machine_ntor, n):
        '''
        Apply the different upper-lower phasings to the upper and lower runs
        Can probably do this section faster... 
        SH: 26Feb2013
        '''
        #Create the arrays to be populated, q95 columns, phasing rows. 
        plot_array_vac_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
        plot_array_plas_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
        plot_array_tot_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
        plot_array_vac_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
        plot_array_plas_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
        plot_array_tot_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)

        #Cycle through each phasing and calculate dBres and dBres_ave for each of them
        for i, curr_phase in enumerate(phasing_array):
            plot_array_vac_res[i,:], plot_array_plas_res[i,:], plot_array_tot_res[i,:], plot_array_vac_res_ave[i,:], plot_array_plas_res_ave[i,:], plot_array_tot_res_ave[i,:] = self.dB_res_single_phasing(curr_phase,phase_machine_ntor, n,res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, res_tot_list_upper, res_tot_list_lower)
        return plot_array_vac_res, plot_array_plas_res, plot_array_tot_res, plot_array_vac_res_ave, plot_array_plas_res_ave,plot_array_tot_res_ave

    def dB_res_single_phasing(self,curr_phase, phase_machine_ntor, n,res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, res_tot_list_upper, res_tot_list_lower):
        print 'phase :', curr_phase
        phasing = curr_phase/180.*np.pi
        if phase_machine_ntor:
            phasor = (np.cos(-phasing*n)+1j*np.sin(-phasing*n))
        else:
            phasor = (np.cos(phasing)+1j*np.sin(phasing))
        #phasor = (np.cos(curr_phase/180.*np.pi)+1j*np.sin(curr_phase/180.*np.pi))
        tmp_vac_list = []; tmp_plas_list = [];tmp_tot_list = []
        tmp_vac_list2 = []; tmp_plas_list2 = []; tmp_tot_list2 = []


        for ii in range(0,len(res_vac_list_upper)):
            #divisor is for calculating the dBres_ave
            divisor = len(res_vac_list_upper[ii])
            tmp_vac_list.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor)))
            tmp_plas_list.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor)))
            tmp_tot_list.append(np.sum(np.abs(res_tot_list_upper[ii] + res_tot_list_lower[ii]*phasor)))

            tmp_vac_list2.append(tmp_vac_list[-1]/divisor)
            tmp_plas_list2.append(tmp_plas_list[-1]/divisor)
            tmp_tot_list2.append(tmp_tot_list[-1]/divisor)
            #tmp_vac_list2.append(np.sum(np.abs(res_vac_list_upper[ii] + res_vac_list_lower[ii]*phasor))/divisor)
            #tmp_plas_list2.append(np.sum(np.abs(res_plas_list_upper[ii] + res_plas_list_lower[ii]*phasor))/divisor)
        return tmp_vac_list, tmp_plas_list, tmp_tot_list, tmp_vac_list2, tmp_plas_list2,  tmp_tot_list2

    def plot_dB_kink_fixed_vac(self,sort_name = 'rote_list', clim1 = None, clim2 = None, xaxis_type = 'linear', xaxis_label = r'$q_{95}$'):
        xaxis = np.array(self.output_dict[sort_name+'_arranged'])
        if clim1==None: clim1 = [0,4.5]
        if clim2==None: clim2 = [0,0.55]
        cm_to_inch=0.393701
        fig, ax = pt.subplots(nrows = 2, sharex =True, sharey = True)
        #if publication_images:
        #    fig.set_figwidth(8.48*cm_to_inch)
        #    fig.set_figheight(8.48*cm_to_inch)
        #color_plot = ax[0].pcolor(np.array(answers['eta_list_arranged']), answers['phasing_array'], answers['plot_array_plasma'], cmap='hot', rasterized= 'True')
        color_plot = ax[0].pcolormesh(xaxis, self.output_dict['phasing_array'], self.output_dict['plot_array_plasma'], cmap='hot', rasterized= 'True')
        color_plot.set_clim(clim1)
        #color_plot2 = ax[1].pcolor(np.array(answers['eta_list_arranged']), answers['phasing_array'], answers['plot_array_vac_fixed'], cmap='hot', rasterized = 'True')
        color_plot2 = ax[1].pcolormesh(xaxis, self.output_dict['phasing_array'], self.output_dict['plot_array_vac_fixed'], cmap='hot', rasterized = 'True')
        color_plot2.set_clim(clim2)
        fig.canvas.draw();fig.show()
        #ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmax(answers['plot_array_tot'],axis=0)],'kx')
        #ax[0].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_tot'],axis=0)],'b.')

        # suppressed_regions = [[3.81,-30,0.01],[3.48,15,0.1],[3.72,15,0.025],[3.75,0,0.025]]
        # for i in range(0,len(suppressed_regions)):
        #     curr_tmp = suppressed_regions[i]
        #     tmp_angle = curr_tmp[1]*-2.
        #     if tmp_angle<0:tmp_angle+=360
        #     if tmp_angle>360:tmp_angle-=360

        #     ax[0].errorbar(curr_tmp[0], tmp_angle, xerr=curr_tmp[2], yerr=0, ecolor='g')
        #ax[1].plot(answers['q95_array'], answers['phasing_array'][np.argmin(answers['plot_array_vac'],axis=0)],'b.')
        #color_plot.set_clim()
        #ax[1].set_xlabel(r'$q_{95}$', fontsize=14)
        ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)')#,fontsize = 20)
        ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)')#,fontsize = 20)
        
        ax[0].set_xlim([np.min(xaxis), np.max(xaxis)])
        ax[0].set_ylim([np.min(self.output_dict['phasing_array']), np.max(self.output_dict['phasing_array'])])
        #ax[0].plot(np.arange(1,10), np.arange(1,10)*(-55.)+180+180,'b-')
        #ax[1].plot(np.arange(1,10), np.arange(1,10)*(-55.)+180+180,'b-')
        ax[0].locator_params(nbins=4)
        ax[1].locator_params(nbins=4)

        cbar = pt.colorbar(color_plot, ax = ax[0])
        ax[1].set_xlabel(xaxis_label)#, fontsize = 20)
        cbar.ax.set_ylabel(r'$\delta B_{kink}^{n=%d}$ G/kA'%(self.n,))#,fontsize=20)
        cbar.ax.set_title('(a)')

        #cbar.set_ticks(np.round(np.linspace(clim1[0], clim1[1],5),decimals=2))

        cbar = pt.colorbar(color_plot2, ax = ax[1])
        cbar.ax.set_ylabel(r'$\delta B_{vac}^{m=nq+%d,n=%d}$ G/kA'%(self.fixed_harmonic,self.output_dict['n']))#,fontsize=20)
        cbar.ax.set_title('(b)')
        #cbar.set_ticks(np.round(np.linspace(clim2[0], clim2[1],5),decimals=2))
        #cbar.locator.nbins=4
        #cbar.set_ticks(cbar.ax.get_yticks()[::2])
        if xaxis_type=='log':
            ax[0].set_xscale('log')
            ax[1].set_xscale('log')
        fig.canvas.draw();
        fig.savefig('tmp2.eps', bbox_inches='tight', pad_inches=0)
        fig.savefig('tmp2.pdf', bbox_inches='tight', pad_inches=0)
        fig.show()

    def dB_res_n2_dB_res_sum(self,sort_name = 'rote_list', clim1 = None, clim2 = None, xaxis_type = 'linear', xaxis_label = r'$q_{95}$'):
        xaxis = np.array(self.output_dict[sort_name+'_arranged'])
        fig, ax = pt.subplots(nrows = 2, sharex = True, sharey = True); #ax = [ax]#nrows = 2, sharex = True, sharey = True)
        #color_plot = ax[0].pcolor(np.array(self.output_dict['eta_list']), self.output_dict['phasing_array'], self.output_dict['plot_array_vac_res'], cmap='hot', rasterized=True)
        #color_plot = ax[1].pcolor(np.array(self.output_dict['eta_list']), self.output_dict['phasing_array'], self.output_dict['plot_array_plas_res'], cmap='hot', rasterized=True)
        color_plot = ax[0].pcolormesh(xaxis, self.output_dict['phasing_array'], self.output_dict['plot_array_vac_res'], cmap='hot', rasterized=True)
        if clim1 == None: clim1 = [0,25]
        if clim2 == None: clim2 = [0,50]
        color_plot.set_clim(clim1)
        #color_plot2 = ax[1].pcolormesh(xaxis, self.output_dict['phasing_array'], self.output_dict['plot_array_plas_res'], cmap='hot', rasterized=True)
        color_plot2 = ax[1].pcolormesh(xaxis, self.output_dict['phasing_array'], self.output_dict['plot_array_tot_res'], cmap='hot', rasterized=True)
        color_plot2.set_clim(clim2)
        title_string1 = 'Total Forcing'
        title_string2 = 'Average Forcing'
        ax[0].set_xlim([np.min(xaxis), np.max(xaxis)])
        ax[0].set_ylim([0,360])

        #ax[0].set_ylim([np.min(self.output_dict['phasing_array']), np.max(self.output_dict['phasing_array'])])
        ax[1].set_xlabel(xaxis_label, fontsize=20)
        ax[0].set_title(r'$\delta B_{res}^{n=3}$ using vacuum',fontsize=20)
        ax[1].set_title(r'$\delta B_{res}^{n=3}$ using total',fontsize=20)

        ax[1].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
        ax[0].set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
        # ax.set_ylabel(r'$\Delta \phi_{ul}$ (deg)',fontsize = 20)
        #ax[0].set_ylabel('Phasing (deg)')
        if xaxis_type=='log':
            ax[0].set_xscale('log')
            ax[1].set_xscale('log')
        #ax[1].set_ylabel('Phasing (deg)')
        fig2, ax2 = pt.subplots(nrows = 2, sharex = True, sharey = True)
        ax2[0].plot(xaxis, self.output_dict['plot_array_vac_res'][0,:], '-o',label='0deg res vac')
        ax2[0].plot(xaxis, self.output_dict['plot_array_plas_res'][0,:], '-o',label='0deg res plas')
        ax2[0].plot(xaxis, -self.output_dict['plot_array_plas_res'][0,:]+self.output_dict['plot_array_vac_res'][0,:], '-o',label='0deg total')
        ax2[0].plot(xaxis, self.output_dict['plot_array_tot_res'][0,:], '-o',label='0deg total2')
        ax2[1].plot(xaxis, self.output_dict['plot_array_vac_res'][180,:], '-o', label='180deg vac')
        ax2[1].plot(xaxis, self.output_dict['plot_array_plas_res'][180,:], '-o',label='180deg plas')
        ax2[1].plot(xaxis, -self.output_dict['plot_array_plas_res'][180,:]+self.output_dict['plot_array_vac_res'][180,:], '-o', label='180deg total')
        ax2[1].plot(xaxis, self.output_dict['plot_array_tot_res'][180,:], '-o', label='180deg total2')
        if xaxis_type=='log':
            ax2[0].set_xscale('log')
            ax2[1].set_xscale('log')
        ax2[0].legend(loc='best')
        #ax2.plot(np.array(self.output_dict['eta_list']), self.output_dict['plot_array_total_res'][0,:], '-o')
        fig2.canvas.draw();fig2.show()
        cbar = pt.colorbar(color_plot, ax = ax[0])
        cbar.ax.set_ylabel('G/kA',fontsize = 16)
        cbar = pt.colorbar(color_plot2, ax = ax[1])
        cbar.ax.set_ylabel('G/kA',fontsize = 16)
        fig.canvas.draw(); fig.show()


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

def extract_eta_rote(tmp_dict):
    '''
    extract some various quantities from a standard pyMARS output dictionary
    '''
    eta_list = []; rote_list = []
    for i in tmp_dict['sims'].keys():
        eta_list.append(tmp_dict['sims'][i]['MARS_settings']['<<ETA>>'])
        rote_list.append(tmp_dict['sims'][i]['MARS_settings']['<<ROTE>>'])
    return eta_list, rote_list

def extract_dB_res(tmp_dict, return_total = False):
    '''
    extract dB_res values from the standard pyMARS output dictionary
    Maybe change this in future to output the total also
    '''
    res_vac_list_upper = []; res_vac_list_lower = []
    res_tot_list_upper = []; res_tot_list_lower = []
    res_plas_list_upper = []; res_plas_list_lower = []
    for i in tmp_dict['sims'].keys():
        upper_tot_res = np.array(tmp_dict['sims'][i]['responses']['total_resonant_response_upper'])
        lower_tot_res = np.array(tmp_dict['sims'][i]['responses']['total_resonant_response_lower'])
        upper_vac_res = np.array(tmp_dict['sims'][i]['responses']['vacuum_resonant_response_upper'])
        lower_vac_res = np.array(tmp_dict['sims'][i]['responses']['vacuum_resonant_response_lower'])

        res_vac_list_upper.append(upper_vac_res)
        res_vac_list_lower.append(lower_vac_res)
        res_tot_list_upper.append(upper_tot_res)
        res_tot_list_lower.append(lower_tot_res)
        res_plas_list_upper.append(upper_tot_res - upper_vac_res)
        res_plas_list_lower.append(lower_tot_res - lower_vac_res)
    if return_total:
        return res_vac_list_upper, res_vac_list_lower, res_plas_list_upper, res_plas_list_lower, res_tot_list_upper, res_tot_list_lower
    else:
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


def dB_kink_phasing_dependence(q95_list_copy, lower_values_plasma, upper_values_plasma, lower_values_vac, upper_values_vac, lower_values_tot, upper_values_tot, lower_values_vac_fixed, upper_values_vac_fixed, phase_machine_ntor, upper_values_plas_fixed, lower_values_plas_fixed, n, n_phases = 360, phasing_array=None):
    '''
    Calculate dBkink upper-lower phasing dependence

    SH: 26Feb2013

    '''
    #Work on the phasing as a function of q95
    if phasing_array == None:
        phasing_array = np.linspace(0,360,n_phases)
    else:
        phasing_array = np.array(phasing_array)
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
    plot_array_tot_res = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
    plot_array_vac_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
    plot_array_plas_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)
    plot_array_tot_res_ave = np.ones((phasing_array.shape[0], len(q95_array)),dtype=float)

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
