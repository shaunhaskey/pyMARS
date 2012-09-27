import numpy as np
import scipy.interpolate as interp
import matplotlib.mlab as mlab
from scipy.interpolate import *
import PythonMARS_funcs as pyMARS
import os, copy,time
from RZfuncs import *
import matplotlib

def combine_data(upper_data, lower_data, phasing):
    phasing = phasing/180.*np.pi
    #print '**********I-coil Phasing******************'
    #print phasing
    #print np.sum(np.abs(upper_data.R-lower_data.R)), np.max(np.abs(upper_data.R-lower_data.R))
    #print np.sum(np.abs(upper_data.Z-lower_data.Z)), np.max(np.abs(upper_data.Z-lower_data.Z))
    #print '****************************'
    phasor = (np.cos(phasing)+1j*np.sin(phasing))
    #print '%.5f,%.5f'%(np.cos(phasing),np.sin(phasing))
    #phasor = -0.5000 + 0.86603*1j
    B1 = np.array(upper_data.B1) + np.array(lower_data.B1)*phasor
    B2 = np.array(upper_data.B2) + np.array(lower_data.B2)*phasor
    B3 = np.array(upper_data.B3) + np.array(lower_data.B3)*phasor
    Bn = np.array(upper_data.Bn) + np.array(lower_data.Bn)*phasor
    BMn = np.array(upper_data.BMn) + np.array(lower_data.BMn)*phasor
    BnPEST = np.array(upper_data.BnPEST) + np.array(lower_data.BnPEST)*phasor
    return upper_data.R, upper_data.Z, B1, B2, B3, Bn, BMn, BnPEST


class data():
    def __init__(self, directory,Nchi=513,link_RMZM=1, I0EXP=1.0e+3 * 3./np.pi, spline_B23=2):
        self.directory = copy.deepcopy(directory)
        self.Nchi = copy.deepcopy(Nchi)
        self.link_RMZM = copy.deepcopy(link_RMZM)
        self.I0EXP = copy.deepcopy(I0EXP)
        self.spline_B23 = spline_B23

        self.extract_single()

    def extract_single(self):
        os.chdir(self.directory) 
        self.chi = np.linspace(np.pi*-1,np.pi,self.Nchi)
        self.chi.resize(1,len(self.chi))
        self.phi = np.linspace(0,2.*np.pi,self.Nchi)
        self.phi.resize(len(self.phi),1)

        #file_name = 'RMZM_F'
        #if self.link_RMZM ==1:
        #    os.system('ln -f -s ../../cheaserun/RMZM_F RMZM_F')

        file_name = '../../cheaserun/RMZM_F'
        self.RM, self.ZM, self.Ns, self.Ns1, self.Ns2, self.Nm0, self.R0EXP, self.B0EXP, self.s = readRMZM(file_name)
        print self.Ns, self.Ns1, self.Ns2
        #print 'R0EXP', self.R0EXP
        self.Nm2 = self.Nm0
        self.R, self.Z =  GetRZ(self.RM,self.ZM,self.Nm0,self.Nm2,self.chi,self.phi)
        self.FEEDI = get_FEEDI('FEEDI')
        self.BNORM = calc_BNORM(self.FEEDI, self.R0EXP, I0EXP = self.I0EXP)
        print self.directory, self.BNORM, self.FEEDI, self.R0EXP, self.I0EXP
        file_name = 'BPLASMA'
        self.BM1, self.BM2, self.BM3, self.Mm = ReadBPLASMA(file_name, self.BNORM, self.Ns, self.s, spline_B23=self.spline_B23)
        self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian = GetUnitVec(self.R, self.Z, self.s, self.chi)
        self.B1,self.B2,self.B3,self.Bn,self.BMn = GetB123(self.BM1, self.BM2, self.BM3, self.R, self.Mm, self.chi, self.dRdchi, self.dZdchi)
        self.Br,self.Bz,self.Bphi = MacGetBphysC(self.R,self.Z,self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian,self.B1,self.B2,self.B3)
        self.Brho,self.Bchi,self.Bphi2 = MacGetBphysT(self.R,self.Z,self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian,self.B1,self.B2,self.B3,self.B0EXP)
        self.NW = int(round(float(pyMARS.extract_value('../../cheaserun/log_chease','NW',' '))))

    def get_VPLASMA(self, VNORM=1.0):
        os.chdir(self.directory)
        self.VNORM = calc_VNORM(self.FEEDI, self.B0EXP, I0EXP = self.I0EXP)
        self.VM1, self.VM2, self.VM3, self.DPSIDS, self.T = ReadVPLASMA('VPLASMA',self.Ns, self.Ns1, self.s, VNORM=self.VNORM)
        self.V1,self.V2,self.V3,self.Vn, self.V1m = GetV123(self.VM1,self.VM2,self.VM3,self.R, self.chi, self.dRds, self.dZds, self.dRdchi, self.dZdchi, self.jacobian, self.Mm, self.Nchi, self.s, self.Ns1, self.DPSIDS, self.T)
        self.Vr, self.Vz, self.Vphi = MacGetVphys(self.R,self.Z,self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian,self.V1,self.V2,self.V3, self.Ns1)

    def plot_Bn(self, plot_quantity, axis, start_surface = 0, end_surface = 300, skip = 1,cmap='spectral', wall_grid = 23, plot_coils_switch=0, plot_boundaries = 0):
        #print self.R.shape, self.Z.shape, self.Bn.shape
        import matplotlib.pyplot as pt

        grid_R = self.R*self.R0EXP
        grid_Z = self.Z*self.R0EXP
        print 'sh_mod', np.max(grid_R), np.min(grid_R), np.max(grid_Z), np.min(grid_Z)
        color_plot = axis.pcolor(grid_R[start_surface:end_surface:skip,:], grid_Z[start_surface:end_surface:skip,:], plot_quantity[start_surface:end_surface:skip,:],cmap = cmap)
        #self.phase_plot = ax2.pcolor(grid_R[start_surface:end_surface:skip,:], grid_Z[start_surface:end_surface:skip,:], np.angle(plot_quantity[start_surface:end_surface:skip,:],deg=True),cmap = cmap)
        
        def plot_coils(ax1, grid_R, grid_Z, plot_list = range(0,13)):
            print 'plotting coils'
            probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL','Inner_pol','Inner_rad', 'MPI11M067',' MPI2A067','MPI2B067', 'MPI66M067']
            #probe type 1: poloidal field, 2: radial field
            probe_type   = np.array([     1,     1,     1,     0,     0,     0,     0, 1, 0, 1, 1, 1, 1])
            #Poloidal geometry
            Rprobe = np.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300,1.,1., 0.973, 0.973, 0.972, 2.413,])
            Zprobe = np.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714,0.,0., -0.004, 0.518, -0.518, 0.003])
            tprobe = np.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6,-90.,0., 90.0, 89.8, 89.8, -89.9])*2*np.pi/360  #DTOR # poloidal inclination
            lprobe = np.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680, 0.05,0.05, 0.141, 0.140, 0.141, 0.141])  # Length of probe

            # Poloidal geometry
            Nprobe = len(probe)
            Navg = 21    # points along probe to interpolate
            Bprobem = [] # final output

            for k in plot_list:
                #depending on poloidal/radial
                if probe_type[k] == 1:
                    #print Rprobe[k],lprobe[k],tprobe[k], np.cos(tprobe[k])
                    #Rprobek=Rprobe[k] + lprobe[k]/2.*np.sin(tprobe[k])*np.linspace(-1,1,num = Navg)
                    #Zprobek=Zprobe[k] - lprobe[k]/2.*np.cos(tprobe[k])*np.linspace(-1,1,num = Navg)
                    Rprobek=Rprobe[k] + lprobe[k]/2.*np.cos(tprobe[k])*np.linspace(-1,1,num = Navg)
                    Zprobek=Zprobe[k] + lprobe[k]/2.*np.sin(tprobe[k])*np.linspace(-1,1,num = Navg)
                else:
                    #print 'radial'
                    Rprobek=Rprobe[k] + lprobe[k]/2.*np.sin(tprobe[k])*np.linspace(-1,1,num = Navg)
                    Zprobek=Zprobe[k] - lprobe[k]/2.*np.cos(tprobe[k])*np.linspace(-1,1,num = Navg)
                ax1.plot(Rprobek,Zprobek,'o-',label = probe[k], linewidth=3.0)
                #ax1.plot(Rprobe[k],Zprobe[k],'o')

            coilN  = np.array([[2.164, 1.012, 2.374, 0.504],[2.164, -1.012, 2.374, -0.504]])
            ax1.plot(coilN[0,0::2],coilN[0,1::2], 'ko', markersize = 5)
            ax1.plot(coilN[1,0::2],coilN[1,1::2], 'ko', markersize = 5)
        def plot_surface_lines(ax1, grid_R, grid_Z, surface, style):
            ax1.plot(grid_R[surface,:],grid_Z[surface,:],style)

        if plot_boundaries ==1:
            #plot_coils(axis)
            plot_surface_lines(axis, grid_R, grid_Z, self.Ns1 + wall_grid -1,'b-')
            plot_surface_lines(axis, grid_R, grid_Z, self.Ns1 -1,'b--')


        if plot_coils_switch ==1:
            plot_list = range(0,13)
            plot_list.remove(7)
            plot_list.remove(8)
            plot_coils(axis, grid_R, grid_Z, plot_list=plot_list)
        return color_plot

    def get_PEST(self, facn = 3.1416/2):
        os.chdir(self.directory) 
        #os.system('ln -f -s ../../cheaserun_PEST/RMZM_F RMZM_F_PEST')
        II=np.arange(1,self.Ns1+21,dtype=int)-1
        BnEQAC = copy.deepcopy(self.Bn[II,:])
        R_EQAC = copy.deepcopy(self.R[II,:])
        Z_EQAC = copy.deepcopy(self.Z[II,:])
        
        Rs = copy.deepcopy(R_EQAC[-1,:])
        Zs = copy.deepcopy(Z_EQAC[-1,:])
        Rc = (np.min(Rs)+np.max(Rs))/2.
        Zc = (np.min(Zs)+np.max(Zs))/2.
        Tg = np.arctan2(Zs-Zc,Rs-Rc)
        BnEDGE = copy.deepcopy(BnEQAC[-1,:])

        file_name = '../../cheaserun_PEST/RMZM_F'
        #file_name = 'RMZM_F_PEST'
        RM, ZM, Ns, Ns1, Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
        Nm2 = copy.deepcopy(Nm0)
        R, Z =  GetRZ(RM, ZM, Nm0, Nm2, self.chi, self.phi)
        dRds, dZds, dRdchi, dZdchi, jacobian = GetUnitVec(R,Z, s, self.chi)

        self.ss = copy.deepcopy(s[II])
        R_PEST = copy.deepcopy(R[II,:])
        Z_PEST = copy.deepcopy(Z[II,:])

        G22_PEST = dRdchi[II,:]**2 + dZdchi[II,:]**2
        G22_PEST[0,:] = copy.deepcopy(G22_PEST[1,:])
        #seems to make sense up to here

        #this section is to make it work in griddata
        R_Z_EQAC = np.ones([len(R_EQAC.flatten()),2],dtype='float')
        R_Z_EQAC[:,0] = R_EQAC.flatten()
        R_Z_EQAC[:,1] = Z_EQAC.flatten()

        R_Z_PEST = np.ones([len(R_EQAC.flatten()),2],dtype='float')
        R_Z_PEST[:,0] = R_PEST.flatten()
        R_Z_PEST[:,1] = Z_PEST.flatten()
        #this section is to make it work in griddata

        BnPEST  = griddata(R_Z_EQAC,BnEQAC.flatten(),R_Z_PEST,method='linear')
        BnPEST.resize(BnEQAC.shape)
        BnPEST = BnPEST*np.sqrt(G22_PEST)*R_PEST

        mk = np.arange(-29,29+1,dtype=int)
        mk.resize(1,len(mk))

        #Fourier transform the data
        expmchi = np.exp(np.dot(-self.chi.transpose(),mk)*1j)
        BMnPEST = np.dot(BnPEST,expmchi)*(self.chi[0,1]-self.chi[0,0])/2./np.pi

        mm = np.arange(-29,29+1,dtype=int)
        mm2 = np.arange(-29,29+1,dtype=int)

        II = mm - mk[0,0] + 1 - 1
        II.resize(len(II),1)

        self.mk = mk[0,II]
        self.mk.resize(1,len(self.mk))
        print 'PEST, facn = %.2f'%(facn)
        BnPEST  = BMnPEST[:,II.flatten()]/facn

        BnPEST[0,:] = BnPEST[1,:]
        BnPEST[-1,:] = BnPEST[-2,:]
        self.BnPEST = BnPEST

    def resonant_strength(self, min_s = 0, power = 1):
        os.chdir(self.directory) 
        #os.system('ln -sf PROFEQ.OUT PROFEQ_PEST')
        #file_name = 'PROFEQ_PEST'
        file_name = 'PROFEQ.OUT'
        n = 2
        qn, sq, q, s, mq = return_q_profile(self.mk,file_name=file_name, n=2)
        mk_grid, ss_grid = np.meshgrid(self.mk.flatten(), self.ss.flatten())
        qn_grid, s_grid = np.meshgrid(q*n, self.s.flatten())
        temp_qn  = griddata((mk_grid.flatten(),ss_grid.flatten()),np.abs(self.BnPEST.flatten()),(q*n, s.flatten()),method='linear')
        temp_discrete  = griddata((mk_grid.flatten(),ss_grid.flatten()),self.BnPEST.flatten(),(qn.flatten()*n, sq.flatten()),method='linear')
        #print s.shape,len(s), temp_qn.shape, len(temp_qn)
        total_integral = 0
        min_location = np.argmin(np.abs(s-min_s))
        print min_s, min_location, s[min_location]
        for i in range(min_location,len(s)-1):
            total_integral += temp_qn[i]*((s[i+1]-s[i])**power)

        #ax2.plot(temp_qn, s,'.-')
        #ax2.plot(mq*0, sq,'o')
        return total_integral, temp_discrete


    def kink_amp(self, psi, q_range, n = 2):
        #self.q_profile
        #self.q_profile_s
        #self.mk, self.ss, self.BnPEST
        os.chdir(self.directory) 
        #os.system('ln -f -s PROFEQ.OUT PROFEQ_PEST')
        #file_name = 'PROFEQ_PEST'
        file_name = 'PROFEQ.OUT'
        qn, sq, q, s, mq = return_q_profile(self.mk,file_name=file_name, n=n)
        self.qn = qn
        self.sq = sq
        self.q_profile = q
        self.q_profile_s = s
        self.mq = mq
        s_loc = np.argmin(np.abs(self.ss-psi))
        relevant_q = q[s_loc]
        print np.max(self.mk), q_range[0]*relevant_q, q_range[1]*relevant_q
        lower_bound = np.argmin(np.abs(self.mk.flatten() - q_range[0]*relevant_q))
        upper_bound = np.argmin(np.abs(self.mk.flatten() - q_range[1]*relevant_q))
        print 'kink_amp: s_loc: %d, self.ss_val: %.2f, self.q_profile_s: %.2f'%(s_loc, self.ss[s_loc], q[s_loc])
        upper_bound_new = np.min([upper_bound, len(self.mk.flatten())-1])
        print lower_bound, upper_bound, upper_bound_new
        print 'relevant_q: %.2f, bounds: %d %d, values: %d, %d'%(relevant_q, lower_bound, upper_bound_new, self.mk.flatten()[lower_bound], self.mk.flatten()[upper_bound_new])
        relevant_values = self.BnPEST[s_loc,lower_bound:upper_bound_new]
        print relevant_values
        print 'sum', np.abs(np.sum(np.abs(relevant_values)))
        return self.mk.flatten()[lower_bound:upper_bound_new], self.ss[s_loc],relevant_values


    def plot1(self,title='',fig_name = '',fig_show = 1,clim_value=[0,1],inc_phase=1, phase_correction=None, cmap = 'gist_rainbow_r', ss_squared = 0, surfmn_file = None, n=2, increase_grid = 0):
        os.chdir(self.directory) 
        #os.system('ln -f -s PROFEQ.OUT PROFEQ_PEST')
        #file_name = 'PROFEQ_PEST'
        file_name = 'PROFEQ.OUT'
        qn, sq, q, s, mq = return_q_profile(self.mk,file_name=file_name, n=n)
        self.q_profile = q
        self.q_profile_s = s
        mk_grid, ss_grid = np.meshgrid(self.mk.flatten(), self.ss.flatten())
        print 'grid values :', np.max(mk_grid.flatten()), np.min(mk_grid.flatten()), np.min(ss_grid.flatten()), np.max(ss_grid.flatten())
        qn_grid, s_grid = np.meshgrid(q*n, self.s.flatten())
        #print q.shape, s.shape, mk_grid.shape, ss_grid.shape, self.BnPEST.shape
        temp_qn  = griddata((mk_grid.flatten(),ss_grid.flatten()),np.abs(self.BnPEST.flatten()),(q*n, s.flatten()),method='linear')
        if increase_grid:
            mk,ss,BnPEST=increase_grid(self.mk.flatten(),self.ss.flatten(),abs(self.BnPEST),number=200)
        else:
            BnPEST= self.BnPEST
            mk = self.mk
            ss = self.ss
        import matplotlib.pyplot as pt
        tmp_cmap = copy.deepcopy(matplotlib.cm.gist_rainbow_r)
        tmp_cmap.name[0] = (0.0, (0.0, 0.0, 0.0))
        dummy = tmp_cmap.name.pop(1)
        tmp_cmap = tmp_cmap.from_list(tmp_cmap.name, tmp_cmap.name)
        ss_plas_edge = np.argmin(np.abs(ss-1.0))


        tmp_mk_range, tmp_ss, tmp_relevant_values = self.kink_amp(0.92, [2,4], n = n)

        if surfmn_file != None:
            import h5py
            fig_tmp, ax_tmp = pt.subplots(nrows = 2, sharex=1, sharey=1)
            #file_name = surfmn_file#'spectral_info.h5'
            tmp_file = h5py.File(surfmn_file)
            stored_data = tmp_file.get('1')
            zdat = stored_data[0][0]; xdat = stored_data[0][1]; ydat = stored_data[0][2]
        
            min_loc = np.argmin(np.abs(xdat[:,0]-(-30.)))
            max_loc = np.argmin(np.abs(xdat[:,0]-(30.)))
            print 'ss, mk values :', min(self.ss), max(self.ss), min(self.mk), max(self.mk)
            max_mk = np.argmin(np.abs(self.mk.flatten()-30))
            min_mk = np.argmin(np.abs(self.mk.flatten()+30))

            #image2 = ax_tmp[1].imshow(abs(self.BnPEST[:,min_mk:max_mk+1]),cmap = tmp_cmap, interpolation='bicubic',aspect='auto',extent = [-30,30,0,1],origin='lower',clim=clim_value)
            #image2 = ax_tmp[1].pcolor(mk,(ss)**2,np.abs(BnPEST),cmap=tmp_cmap,clim=clim_value)
            image2 = ax_tmp[1].pcolor(mk,(ss[:ss_plas_edge]),np.abs(BnPEST[:ss_plas_edge,:]),cmap='hot')#clim_value)
            ax_tmp[1].set_title('MARS-F, max:%.2f'%(np.max(np.abs(BnPEST[:ss_plas_edge,:]))))
            image2.set_clim([0,1.6])
            #image1 = ax_tmp[0].imshow(zdat[min_loc:max_loc+1,:].transpose(),cmap = tmp_cmap, interpolation='bicubic',aspect='auto',extent = [-30,30,0,1],origin='lower',clim=clim_value)
            image1 = ax_tmp[0].pcolor(xdat,ydat,zdat,cmap = 'hot')
            ax_tmp[0].set_title('SURFMN, max:%.2f'%(np.max(np.abs(zdat))))
            image1.set_clim([0,1.6])
            for ax_tmp1 in ax_tmp:
                ax_tmp1.plot(mq,sq,'wo')
                ax_tmp1.plot(mq,sq**2,'yo')
                ax_tmp1.plot(q*n,s,'w--') 
                ax_tmp1.plot(q*n,s**2,'y--') 
                ax_tmp1.set_xlim([-30,30])
                ax_tmp1.set_ylim([0,1])
            fig_tmp.canvas.draw(); fig_tmp.show()
            #MATLAB PART
            RZ_dir = '/home/srh112/Desktop/Test_Case/matlab_outputs/'
            print 'reading in data from ', RZ_dir
            BnPEST_matlab = np.loadtxt(RZ_dir+'PEST_BnPEST_real.txt',delimiter=',')+np.loadtxt(RZ_dir+'PEST_BnPEST_imag.txt',delimiter=',')*1j
            ss_matlab = np.loadtxt(RZ_dir+'PEST_ss.txt',delimiter=',')
            mk_matlab = np.loadtxt(RZ_dir+'PEST_mk.txt',delimiter=',')
            mat_fig, mat_ax = pt.subplots()
            mat_image = mat_ax.pcolor(mk_matlab, (ss_matlab)**2, np.abs(BnPEST_matlab),cmap = tmp_cmap)
            mat_image.set_clim([0,1.2])
            mat_ax.set_xlim([-30,30])
            mat_ax.set_ylim([0,1])
            mat_fig.canvas.draw();mat_fig.show()
            print 'finished reading in data from ', RZ_dir
            fig_tmp, ax_tmp = pt.subplots(nrows = 2)
            diff = np.abs(BnPEST-BnPEST_matlab)
            self.diff = diff
            self.BnPEST = BnPEST
            print 'differences'
            print 'max diff :'
            print np.max(diff), np.mean(np.max(diff)), np.max(diff).shape
            print 'as a percent:'
            self.diff_percent = diff/np.abs(BnPEST)*100.
            print np.max(self.diff_percent), np.mean(self.diff_percent)

            diff_image = ax_tmp[0].pcolor(mk_matlab, ss_matlab, diff, cmap=tmp_cmap)
            diff_image2 = ax_tmp[1].pcolor(mk_matlab, ss_matlab, (diff/np.abs(BnPEST))*100, cmap=tmp_cmap)
            diff_image.set_clim([0,0.1])
            diff_image2.set_clim([0,5])
            fig_tmp.canvas.draw(); fig_tmp.show()
            print diff_image.get_clim(), diff_image2.get_clim()
        fig = pt.figure()
        #ax =fig.add_subplot(121)
        if inc_phase==0:
            ax = fig.add_axes([0.05,0.1,0.65,0.8])
            ax2 = fig.add_axes([0.75,0.1,0.2,0.8],sharey=ax)
        else:
            ax = fig.add_axes([0.05,0.55,0.65,0.4])
            ax2 = fig.add_axes([0.75,0.55,0.2,0.4])
            ax3 = fig.add_axes([0.05,0.05,0.65,0.4])
            ax4 = fig.add_axes([0.75,0.05,0.2,0.4])
        print 'maximum value : ', np.max(np.abs(BnPEST))

        if ss_squared:
            color_ax = ax.pcolor(mk,(ss)**2,np.abs(BnPEST),cmap=tmp_cmap)
            ax.plot(mq,sq**2,'wo')
            ax.plot(q*n,s**2,'w--') 
        else:
            print 'not ss_squared'
            color_ax = ax.pcolor(mk,ss,np.abs(BnPEST),cmap='hot')#tmp_cmap)
            ax.plot(mq,sq,'wo')
            ax.plot(tmp_mk_range, tmp_mk_range*0+tmp_ss,'ko')
            #tmp_relevant_values = self.kink_amp(0.92, [2,4])
            ax.plot(q*n,s,'w--') 

        if inc_phase!=0:
            if phase_correction!=None:
                angles = np.angle(self.BnPEST)-np.angle(phase_correction)
                #mk_angle,ss_angle,BnPEST_angle=increase_grid(self.mk.flatten(),self.ss.flatten(),np.angle(self.BnPEST)-np.angle(phase_correction),number=200)
                #print 'angle correction'
                upper_limit = 2.*np.pi - np.pi/8.
                lower_limit = upper_limit - 2.*np.pi
                a,b=angles.shape
                for i in range(0,a):
                    for j in range(0,b):
                        while angles[i,j]<lower_limit or angles[i,j]>upper_limit:
                            if angles[i,j]<lower_limit:
                                angles[i,j]+=2.*np.pi
                            elif angles[i,j]>upper_limit:
                                angles[i,j]-=2.*np.pi
                mk_angle,ss_angle,BnPEST_angle=increase_grid(self.mk.flatten(),self.ss.flatten(),angles,number=200)
                #print '******',lower_limit, upper_limit, np.max(BnPEST_angle), np.min(BnPEST_angle), np.max(angles), np.min(angles)
            else:
                mk_angle,ss_angle,BnPEST_angle=increase_grid(self.mk.flatten(),self.ss.flatten(),np.angle(self.BnPEST),number=200)

            masked_array = np.ma.array(BnPEST_angle*180./np.pi)
            threshold = 0.05
            new_masked_array = np.ma.masked_where(np.abs(BnPEST) < threshold , masked_array)

            #color_ax3 = ax3.pcolor(mk,ss,BnPEST_angle*180./np.pi,cmap='hot')
            color_ax3 = ax3.pcolor(mk,ss,new_masked_array,cmap='hot')
            color_ax3.set_clim([lower_limit*180./np.pi,upper_limit*180./np.pi])
            pt.colorbar(color_ax3,ax=ax3)
            ax3.set_ylim([0,1])#[min(ss.flatten()),max(ss.flatten())])
            ax3.set_xlim([-29,29])
            ax3.plot(mq,sq,'bo')
            ax3.plot(q*n,s,'b--') 

        if clim_value == None:
            pass
        else:
            color_ax.set_clim(clim_value)
        
        
        ax.set_title(title)
        ax.set_xlim([min(mk.flatten()),max(mk.flatten())])
        ax.set_ylim([0,1])#[min(ss.flatten()),max(ss.flatten())])
        ax.set_xlim([-29,29])
        #pt.colorbar(color_ax, ax=ax)
        pt.colorbar(mappable = color_ax, ax=ax)


        #mk_ss = np.ones([len(R_EQAC.flatten()),2],dtype='float')
#        R_Z_PEST[:,0] = R_PEST.flatten()
#        R_Z_PEST[:,1] = Z_PEST.flatten()
        #print temp_qn.shape, qn_grid.shape, s_grid.shape
        #print s.shape
        #fig2 = pt.figure()
        #ax = fig2.add_subplot(111)
        #ax.plot(q*n,temp_qn,'.-')
        #ax.plot(mq, mq*0,'o')
        #fig2.canvas.draw()
        #fig2.show()

        #fig = pt.figure()
       # ax2 = fig.add_subplot(122)
        ax2.plot(temp_qn, s,'.-')
        ax2.plot(mq*0, sq,'o')
        for i in range(0,len(mq)):
            ax2.plot([0, np.max(temp_qn)],[sq[i], sq[i]],'--')
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])

        if inc_phase!=0:
            temp_qn_angle  = griddata((mk_grid.flatten(),ss_grid.flatten()),angles.flatten(), (q*n, s.flatten()),method='linear')
            ax4.plot(temp_qn_angle*180./np.pi, s,'.-')
            for i in range(0,len(mq)):
                ax4.plot([lower_limit*180./np.pi, upper_limit*180./np.pi],[sq[i], sq[i]],'--')
                ax4.set_xlim([lower_limit*180./np.pi,upper_limit*180./np.pi])
                ax4.set_ylim([0,1])

        if fig_show==1:
            fig.canvas.draw()
            fig.show()
        if fig_name != '':
            print 'saving figure'
            fig.savefig(fig_name, dpi=200)

class results():
    def __init__(self,directory_v,directory_p):
        self.directory_v = directory_v
        self.directory_p = directory_p
    def get_plasma_data(self):
        self.data_p = data(self.directory_p)
    def get_vac_data(self):
        self.data_v = data(self.directory_v)

class results_combine():
    def __init__(self,directory_v_u,directory_v_l, directory_p_u, directory_p_l):
        self.directory_v_u = directory_v_u
        self.directory_v_l = directory_v_l
        self.directory_p_u = directory_p_u
        self.directory_p_l = directory_p_l
    def get_plasma_data_upper(self):
        self.data_p_u = data(self.directory_p_u)
    def get_plasma_data_lower(self):
        self.data_p_l = data(self.directory_p_l)
    def get_vac_data_upper(self):
        self.data_v_u = data(self.directory_v_u)
    def get_vac_data_lower(self):
        self.data_v_l = data(self.directory_v_l)

'''
dir1='/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult1.980/exp0.430/marsrun/RUNrfa_FEEDI-300.vac/'
dir2='/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult1.980/exp0.430/marsrun/RUNrfa_FEEDI-300.p/'

dir1_p_u = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult1.980/exp0.430/marsrun/RUNrfa_COILupper.p/'
dir1_p_l = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult1.980/exp0.430/marsrun/RUNrfa_COILlower.p/'
dir1_v_u = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult1.980/exp0.430/marsrun/RUNrfa_COILupper.vac/'
dir1_v_l = '/scratch/haskeysr/mars/project1_new_eq/shot138344/tc_003/qmult1.980/exp0.430/marsrun/RUNrfa_COILlower.vac/'

combined_data = results_combine(dir1_v_u, dir1_v_l, dir1_p_u, dir1_p_l)
combined_data.get_plasma_data_upper()
combined_data.get_plasma_data_lower()
combined_data.get_vac_data_upper()
combined_data.get_vac_data_lower()

comb_p = copy.deepcopy(combined_data.data_p_l)
comb_v = copy.deepcopy(combined_data.data_v_l)
comb_p_only = copy.deepcopy(combined_data.data_p_l)
comb_p.R,comb_p.Z, comb_p.B1, comb_p.B2, comb_p.B3, comb_p.Bn, comb_p.BMn = combine_data(combined_data.data_p_u, combined_data.data_p_l, -300)
comb_v.R,comb_v.Z, comb_v.B1, comb_v.B2, comb_v.B3, comb_v.Bn, comb_v.BMn = combine_data(combined_data.data_v_u, combined_data.data_v_l, -300)
comb_p.get_PEST()
comb_p.plot1(title='blah', fig_name='/u/haskeysr/hello_test2.png',fig_show=0)
#    def plot1(self,title='',fig_name = '',show_fig = 1):

comb_p_only.B1 = comb_p.B1 - comb_v.B1
comb_p_only.B2 = comb_p.B2 - comb_v.B2
comb_p_only.B3 = comb_p.B3 - comb_v.B3
comb_p_only.Bn = comb_p.Bn - comb_v.Bn
comb_p_only.BMn = comb_p.BMn - comb_v.BMn

comb_p_only.get_PEST()
#comb_p_only.plot1()

class_res = results(dir1,dir2)
class_res.get_plasma_data()
class_res.get_vac_data()
class_res.data_p.get_PEST()
class_res.data_p.plot1()


print '--------------plasma case ------------------'
print np.sum(np.abs(class_res.data_p.R-comb_p.R))
print np.sum(np.abs(class_res.data_p.Z-comb_p.Z))
print np.sum(np.abs(class_res.data_p.B1-comb_p.B1))
print np.sum(np.abs(class_res.data_p.B2-comb_p.B2))
print np.sum(np.abs(class_res.data_p.B3-comb_p.B3))
print np.sum(np.abs(class_res.data_p.Bn-comb_p.Bn))
print np.sum(np.abs(class_res.data_p.BMn-comb_p.BMn))


print '--------------vacuum case ------------------'
print np.sum(np.abs(class_res.data_v.R-comb_v.R))
print np.sum(np.abs(class_res.data_v.Z-comb_v.Z))
print np.sum(np.abs(class_res.data_v.B1-comb_v.B1))
print np.sum(np.abs(class_res.data_v.B2-comb_v.B2))
print np.sum(np.abs(class_res.data_v.B3-comb_v.B3))
print np.sum(np.abs(class_res.data_v.Bn-comb_v.Bn))
print np.sum(np.abs(class_res.data_v.BMn-comb_v.BMn))
'''
