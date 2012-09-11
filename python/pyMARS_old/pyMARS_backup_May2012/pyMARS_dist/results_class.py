import numpy as num
import scipy.interpolate as interp
import matplotlib.mlab as mlab
from scipy.interpolate import *
import os, copy,time
from RZfuncs import *

def combine_data(upper_data, lower_data, phasing):
    phasing = phasing/180.*num.pi
    #print '**********I-coil Phasing******************'
    #print phasing
    #print num.sum(num.abs(upper_data.R-lower_data.R)), num.max(num.abs(upper_data.R-lower_data.R))
    #print num.sum(num.abs(upper_data.Z-lower_data.Z)), num.max(num.abs(upper_data.Z-lower_data.Z))
    #print '****************************'
    phasor = (num.cos(phasing)+1j*num.sin(phasing))
    #print '%.5f,%.5f'%(num.cos(phasing),num.sin(phasing))
    #phasor = -0.5000 + 0.86603*1j
    B1 = num.array(upper_data.B1) + num.array(lower_data.B1)*phasor
    B2 = num.array(upper_data.B2) + num.array(lower_data.B2)*phasor
    B3 = num.array(upper_data.B3) + num.array(lower_data.B3)*phasor
    Bn = num.array(upper_data.Bn) + num.array(lower_data.Bn)*phasor
    BMn = num.array(upper_data.BMn) + num.array(lower_data.BMn)*phasor
    BnPEST = num.array(upper_data.BnPEST) + num.array(lower_data.BnPEST)*phasor
    return upper_data.R, upper_data.Z, B1, B2, B3, Bn, BMn, BnPEST


class data():
    def __init__(self, directory,Nchi=513,link_RMZM=1, I0EXP=1.0e+3 * 3./num.pi):
        self.directory = copy.deepcopy(directory)
        self.Nchi = copy.deepcopy(Nchi)
        self.link_RMZM = copy.deepcopy(link_RMZM)
        self.I0EXP = copy.deepcopy(I0EXP)
        self.extract_single()
        
    def extract_single(self):
        os.chdir(self.directory) 
        self.chi = num.linspace(num.pi*-1,num.pi,self.Nchi)
        self.chi.resize(1,len(self.chi))
        self.phi = num.linspace(0,2.*num.pi,self.Nchi)
        self.phi.resize(len(self.phi),1)

        file_name = 'RMZM_F'
        if self.link_RMZM ==1:
            os.system('ln -sf ../../cheaserun/RMZM_F RMZM_F')
        self.RM, self.ZM, self.Ns, self.Ns1, self.Ns2, self.Nm0, self.R0EXP, self.B0EXP, self.s = readRMZM(file_name)
            
        #print 'R0EXP', self.R0EXP
        self.Nm2 = self.Nm0
        self.R, self.Z =  GetRZ(self.RM,self.ZM,self.Nm0,self.Nm2,self.chi,self.phi)
        self.FEEDI = get_FEEDI('FEEDI')
        self.BNORM = calc_BNORM(self.FEEDI, self.R0EXP,I0EXP = self.I0EXP)
        
        file_name = 'BPLASMA'
        self.BM1,self.BM2,self.BM3,self.Mm = ReadBPLASMA(file_name, self.BNORM, self.Ns)
        self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian = GetUnitVec(self.R, self.Z, self.s, self.chi)
        self.B1,self.B2,self.B3,self.Bn,self.BMn = GetB123(self.BM1, self.BM2, self.BM3, self.R, self.Mm, self.chi, self.dRdchi, self.dZdchi)
        self.Br,self.Bz,self.Bphi = MacGetBphysC(self.R,self.Z,self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian,self.B1,self.B2,self.B3)

    def get_PEST(self):
        os.chdir(self.directory) 
        os.system('ln -sf ../../cheaserun_PEST/RMZM_F RMZM_F_PEST')
        II=num.arange(1,self.Ns1+21,dtype=int)-1
        BnEQAC = self.Bn[II,:]
        R_EQAC = self.R[II,:]
        Z_EQAC = self.Z[II,:]
        
        Rs = R_EQAC[-1,:]
        Zs = Z_EQAC[-1,:]
        Rc = (num.min(Rs)+num.max(Rs))/2
        Zc = (num.min(Zs)+num.max(Zs))/2
        Tg = num.arctan2(Zs-Zc,Rs-Rc)
        BnEDGE = BnEQAC[-1,:]

        file_name = 'RMZM_F_PEST'
        RM, ZM, Ns,Ns1,Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
        #print self.Ns1, Ns1, self.Ns, Ns, self.Ns2, Ns2, self.Nm0, Nm0
        #print self.R0EXP,R0EXP, self.B0EXP, B0EXP
        #print sum(num.abs(self.s-s))
        Nm2 = Nm0
        R,Z =  GetRZ(RM,ZM,Nm0,Nm2,self.chi,self.phi)
        dRds,dZds,dRdchi,dZdchi,jacobian = GetUnitVec(R,Z, s, self.chi)

        self.ss     = s[II]

        R_PEST = R[II,:]
        Z_PEST = Z[II,:]

        G22_PEST  = dRdchi[II,:]**2 + dZdchi[II,:]**2

        G22_PEST[0,:] = G22_PEST[1,:]

        R_Z_EQAC = num.ones([len(R_EQAC.flatten()),2],dtype='float')
        R_Z_EQAC[:,0] = R_EQAC.flatten()
        R_Z_EQAC[:,1] = Z_EQAC.flatten()

        R_Z_PEST = num.ones([len(R_EQAC.flatten()),2],dtype='float')
        R_Z_PEST[:,0] = R_PEST.flatten()
        R_Z_PEST[:,1] = Z_PEST.flatten()

        BnPEST  = griddata(R_Z_EQAC,BnEQAC.flatten(),R_Z_PEST,method='linear')
        BnPEST.resize(BnEQAC.shape)
        BnPEST = BnPEST*num.sqrt(G22_PEST)*R_PEST 

        mk = num.arange(-29,29+1,dtype=int)
        mk.resize(1,len(mk))

        expmchi = num.exp(num.dot(-self.chi.transpose(),mk)*1j)
        BMnPEST = num.dot(BnPEST,expmchi)*(self.chi[0,1]-self.chi[0,0])/2./num.pi


        mm = num.arange(-29,29+1,dtype=int)
        mm2 = num.arange(-29,29+1,dtype=int)

        II = mm - mk[0,0] + 1 - 1
        II.resize(len(II),1)

        self.mk = mk[0,II]

        self.mk.resize(1,len(self.mk))
        facn = num.pi/2.

        BnPEST  = BMnPEST[:,II.flatten()]/facn

        BnPEST[0,:] = BnPEST[1,:]
        BnPEST[-1,:] = BnPEST[-2,:]
        self.BnPEST = BnPEST

    def resonant_strength(self, min_s = 0, power = 1):
        os.chdir(self.directory) 
        os.system('ln -sf PROFEQ.OUT PROFEQ_PEST')
        file_name = 'PROFEQ_PEST'
        n = 2
        qn, sq, q, s, mq = return_q_profile(self.mk,file_name=file_name, n=2)
        mk_grid, ss_grid = num.meshgrid(self.mk.flatten(), self.ss.flatten())
        qn_grid, s_grid = num.meshgrid(q*n, self.s.flatten())
        temp_qn  = griddata((mk_grid.flatten(),ss_grid.flatten()),num.abs(self.BnPEST.flatten()),(q*n, s.flatten()),method='linear')
        #print s.shape,len(s), temp_qn.shape, len(temp_qn)
        total_integral = 0
        min_location = num.argmin(num.abs(s-min_s))
        print min_s, min_location, s[min_location]
        for i in range(min_location,len(s)-1):
            total_integral += temp_qn[i]*((s[i+1]-s[i])**power)

        #ax2.plot(temp_qn, s,'.-')
        #ax2.plot(mq*0, sq,'o')
        return total_integral


    def plot1(self,title='',fig_name = '',fig_show = 1,clim_value=[0,1],inc_phase=1, phase_correction=None):
        os.chdir(self.directory) 
        os.system('ln -sf PROFEQ.OUT PROFEQ_PEST')
        file_name = 'PROFEQ_PEST'
        qn, sq, q, s, mq = return_q_profile(self.mk,file_name=file_name, n=2)


        n = 2
        mk_grid, ss_grid = num.meshgrid(self.mk.flatten(), self.ss.flatten())
        qn_grid, s_grid = num.meshgrid(q*n, self.s.flatten())
        #print q.shape, s.shape, mk_grid.shape, ss_grid.shape, self.BnPEST.shape
        temp_qn  = griddata((mk_grid.flatten(),ss_grid.flatten()),num.abs(self.BnPEST.flatten()),(q*n, s.flatten()),method='linear')

        mk,ss,BnPEST=increase_grid(self.mk.flatten(),self.ss.flatten(),abs(self.BnPEST),number=200)
        import matplotlib.pyplot as pt
        fig = pt.figure()
        #ax =fig.add_subplot(121)
        if inc_phase==0:
            ax = fig.add_axes([0.05,0.1,0.65,0.8])
            ax2 = fig.add_axes([0.75,0.1,0.2,0.8])
        else:
            ax = fig.add_axes([0.05,0.55,0.65,0.4])
            ax2 = fig.add_axes([0.75,0.55,0.2,0.4])
            ax3 = fig.add_axes([0.05,0.05,0.65,0.4])
            ax4 = fig.add_axes([0.75,0.05,0.2,0.4])

        color_ax = ax.pcolor(mk,ss,abs(BnPEST),cmap='spectral')

        if inc_phase!=0:
            if phase_correction!=None:
                angles = num.angle(self.BnPEST)-num.angle(phase_correction)
                #mk_angle,ss_angle,BnPEST_angle=increase_grid(self.mk.flatten(),self.ss.flatten(),num.angle(self.BnPEST)-num.angle(phase_correction),number=200)
                #print 'angle correction'
                upper_limit = 2.*num.pi - num.pi/8.
                lower_limit = upper_limit - 2.*num.pi
                a,b=angles.shape
                for i in range(0,a):
                    for j in range(0,b):
                        while angles[i,j]<lower_limit or angles[i,j]>upper_limit:
                            if angles[i,j]<lower_limit:
                                angles[i,j]+=2.*num.pi
                            elif angles[i,j]>upper_limit:
                                angles[i,j]-=2.*num.pi
                mk_angle,ss_angle,BnPEST_angle=increase_grid(self.mk.flatten(),self.ss.flatten(),angles,number=200)
                #print '******',lower_limit, upper_limit, num.max(BnPEST_angle), num.min(BnPEST_angle), num.max(angles), num.min(angles)
            else:
                mk_angle,ss_angle,BnPEST_angle=increase_grid(self.mk.flatten(),self.ss.flatten(),num.angle(self.BnPEST),number=200)

            masked_array = num.ma.array(BnPEST_angle*180./num.pi)
            threshold = 0.05
            new_masked_array = num.ma.masked_where(num.abs(BnPEST) < threshold , masked_array)

            #color_ax3 = ax3.pcolor(mk,ss,BnPEST_angle*180./num.pi,cmap='hot')
            color_ax3 = ax3.pcolor(mk,ss,new_masked_array,cmap='hot')
            color_ax3.set_clim([lower_limit*180./num.pi,upper_limit*180./num.pi])
            pt.colorbar(color_ax3,ax=ax3)
            ax3.set_ylim([0,1])#[min(ss.flatten()),max(ss.flatten())])
            ax3.set_xlim([-29,29])
            ax3.plot(mq,sq,'bo')
            n=2
            ax3.plot(q*n,s,'b--') 

        if clim_value == None:
            pass
        else:
            color_ax.set_clim(clim_value)
        
        ax.plot(mq,sq,'wo')
        n=2
        ax.plot(q*n,s,'w--') 
        
        ax.set_title(title)
        ax.set_xlim([min(mk.flatten()),max(mk.flatten())])
        ax.set_ylim([0,1])#[min(ss.flatten()),max(ss.flatten())])
        ax.set_xlim([-29,29])
        pt.colorbar(color_ax,ax=ax)


        #mk_ss = num.ones([len(R_EQAC.flatten()),2],dtype='float')
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
            ax2.plot([0, num.max(temp_qn)],[sq[i], sq[i]],'--')
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])

        if inc_phase!=0:
            temp_qn_angle  = griddata((mk_grid.flatten(),ss_grid.flatten()),angles.flatten(), (q*n, s.flatten()),method='linear')
            ax4.plot(temp_qn_angle*180./num.pi, s,'.-')
            for i in range(0,len(mq)):
                ax4.plot([lower_limit*180./num.pi, upper_limit*180./num.pi],[sq[i], sq[i]],'--')
                ax4.set_xlim([lower_limit*180./num.pi,upper_limit*180./num.pi])
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
print num.sum(num.abs(class_res.data_p.R-comb_p.R))
print num.sum(num.abs(class_res.data_p.Z-comb_p.Z))
print num.sum(num.abs(class_res.data_p.B1-comb_p.B1))
print num.sum(num.abs(class_res.data_p.B2-comb_p.B2))
print num.sum(num.abs(class_res.data_p.B3-comb_p.B3))
print num.sum(num.abs(class_res.data_p.Bn-comb_p.Bn))
print num.sum(num.abs(class_res.data_p.BMn-comb_p.BMn))


print '--------------vacuum case ------------------'
print num.sum(num.abs(class_res.data_v.R-comb_v.R))
print num.sum(num.abs(class_res.data_v.Z-comb_v.Z))
print num.sum(num.abs(class_res.data_v.B1-comb_v.B1))
print num.sum(num.abs(class_res.data_v.B2-comb_v.B2))
print num.sum(num.abs(class_res.data_v.B3-comb_v.B3))
print num.sum(num.abs(class_res.data_v.Bn-comb_v.Bn))
print num.sum(num.abs(class_res.data_v.BMn-comb_v.BMn))
'''
