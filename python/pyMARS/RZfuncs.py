import numpy as np
#import matplotlib.pyplot as pt
from scipy.interpolate import *
from scipy.interpolate import griddata as scipy_griddata
import os,copy
import pyMARS.magnetics_details as mag_details

def I0EXP_calc(N,n,I):
    # Calculate I0EXP based on Chu's memo. This provides the conversion between MARS-F
    # and real geometry based on several assumptions, refer to memo for details
    i = np.arange(0,N,dtype=float)+1.
    #N = 6.
    #n = 2.

    ###I = np.array([1.,-0.5,-0.5,1,-0.5,-0.5]) Was using this one
    #I = np.array([1.,-1.,0.,1,-1.,0.])
    #I = np.array([1.,-1.,1.,-1.,1.,-1.])
    #I = np.array([1.,0.5,-0.5,-1.,-0.5,0.5])

    answer = 2*np.sin(n*np.pi/N)/(n*np.pi)*np.sum(I*np.cos(2*np.pi*n/N*(i-1)))
    I0EXP = answer * 1.e3
    print 'I0EXP :', I0EXP
    return I0EXP

def I0EXP_calc_real(n,I,discrete_pts=1000, produce_plot=0, plot_axes = None, return_components =0):
    '''
    SH : 19Nov2012
    For given coil currents described by list I, and coil geometry described in
    mag_details, calculate the Fourier component n in that particular coil configuration
    '''
    phi_zero = mag_details.coils.phi('I_coils_upper')/180.*np.pi
    phi_range = mag_details.coils.width('I_coils_upper')/180.*np.pi
    phi_tmp = np.linspace(0,2.*np.pi,discrete_pts)
    current_output = phi_tmp * 0

    for i, phi in enumerate(phi_zero):
        phi_range_list = [phi-phi_range[i]/2., phi+phi_range[i]/2.]
        curr_range = ((phi_tmp>phi_range_list[0]) * (phi_tmp<phi_range_list[1]))
        current_output[curr_range] = I[i]

    phi_array = np.array(phi_tmp)
    current_array = np.array(current_output)
    current_fft = np.fft.fft(current_array)
    current_fft_freq = np.fft.fftfreq(len(current_fft),d=(phi_tmp[1]-phi_tmp[0])/(np.pi*2))
    print len(current_fft_freq)
    n_loc = np.argmin(np.abs(current_fft_freq - n))
    if produce_plot:
        if plot_axes == None:
            import matplotlib.pyplot as pt
            fig, ax = pt.subplots(nrows = 2)
        else:
            ax = plot_axes
        ax[0].plot(phi_tmp*180./np.pi, current_array)
        #ax[0].plot(current_array,'.-')
        for i, phi in enumerate(phi_zero):
            phi_tmp1 = phi_zero[i]/np.pi*180.
            phi_range_tmp = phi_range[i]/np.pi*180.
            ax[0].plot([phi_tmp1-phi_range_tmp/2., phi_tmp1-phi_range_tmp/2.],[-1.1,1.1], 'k--')
            ax[0].plot([phi_tmp1+phi_range_tmp/2., phi_tmp1+phi_range_tmp/2.],[-1.1,1.1], 'k--')
            ax[0].text(phi_tmp1,0,'Coil %d'%(i+1,),horizontalalignment='center')
            #ax[0].fill_between(current_array, phi_tmp-phi_range_tmp, phi_tmp+phi_range_tmp, facecolor='blue', alpha=0.5)
            #ax[0].fill_between(phi_tmp/np.pi*180., current_array*0, current_array, facecolor='blue', alpha=0.5)
        ax[0].set_ylim([-1.1,1.1])
        ax[0].set_xlim([0,360])
        ax[0].set_xlabel(r'$\phi$ (deg)',fontsize = 14)
        ax[0].set_ylabel('Current (kA)')
        ax[1].set_xlabel('n')
        ax[1].set_ylabel('Current (kA)')
        start_loc = np.argmin(np.abs(current_fft_freq-0))
        end_loc = np.argmin(np.abs(current_fft_freq-100))
        print start_loc, end_loc
        ax[1].stem(current_fft_freq[start_loc:end_loc], 2.*np.abs(current_fft[start_loc:end_loc]/len(current_fft)), 'b-')
        #ax[1].plot(current_fft_freq[start_loc:end_loc], 2.*np.abs(current_fft[start_loc:end_loc]/len(current_fft)), 'o')
        #ax[1].plot(current_fft_freq, 2.*np.abs(current_fft/len(current_fft)), 'o')
        ax[1].set_xlim([0,11])
        ax[1].set_ylim([0,1.1])
        if plot_axes == None:
            fig.canvas.draw(); fig.show()
    if return_components:
        return 2.*np.abs(current_fft[n_loc]/len(current_fft))*1.e3, current_fft_freq[start_loc:end_loc], 2.*np.abs(current_fft[start_loc:end_loc]/len(current_fft))
    else:
        return 2.*np.abs(current_fft[n_loc]/len(current_fft))*1.e3



def Icoil_MARS_grid_details(coilN,RMZMFILE,Nchi):
    chi = np.linspace(np.pi*-1,np.pi,Nchi)
    chi.resize(1,len(chi))
    phi = np.linspace(0,2.*np.pi,Nchi)
    phi.resize(len(phi),1)
    RM, ZM, Ns, Ns1, Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(RMZMFILE)
    Nm2 = Nm0*1
    R, Z =  GetRZ(RM,ZM,Nm0,Nm2,chi,phi)
    coil = MacResetCoil(R,Z,coilN,Ns1,Ns,R0EXP,chi,s)
    FCCHI, FWCHI, IFEED = MacGetRZcoil(RM,ZM,Nm0,Nm2,coil,s,Ns1)
    return FCCHI, FWCHI, IFEED


def MacGetRZcoil(RM,ZM,Nm0,Nm2,coil,s,Ns1):
    if Nm0<Nm2:
        Nm2 = Nm0*1

    #Define the coils input for MARS-F
    IFEED = [];
    for k in range(0,coil.shape[0]):
        smin = np.min(np.abs(s-coil[k,0]),0)
        II = np.argmin(np.abs(s-coil[k,0]),0)
        #[smin,II] = min(abs(Mac.s-Mac.coil(k,1)));
        IFEED.append(int(II+1-Ns1+1))  #**is this correct - too many +1??

    FCCHI = (coil[:,1] + coil[:,2])/2.
    FWCHI = coil[:,2] - coil[:,1]

    return FCCHI,FWCHI,IFEED

def MacResetCoil(R,Z,coilN,Ns1,Ns,R0EXP,chi,s):
    coil = coilN*1;
    Dm = [];  km = [];
    for kc in range(0,coilN.shape[0]):
        I0 = []; J0 = []; D0 = [];
        for k in range(int(Ns1)-1,int(Ns)):
            Ri, Rj = np.meshgrid(R[k,:],R[k,:])
            Zi, Zj = np.meshgrid(Z[k,:],Z[k,:])
            Ri = Ri*R0EXP; Rj = Rj*R0EXP; Zi = Zi*R0EXP; Zj = Zj*R0EXP
            tmp = np.sqrt((Ri-coilN[kc,0])**2 + (Zi-coilN[kc,1])**2) + np.sqrt((Rj-coilN[kc,2])**2 + (Zj-coilN[kc,3])**2)
            Y = np.min(tmp,0)
            II = np.argmin(tmp,0)
            X = np.min(Y,0)
            JJ = np.argmin(Y,0)
            I0.append(II[JJ])
            J0.append(JJ)
            D0.append(X)
        Dmin = np.min(D0,0)
        kmin = np.argmin(D0,0)
        Imin = I0[kmin]
        Jmin = J0[kmin]
        Dm.append(Dmin)
        km.append(kmin)
        coil[kc,0] = s[Ns1+kmin-1]  #** is this correct?
        chi1 = chi[0,Jmin]/np.pi
        chi2 = chi[0,Imin]/np.pi
        if chi1 > 1.0:
            chi1 = chi1 - 2
        if chi2 > 1.0:
            chi2 = chi2 - 2
        if chi1 > chi2:
            tmp = chi1*1
            chi1 = chi2*1
            chi2 = tmp*1
        coil[kc,1] = chi1
        coil[kc,2] = chi2
    return coil

def MacGetBphysC(R,Z,dRds,dZds,dRdchi,dZdchi,jacobian,B1,B2,B3):
    #return Br, Bz, Bphi based on B1,B2,B3
    Br = (B1*dRds + B2*dRdchi)/jacobian
    Br[0,:] = Br[1,:]
    Bz = (B1*dZds + B2*dZdchi)/jacobian
    Bz[0,:] = Bz[1,:]
    Bphi = B3*R/jacobian
    Bphi[0,:] = Bphi[1,:]
    return Br,Bz,Bphi

def MacGetBphysT(R,Z,dRds,dZds,dRdchi,dZdchi,jacobian,B1,B2,B3,B0EXP):
    #return Brho, Bchi, Bphi based on B1,B2,B3
    sqrtG11 = np.sqrt(dRds**2 + dZds**2)
    sqrtG22 = np.sqrt(dRdchi**2 + dZdchi**2)
    sqrtG33 = R
    B0EXP=1.0
    Brho = B1*sqrtG11/jacobian*B0EXP
    Brho[0,:] = Brho[1,:]
    Bchi = B2*sqrtG22/jacobian*B0EXP
    Bchi[0,:] = Bchi[1,:]
    Bphi = B3*sqrtG33/jacobian*B0EXP
    Bphi[0,:] = Bphi[1,:]
    return Brho,Bchi,Bphi
    

def readRMZM(file_name):
    #read RMZM from Chease run
    RMZM = np.loadtxt(file_name)
    Nm0 = np.round(RMZM[0,0])
    Ns1 = np.round(RMZM[0,1])
    Ns2 = np.round(RMZM[0,2])

    R0EXP = RMZM[0,3]
    B0EXP = RMZM[1,3]

    Ns = Ns1 + Ns2

    s = np.array(RMZM[1:Ns+1,0])
    RM = RMZM[Ns+1:,0] + RMZM[Ns+1:,1]*1j
    ZM = RMZM[Ns+1:,2] + RMZM[Ns+1:,3]*1j

    RM = np.reshape(RM,[Ns,Nm0],order='F')
    ZM = np.reshape(ZM,[Ns,Nm0],order='F')

    RM[:,1:] = 2.*RM[:,1:]
    ZM[:,1:] = 2.*ZM[:,1:]

    return RM, ZM, Ns, Ns1, Ns2, Nm0, R0EXP, B0EXP, s

def GetRZ(RM,ZM,Nm0,Nm2,chi,phi):
    #convert RM and ZM into R and Z real space co-ordinates
    Nm2 = Nm0
    m = np.arange(0,Nm2,1)
    m.resize(len(m),1)

    expmchi = np.exp(np.dot(m,chi)*1j)
    R = np.real(np.dot(RM[:,0:Nm2],expmchi))
    Z = np.real(np.dot(ZM[:,0:Nm2],expmchi))
    return R,Z




def ReadVPLASMA(file_name, Ns, Ns1, s, spline_V23=1,VNORM=1.):
    #Read the BPLASMA output file from MARS-F
    #Return BM1, BM2, BM3
    VPLASMA = np.loadtxt(open(file_name))
 
    Nm1 = VPLASMA[0,0]
    print 'Nm1 ', Nm1
    n = np.round(VPLASMA[0,2])
    print 'VNORM ', VNORM

    Mm = np.round(VPLASMA[1:Nm1+1,0])
    Mm.resize([len(Mm),1])
    DPSIDS = VPLASMA[Nm1+1:Nm1+1+Ns1,0]
    T = VPLASMA[Nm1+1:Nm1+1+Ns1,3]

    VM1 = VPLASMA[Nm1+1+Ns1:,0] + VPLASMA[Nm1+1+Ns1:,1]*1j
    VM2 = VPLASMA[Nm1+1+Ns1:,2] + VPLASMA[Nm1+1+Ns1:,3]*1j
    VM3 = VPLASMA[Nm1+1+Ns1:,4] + VPLASMA[Nm1+1+Ns1:,5]*1j
    print VM1.shape, VM2.shape, VM3.shape
    VM1 = np.reshape(VM1,[Ns1,Nm1],order='F')*VNORM
    VM2 = np.reshape(VM2,[Ns1,Nm1],order='F')*VNORM
    VM3 = np.reshape(VM3,[Ns1,Nm1],order='F')*VNORM
    print VM1.shape, VM2.shape, VM3.shape
    print VM1[100,30],VM2[100,30],VM3[100,30]

    if spline_V23==2:
        'spine_B23 is 2'
        VM2[1:,:] = VM2[0:-1,:]
        VM3[1:,:] = VM3[0:-1,:]
    elif spline_V23==1:
        print 'spline B23 is 1'
        x = (s[0:Ns1-1]+s[1:Ns1])*0.5
        VM2new = copy.deepcopy(VM2)
        print x.flatten().shape, VM2[0:-1,:].shape, s[1:Ns1-1].flatten().shape, VM2new[1:-1,:].shape
        #
        VM2new[1:-1,:] = scipy_griddata(x.flatten(),VM2[0:-1,:],s[1:Ns1-1].flatten(),method='cubic')
        VM2new[0,:] = VM2new[1,:]
        VM2new[-1,:] = VM2new[-2,:]
        VM2 = copy.deepcopy(VM2new)

        x = (s[0:Ns1-1]+s[1:Ns1])*0.5
        VM3new = copy.deepcopy(VM3)
        VM3new[1:-1,:] = scipy_griddata(x.flatten(),VM3[0:-1,:],s[1:Ns1-1].flatten(),method='cubic')
        VM3new[0,:] = VM3new[1,:]
        VM3new[-1,:] = VM3new[-2,:]
        VM3 = copy.deepcopy(VM3new)
    VM1[0,:]=VM1[1,:]

    return VM1, VM2, VM3, DPSIDS, T




def ReadBPLASMA(file_name,BNORM,Ns,s, spline_B23=2):
    #Read the BPLASMA output file from MARS-F
    #Return BM1, BM2, BM3
    BPLASMA = np.loadtxt(open(file_name))
 
    Nm1 = BPLASMA[0,0]
    n = np.round(BPLASMA[0,2])
    Mm = np.round(BPLASMA[1:Nm1+1,0])
    Mm.resize([len(Mm),1])


    BM1 = BPLASMA[Nm1+1:,0] + BPLASMA[Nm1+1:,1]*1j
    BM2 = BPLASMA[Nm1+1:,2] + BPLASMA[Nm1+1:,3]*1j
    BM3 = BPLASMA[Nm1+1:,4] + BPLASMA[Nm1+1:,5]*1j
    print BM1.shape, Ns, Nm1
    BM1 = np.reshape(BM1,[Ns,Nm1],order='F')
    BM2 = np.reshape(BM2,[Ns,Nm1],order='F')
    BM3 = np.reshape(BM3,[Ns,Nm1],order='F')

    print 'BNORM used in ReadBPLASMA', BNORM
    BM1 = BM1[0:Ns,:]*BNORM
    BM2 = BM2[0:Ns,:]*BNORM
    BM3 = BM3[0:Ns,:]*BNORM

    #NEED TO KNOW WHY THIS SECTION IS INCLUDED - to do with half grid???!!
    if spline_B23==2:
        print 'spine_B23 is 2'
        BM2[1:,:] = BM2[0:-1,:]
        BM3[1:,:] = BM3[0:-1,:]
    elif spline_B23==1:
        print 'spline B23 is 1'
        x = (s[0:Ns-1]+s[1:Ns])*0.5
        BM2new = copy.deepcopy(BM2)
        BM2new[1:-1,:] = scipy_griddata(x.flatten(),BM2[0:-1,:],s[1:Ns-1].flatten(),method='cubic')
        print BM2new[1:-1,:].shape
        BM2new[0,:] = 0
        BM2new[-1,:] = BM2new[-2,:]
        BM2 = copy.deepcopy(BM2new)

        x = (s[0:Ns-1]+s[1:Ns])*0.5
        BM3new = copy.deepcopy(BM3)
        BM3new[1:-1,:] = scipy_griddata(x.flatten(),BM3[0:-1,:],s[1:Ns-1].flatten(),method='cubic')
        BM3new[0,:] = 0
        BM3new[-1,:] = BM3new[-2,:]
        BM3 = copy.deepcopy(BM3new)

    return BM1, BM2, BM3, Mm




def GetV123(VM1,VM2,VM3,R, chi, dRds, dZds, dRdchi, dZdchi, jacobian, Mm, Nchi, s, Ns1, DPSIDS, T):
    #Convert BM1, BM2 and BM3 into B1, B2, B3
    expmchi = np.exp(np.dot(Mm,chi)*1j)

    V1a = np.dot(VM1,expmchi)
    V2a = np.dot(VM2,expmchi)
    V3a = np.dot(VM3,expmchi)

    V1 = V1a
    chione = np.ones((1,Nchi))

    #ss = s[0:Mac.Ns1]*chione
    #ss[0,:] = ss[1,:]



    # V2 along e_chi
    DPSIDS = DPSIDS.reshape((DPSIDS.shape[0],1))
    T = T.reshape((T.shape[0],1))
    
    Bchi = np.dot(DPSIDS,chione)/jacobian[0:Ns1,:]
    Bphi = np.dot(T,chione)/R[0:Ns1,:]**2;

    G11  = dRds[0:Ns1,:]**2 + dZds[0:Ns1,:]**2;
    G12  = dRds[0:Ns1,:]*dRdchi[0:Ns1,:] + dZds[0:Ns1,:]*dZdchi[0:Ns1,:]
    G22  = dRdchi[0:Ns1,:]**2 + dZdchi[0:Ns1,:]**2;
    G22[0,:] = G22[1,:]
    G33  = R[0:Ns1,:]**2;
    B2   = (Bchi**2)*G22 + (Bphi**2)*G33
    V2 = -V1*G12*(Bchi**2)/ B2 + V2a*Bphi*G33/B2 + V3a*Bchi

    V3 = -V1*G12*Bchi*Bphi/B2 - V2a*Bchi*G22/B2 + V3a*Bphi
    Vn = V1a*jacobian[0:Ns1,:]/np.sqrt(G33*G22)
    
    expmchi = np.exp(np.dot(np.transpose(-chi),np.transpose(Mm)*1j))
    V1M = np.dot(Vn,expmchi)*(chi[0,1]-chi[0,0])/2./np.pi

    return V1,V2,V3,Vn, V1M



def MacGetVphys(R,Z,dRds,dZds,dRdchi,dZdchi,jacobian,V1,V2,V3, Ns1):
    print '!!!!!!!!! new MacGetVphys !!!!!!!!'
    N = Ns1
    R0 = R[0:N,:]
    Z0 = Z[0:N,:]

    #compute Vr,Vz,Vphi
    jacobian[0,:] = jacobian[1,:]
    fac = 1.0
    Vr = (V1*dRds[0:N,:] + V2*dRdchi[0:N,:])*fac
    Vr[0,:] = Vr[1,:]
    Vz = (V1*dZds[0:N,:] + V2*dZdchi[0:N,:])*fac
    Vz[0,:] = Vz[1,:]
    Vphi = V3*R0*fac
    Vphi[0,:] = Vphi[1,:]

    return Vr,Vz,Vphi

def GetB123(BM1,BM2,BM3,R, Mm, chi, dRdchi, dZdchi):
    #Convert BM1, BM2 and BM3 into B1, B2, B3
    expmchi = np.exp(np.dot(Mm,chi)*1j)

    B1 = np.dot(BM1,expmchi)
    B2 = np.dot(BM2,expmchi)
    B3 = np.dot(BM3,expmchi)
    G22  = dRdchi**2 + dZdchi**2
    G22[0,:] = G22[1,:]
    Bn   = B1/np.sqrt(G22)/R
    expmchi = np.exp(np.dot(-(chi.transpose()), Mm.transpose()*1j))
    BMn = np.dot(Bn, expmchi)*(chi[0,1]-chi[0,0])/2./np.pi

    return B1,B2,B3,Bn, BMn


def GetUnitVec_old(R,Z,s,chi):
    s0 = np.resize(s,[1,len(s)])
    R0 = R
    chi0 = chi
    Z0 = Z
    hs = np.min(s0[0,1:]-s0[0,0:-1])/2.
    hs = np.min([hs,1e-4])
    hchi = np.min(chi0[0,1:]-chi0[0,0:-1])/2
    hchi = np.min([hchi,1e-4])
    s1 = s0 - hs
    s2 = s0 + hs
    chi1 = chi0 - hchi
    chi2 = chi0 + hchi

    R_func = interp.interp1d(s0.flatten(), R0.transpose(), kind='cubic', bounds_error = 0)
    R1 = R_func(s1.flatten())
    R2 = R_func(s2.flatten())
    dRds = np.transpose((R2-R1)/hs/2)

    # compute dZ/ds using Z(s,chi) and spline
    Z_func = interp.interp1d(s0.flatten(), Z0.transpose(), kind='cubic', bounds_error = 0)
    Z1 = Z_func(s1.flatten())
    Z2 = Z_func(s2.flatten())
    dZds = np.transpose((Z2-Z1)/hs/2)


    # compute dR/dchi using R(s,chi) and spline
    R_func2 = interp.interp1d(chi0.flatten(), R0, kind='cubic', bounds_error = 0)
    R1_2 = R_func2(chi1.flatten())
    R2_2 = R_func2(chi2.flatten())

    R1_2[:,0] = R1_2[:,-1]
    R2_2[:,-1] = R2_2[:,0]
    dRdchi = (R2_2-R1_2)/hchi/2

    # compute dZ/dchi using Z(s,chi) and spline
    Z_func2 = interp.interp1d(chi0.flatten(), Z0, kind='cubic', bounds_error = 0)
    Z1_2 = Z_func2(chi1.flatten())
    Z2_2 = Z_func2(chi2.flatten())

    Z1_2[:,0] = Z1_2[:,-1]
    Z2_2[:,-1] = Z2_2[:,0]
    dZdchi = (Z2_2-Z1_2)/hchi/2
    # compute jacobian from (x,y,z) --> (s,chi,phi)
    jacobian = (-dRdchi*dZds + dRds*dZdchi)*R0
    jacobian[0,:] = jacobian[1,:]
    jacobian[-1,:] = jacobian[-2,:]

    return dRds,dZds,dRdchi,dZdchi,jacobian




def GetUnitVec(R,Z,s,chi):
    s0 = np.resize(s,[1,len(s)])
    R0 = R; chi0 = chi; Z0 = Z
    hs = np.min(s0[0,1:]-s0[0,0:-1])/2.
    hs = np.min([hs,1e-4]) #grid offset
    hchi = np.min(chi0[0,1:]-chi0[0,0:-1])/2
    hchi = np.min([hchi,1e-4]) #grid offset

    #describe new grid 
    s1 = s0 - hs
    s2 = s0 + hs
    chi1 = chi0 - hchi
    chi2 = chi0 + hchi

    R1 = R0*0; R2 = R0*0; Z1 = Z0*0; Z2 = Z0*0

    #Interpolate onto the two new slightly offset grids
    R1=np.transpose(scipy_griddata(s0.flatten(),R0,s1.flatten(), method='cubic'))
    R2=np.transpose(scipy_griddata(s0.flatten(),R0,s2.flatten(), method='cubic'))
    Z1=np.transpose(scipy_griddata(s0.flatten(),Z0,s1.flatten(), method='cubic'))
    Z2=np.transpose(scipy_griddata(s0.flatten(),Z0,s2.flatten(), method='cubic'))
    R1_2=np.transpose(scipy_griddata(chi0.flatten(),np.transpose(R0),chi1.flatten(), method='cubic'))
    R2_2=np.transpose(scipy_griddata(chi0.flatten(),np.transpose(R0),chi2.flatten(), method='cubic'))
    Z1_2=np.transpose(scipy_griddata(chi0.flatten(),np.transpose(Z0),chi1.flatten(), method='cubic'))
    Z2_2=np.transpose(scipy_griddata(chi0.flatten(),np.transpose(Z0),chi2.flatten(), method='cubic'))

    dRds = np.transpose(((R2-R1)/hs/2.))
    dZds = np.transpose(((Z2-Z1)/hs/2.))

    #remove nan's that have been created outside the grid, need to check this - is there a better way?
    #matlab doesn't create as many nan's, instead it seems to extrapolate with griddata doesn't do
    #which is probably the correct way to do it
    dRds[0,:]=dRds[1,:];    dRds[-1,:]=dRds[-2,:]
    dZds[0,:]=dZds[1,:];    dZds[-1,:]=dZds[-2,:]
    R1_2[:,0]=R1_2[:,1];    R1_2[:,-1]=R1_2[:,-2]
    R2_2[:,0]=R2_2[:,1];    R2_2[:,-1]=R2_2[:,-2]
    Z1_2[:,0]=Z1_2[:,1];    Z1_2[:,-1]=Z1_2[:,-2]
    Z2_2[:,0]=Z2_2[:,1];    Z2_2[:,-1]=Z2_2[:,-2]

    dRdchi = (R2_2-R1_2)/hchi/2.
    dZdchi = ((Z2_2-Z1_2)/hchi/2.)

    dRdchi[0,:]=dRdchi[1,:]
    dZdchi[0,:]=dZdchi[1,:]

    # compute jacobian from (x,y,z) --> (s,chi,phi)
    jacobian = (-dRdchi*dZds + dRds*dZdchi)*R0
    jacobian[:,0] = jacobian[:,1]
    jacobian[:,-1] = jacobian[:,-2]

    return dRds,dZds,dRdchi,dZdchi,jacobian


def get_FEEDI(file_name):
    #return FEEDI from the mars log file
    #log_file = open(file_name,'r')
    #input_string = log_file.read()
    #log_file.close()
    #success = 0
    #variable_name = 'ABS'
    #name_start_location = input_string.find(variable_name)
    #end_location = input_string[name_start_location:].find('\n')
    #imp_line = input_string[name_start_location:name_start_location+end_location]
    #FEEDI_string = imp_line.strip('ABS(FDI(K))= ')
    #FEEDI_float = float(FEEDI_string)
    #print imp_line, FEEDI_float

    FEEDI_matrix = np.loadtxt(file_name)
    FEEDI_1 = np.sqrt(np.sum(FEEDI_matrix[0,:]**2))
    FEEDI_2 = np.sqrt(np.sum(FEEDI_matrix[1,:]**2))
    FEEDI_float= np.max([FEEDI_1,FEEDI_2])
    print 'FEEDI :',FEEDI_float
    return FEEDI_float

def calc_VNORM(FEEDI, B0EXP, I0EXP=1.0e+3 * 3./np.pi,phas=0.):
    #calculate BNORM
    #phas = 0.5*np.pi
    #phas = 0 #is this correct?
    #I0EXP = 1.0e+3 * 3./np.pi
    mu0   = 4.e-7*np.pi
    vfac   = mu0/B0EXP;
    #FEEDI = 1.0;
    #FEEDI = temp_ans
    VNORM  = vfac*I0EXP/FEEDI
    return VNORM

def calc_BNORM(FEEDI, R0EXP, I0EXP=1.0e+3 * 3./np.pi,phas=0.):
    #calculate BNORM
    #phas = 0.5*np.pi
    #phas = 0 #is this correct?
    BNORM = 1.0*np.exp(phas*1j)
    #I0EXP = 1.0e+3 * 3./np.pi
    mu0   = 4.e-7*np.pi
    fac   = mu0/R0EXP;
    #FEEDI = 1.0;
    #FEEDI = temp_ans
    BNORM  = BNORM*fac*I0EXP/FEEDI*1.e4
    return BNORM

def increase_grid(x, y, z, increase_y = 0, increase_x = 0, new_y_lims = None, number=100):
    #generic use function to re-grid data
    x_grid,y_grid = np.meshgrid(x,y)
    input_grid = np.ones([len(x_grid.flatten()),2],dtype=float)
    input_grid[:,0]=x_grid.flatten()
    input_grid[:,1]=y_grid.flatten()
    if increase_x:
        x_out = np.linspace(min(x),max(x),num=number)
    else:
        x_out = x
    if increase_y:
        if new_y_lims == None:
            y_out = np.linspace(min(y),max(y),num=number)
        else:
            y_out = np.linspace(new_y_lims[0],new_y_lims[1],num=number)
    else:
        y_out = y
    x_grid_output, y_grid_output = np.meshgrid(x_out,y_out)
    output_grid = np.ones([len(x_grid_output.flatten()),2],dtype=float)
    output_grid[:,0]=x_grid_output.flatten()
    output_grid[:,1]=y_grid_output.flatten()

    z_out = griddata(input_grid,z.flatten(),output_grid)
    z_out.resize(len(y_out),len(x_out))
    return x_out, y_out, z_out

def return_q_profile(mk,file_name='PROFEQ_PEST', n=2):
    '''
    SH : 19 Nov 2012
    read in q as a function of s (which is part of the output)
    (mq, sq) are the locations of the m=nq surfaces
    qn is the q value of the rational surface
    Not entirely sure how this function is working....
    '''
    #return the q profile
    dataq = np.loadtxt(open(file_name,'r'))
    s = dataq[:,0]
    q = dataq[:,1]
    mq = np.arange(np.ceil(np.min(q)*abs(n)),max(mk.flatten())+1)
    qq = mq/abs(n)

    def FindX(x,y,yy):
        xn = []
        yn = []

        for k in range(0,len(yy)):
            I = np.nonzero((y[0:-2]-yy[k])*(y[1:-1]-yy[k]) <= 0);

            for m in range(0,len(I)):
                J = I[m]
                if len(J)>=1:
                    xn.append(x[J] + (x[J+1]-x[J])*(yy[k]-y[J])/(y[J+1]-y[J]))
                    yn.append(yy[k])
        return np.array(xn),np.array(yn)

    sq, qn = FindX(s,q,qq)
    mq = qn*np.abs(n)

    return np.array(qn), np.array(sq), np.array(q), np.array(s),np.array(mq)




def pest_plot(directory, fig_name,title):
    os.chdir(directory)
    print directory
    Nchi = 513
    chi = np.linspace(np.pi*-1,np.pi,Nchi)
    chi.resize(1,len(chi))

    phi = np.linspace(0,2.*np.pi,Nchi)
    phi.resize(len(phi),1)

    file_name = 'RMZM_F_EQAC'
#    file_name = 'RMZM_F'

    RM, ZM, Ns,Ns1,Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
#    Nm2 = Nm0
#    R,Z =  GetRZ(RM,ZM,Nm0,Nm2,chi,phi)

#    file_name = 'BPLASMA'

#    FEEDI = get_FEEDI('log_mars')
#    BNORM = calc_BNORM(FEEDI, R0EXP)

#    BM1,BM2,BM3,Mm = ReadBPLASMA(file_name, BNORM, Ns)
#    dRds,dZds,dRdchi,dZdchi,jacobian = GetUnitVec(R, Z, s, chi)
#    B1,B2,B3,Bn,BMn = GetB123(BM1, BM2, BM3, R, Mm, chi, dRdchi, dZdchi)

    R,Z,B1,B2,B3,Bn,BMn = extract_data_temp(directory)
#    print np.sum(np.abs(R_upper-R))
#    print np.sum(np.abs(Z_upper-Z))
#    print np.sum(np.abs(B1_upper-B1))
#    print np.sum(np.abs(B2_upper-B2))
#    print np.sum(np.abs(B3_upper-B3))
#    print np.sum(np.abs(Bn_upper-Bn))
#    print np.sum(np.abs(BMn_upper-BMn))

    II=np.arange(1,Ns1+21,dtype=int)-1
    BnEQAC = Bn[II,:]
    R_EQAC = R[II,:]
    Z_EQAC = Z[II,:]

    Rs = R_EQAC[-1,:]
    Zs = Z_EQAC[-1,:]
    Rc = (np.min(Rs)+np.max(Rs))/2
    Zc = (np.min(Zs)+np.max(Zs))/2
    Tg = np.arctan2(Zs-Zc,Rs-Rc)
    BnEDGE = BnEQAC[-1,:]

    file_name = 'RMZM_F_PEST'
    RM, ZM, Ns,Ns1,Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
    Nm2 = Nm0
    R,Z =  GetRZ(RM,ZM,Nm0,Nm2,chi,phi)
    dRds,dZds,dRdchi,dZdchi,jacobian = GetUnitVec(R,Z, s, chi)

    ss     = s[II]

    R_PEST = R[II,:]
    Z_PEST = Z[II,:]

    G22_PEST  = dRdchi[II,:]**2 + dZdchi[II,:]**2

    G22_PEST[0,:] = G22_PEST[1,:]

    R_Z_EQAC = np.ones([len(R_EQAC.flatten()),2],dtype='float')
    R_Z_EQAC[:,0] = R_EQAC.flatten()
    R_Z_EQAC[:,1] = Z_EQAC.flatten()

    R_Z_PEST = np.ones([len(R_EQAC.flatten()),2],dtype='float')
    R_Z_PEST[:,0] = R_PEST.flatten()
    R_Z_PEST[:,1] = Z_PEST.flatten()

    BnPEST  = griddata(R_Z_EQAC,BnEQAC.flatten(),R_Z_PEST,method='linear')
    BnPEST.resize(BnEQAC.shape)
    BnPEST = BnPEST*np.sqrt(G22_PEST)*R_PEST 

    mk = np.arange(-29,29+1,dtype=int)
    mk.resize(1,len(mk))

    expmchi = np.exp(np.dot(-chi.transpose(),mk)*1j)
    BMnPEST = np.dot(BnPEST,expmchi)*(chi[0,1]-chi[0,0])/2./np.pi


    mm = np.arange(-29,29+1,dtype=int)
    mm2 = np.arange(-29,29+1,dtype=int)

    II = mm - mk[0,0] + 1 - 1
    II.resize(len(II),1)

    mk = mk[0,II]

    mk.resize(1,len(mk))
    facn = np.pi/2.

    BnPEST  = BMnPEST[:,II.flatten()]/facn

    BnPEST[0,:] = BnPEST[1,:]
    BnPEST[-1,:] = BnPEST[-2,:]


    fig = pt.figure()
    ax = fig.add_subplot(111) 
    II=np.arange(1,Ns1+1,dtype=int)
    ax.plot(s[II],np.real(BMn[II,:]))
    ax.set_ylabel('Re[B_n^{(m)}]')
    ax.set_xlabel('\psi_p^{1/2}')
#    fig.savefig('hello_pic1.png')
    #fig.canvas.draw()
    #fig.show()


    MM = np.arange(-15-Mm[0]+1,15+1+1-Mm[0],dtype=int)# - Mm[0] + 1;
    x = Mm[MM]
    y = s[II]
    z = abs(BMn[II,:][:,MM])


    file_name = 'PROFEQ_PEST'
    qn, sq, q, s, mq = return_q_profile(mk,file_name=file_name, n=2)

    x_out,y_out,z_out=increase_grid(x,y,z,number=100)

    fig = pt.figure()
    ax = fig.add_subplot(111) 
    ax.pcolor(x_out.flatten(),y_out.flatten(),z_out,cmap='hot')
    ax.set_xlim([min(x_out.flatten()),max(x_out.flatten())])
    ax.set_ylim([0,1])#[min(y_out.flatten()),max(y_out.flatten())])
#    fig.savefig('hello_pic2.png')
#    fig.canvas.draw()
#    fig.show()


    mk,ss,BnPEST=increase_grid(mk.flatten(),ss.flatten(),abs(BnPEST),number=200)

    fig = pt.figure()
    ax =fig.add_subplot(111)
    color_ax = ax.pcolor(mk,ss,abs(BnPEST),cmap='spectral')
    color_ax.set_clim([0,1])
    ax.plot(mq,sq,'wo')
    n=2
    ax.plot(q*n,s,'w--') 

    ax.set_title(title)
    ax.set_xlim([min(mk.flatten()),max(mk.flatten())])
    ax.set_ylim([0,1])#[min(ss.flatten()),max(ss.flatten())])
    pt.colorbar(color_ax)
    fig.savefig(fig_name, dpi=200 )
    #fig.canvas.draw()
    #fig.show()

    #  shading interpA
    #  contour(Mac.Mm(MM),Mac.s(II),abs(BMn(II,MM)),'k-'), hold on,
    #  axis([min(Mac.Mm(MM)) max(Mac.Mm(MM)) 0 1])
    #  xlabel('m','FontSize',16,'FontWeight','Bold')
    #  ylabel('sqrt(\psi_p)','FontSize',16,'FontWeight','Bold')
    #  title('|b_n| [Gauss/kA] in PEST CS','FontSize',16,'FontWeight','Bold')
    #  ha = get(10*Mac.plot_Bn+2,'CurrentAxes');
    #  set(ha,'FontSize',14,'FontWeight','Bold')
    #  colorbar, colormap(hot)


def extract_data_temp(directory):
    #Probably obsolete
    os.chdir(directory)
    Nchi = 513
    chi = np.linspace(np.pi*-1,np.pi,Nchi)
    chi.resize(1,len(chi))
    phi = np.linspace(0,2.*np.pi,Nchi)
    phi.resize(len(phi),1)

    file_name = 'RMZM_F'
    RM, ZM, Ns,Ns1,Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
    print 'R0EXP', R0EXP
    Nm2 = Nm0
    R,Z =  GetRZ(RM,ZM,Nm0,Nm2,chi,phi)
    FEEDI = get_FEEDI('FEEDI')
    BNORM = calc_BNORM(FEEDI, R0EXP)

    file_name = 'BPLASMA'
    BM1,BM2,BM3,Mm = ReadBPLASMA(file_name, BNORM, Ns)
    dRds,dZds,dRdchi,dZdchi,jacobian = GetUnitVec(R, Z, s, chi)
    B1,B2,B3,Bn,BMn = GetB123(BM1, BM2, BM3, R, Mm, chi, dRdchi, dZdchi)
    return R, Z, B1,B2,B3,Bn,BMn

def combine_data(directory_upper, directory_lower,phasing):
    R_upper,Z_upper,B1_upper,B2_upper,B3_upper,Bn_upper,BMn_upper = extract_data_temp(directory_upper)
    R_lower,Z_lower,B1_lower,B2_lower,B3_lower,Bn_lower,BMn_lower = extract_data_temp(directory_lower)
    phasing = phasing/180.*np.pi
    print '**********I-coil Phasing******************'
    print phasing
    print np.sum(np.abs(R_upper-R_lower)), np.max(np.abs(R_upper-R_lower))
    print np.sum(np.abs(Z_upper-Z_lower)), np.max(np.abs(Z_upper-Z_lower))
    print '****************************'
    phasor = (np.cos(phasing)+1j*np.sin(phasing))
    print '%.5f,%.5f'%(np.cos(phasing),np.sin(phasing))
    phasor = -0.5000 + 0.86603*1j
    B1 = np.array(B1_upper) + np.array(B1_lower)*phasor
    B2 = np.array(B2_upper) + np.array(B2_lower)*phasor
    B3 = np.array(B3_upper) + np.array(B3_lower)*phasor
    Bn = np.array(Bn_upper) + np.array(Bn_lower)*phasor
    BMn = np.array(BMn_upper) + np.array(BMn_lower)*phasor

    return R_upper, Z_upper, B1, B2, B3, Bn, BMn


def plot_error(R,Z,error):
    rw=[1,1.1999]
    file_name = 'RMZM_F'
    RM_tmp, ZM_tmp, Ns_tmp,Ns1_tmp,Ns2_tmp, Nm0_tmp, R0EXP_tmp, B0EXP_tmp, s_tmp = readRMZM(file_name)
    fig = pt.figure()
    ax = fig.add_subplot(111)
    color_ax = ax.pcolor(R[1:-2,:]*R0EXP_tmp,Z[1:-2,:]*R0EXP_tmp,error)
    II = np.zeros([1,len(rw)+1],dtype=int)
    for j in range(0,len(rw)):
        II[0,j]=round(np.argmin(np.abs(s_tmp - rw[j])))
        smin=np.min(np.abs(s_tmp - rw[j]))
    for j in range(0,len(rw)):
        ax.plot(R[II[0,j],:].flatten()*R0EXP_tmp,Z[II[0,j],:].flatten()*R0EXP_tmp,'k-')
    color_ax.set_clim([0,0.4])
    cbar = pt.colorbar(color_ax)
    cbar.set_label('Error %')
    ax.set_title('B1 Linear Superposition Error %')
    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')
    
    fig.canvas.draw()
    fig.show()

