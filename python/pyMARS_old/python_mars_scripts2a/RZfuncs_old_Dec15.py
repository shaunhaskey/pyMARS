import numpy as num
import scipy.interpolate as interp
import matplotlib.pyplot as pt
import matplotlib.mlab as mlab
from scipy.interpolate import *
import os


def MacGetBphysC(R,Z,dRds,dZds,dRdchi,dZdchi,jacobian,B1,B2,B3):
    Br = (B1*dRds + B2*dRdchi)/jacobian
    Br[0,:] = Br[1,:]
    Bz = (B1*dZds + B2*dZdchi)/jacobian
    Bz[0,:] = Bz[1,:]
    Bphi = B3*R/jacobian
    Bphi[0,:] = Bphi[1,:]
    return Br,Bz,Bphi
    

def readRMZM(file_name):
    RMZM = num.loadtxt(open(file_name))
    Nm0 = num.round(RMZM[0,0])
    Ns1 = num.round(RMZM[0,1])
    Ns2 = num.round(RMZM[0,2])

    R0EXP = RMZM[0,3]
    B0EXP = RMZM[1,3]

    Ns = Ns1 + Ns2

    s = num.array(RMZM[1:Ns+1,0])
    RM = RMZM[Ns+1:,0] + RMZM[Ns+1:,1]*1j
    ZM = RMZM[Ns+1:,2] + RMZM[Ns+1:,3]*1j

    RM = num.reshape(RM,[Ns,Nm0],order='F')
    ZM = num.reshape(ZM,[Ns,Nm0],order='F')

    RM[:,1:] = 2.*RM[:,1:]
    ZM[:,1:] = 2.*ZM[:,1:]

    return RM, ZM, Ns, Ns1, Ns2, Nm0, R0EXP, B0EXP, s

def GetRZ(RM,ZM,Nm0,Nm2,chi,phi):
    Nm2 = Nm0
    m = num.arange(0,Nm2,1)
    m.resize(len(m),1)

    expmchi = num.exp(num.dot(m,chi)*1j)
    R = num.real(num.dot(RM[:,0:Nm2],expmchi))
    Z = num.real(num.dot(ZM[:,0:Nm2],expmchi))
    return R,Z

def ReadBPLASMA(file_name,BNORM,Ns):
    BPLASMA = num.loadtxt(open(file_name))
 
    Nm1 = BPLASMA[0,0]
    n = num.round(BPLASMA[0,2])
    Mm = num.round(BPLASMA[1:Nm1+1,0])
    Mm.resize([len(Mm),1])


    BM1 = BPLASMA[Nm1+1:,0] + BPLASMA[Nm1+1:,1]*1j
    BM2 = BPLASMA[Nm1+1:,2] + BPLASMA[Nm1+1:,3]*1j
    BM3 = BPLASMA[Nm1+1:,4] + BPLASMA[Nm1+1:,5]*1j

    BM1 = num.reshape(BM1,[Ns,Nm1],order='F')
    BM2 = num.reshape(BM2,[Ns,Nm1],order='F')
    BM3 = num.reshape(BM3,[Ns,Nm1],order='F')

    BM1 = BM1[0:Ns,:]*BNORM
    BM2 = BM2[0:Ns,:]*BNORM
    BM3 = BM3[0:Ns,:]*BNORM

    #NEED TO KNOW WHY THIS SECTION IS INCLUDED - to do with half grid???!!
    BM2[1:,:] = BM2[0:-1,:]
    BM3[1:,:] = BM3[0:-1,:]

    return BM1, BM2, BM3,Mm

def GetB123(BM1,BM2,BM3,R, Mm, chi, dRdchi, dZdchi):

    expmchi = num.exp(num.dot(Mm,chi)*1j)

    B1 = num.dot(BM1,expmchi)
    B2 = num.dot(BM2,expmchi)
    B3 = num.dot(BM3,expmchi)
    G22  = dRdchi**2 + dZdchi**2
    G22[0,:] = G22[1,:]
    Bn   = B1/num.sqrt(G22)/R
    expmchi = num.exp(num.dot(-(chi.transpose()), Mm.transpose()*1j))
    BMn = num.dot(Bn,expmchi)*(chi[0,1]-chi[0,0])/2/num.pi

    return B1,B2,B3,Bn, BMn

def GetUnitVec(R,Z,s,chi):

    s0 = num.resize(s,[1,len(s)])
    R0 = R
    chi0 = chi
    Z0 = Z
    hs = num.min(s0[0,1:]-s0[0,0:-1])/2.
    hs = num.min([hs,1e-4])
    hchi = num.min(chi0[0,1:]-chi0[0,0:-1])/2
    hchi = num.min([hchi,1e-4])
    s1 = s0 - hs
    s2 = s0 + hs
    chi1 = chi0 - hchi
    chi2 = chi0 + hchi

    R_func = interp.interp1d(s0.flatten(), R0.transpose(), kind='cubic', bounds_error = 0)
    R1 = R_func(s1.flatten())
    R2 = R_func(s2.flatten())
    dRds = num.transpose((R2-R1)/hs/2)

    # compute dZ/ds using Z(s,chi) and spline
    Z_func = interp.interp1d(s0.flatten(), Z0.transpose(), kind='cubic', bounds_error = 0)
    Z1 = Z_func(s1.flatten())
    Z2 = Z_func(s2.flatten())
    dZds = num.transpose((Z2-Z1)/hs/2)


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

def get_FEEDI(file_name):
    file = open(file_name,'r')
    input_string = file.read()
    file.close()
    success = 0
    variable_name = 'ABS'
    name_start_location = input_string.find(variable_name)
    end_location = input_string[name_start_location:].find('\n')
    imp_line = input_string[name_start_location:name_start_location+end_location]
    FEEDI_string = imp_line.strip('ABS(FDI(K))= ')
    FEEDI_float = float(FEEDI_string)
    print imp_line, FEEDI_float
    return FEEDI_float

def calc_BNORM(FEEDI, R0EXP):
    phas = 0.5*num.pi
    phas = 0 #is this correct?
    BNORM = 1.0*num.exp(phas*1j)
    I0EXP = 1.0e+3 #* 3./num.pi
    mu0   = 4.e-7*num.pi
    fac   = mu0/R0EXP;
    #FEEDI = 1.0;
    #FEEDI = temp_ans
    BNORM  = BNORM*fac*I0EXP/FEEDI*1e+4
    return BNORM

def increase_grid(x, y, z, number=100):
    x_grid,y_grid = num.meshgrid(x,y)
    input_grid = num.ones([len(x_grid.flatten()),2],dtype=float)
    input_grid[:,0]=x_grid.flatten()
    input_grid[:,1]=y_grid.flatten()

    x_out = num.linspace(min(x),max(x),num=number)
    y_out = y
    x_grid_output, y_grid_output = num.meshgrid(x_out,y_out)
    output_grid = num.ones([len(x_grid_output.flatten()),2],dtype=float)
    output_grid[:,0]=x_grid_output.flatten()
    output_grid[:,1]=y_grid_output.flatten()

    z_out = griddata(input_grid,z.flatten(),output_grid)
    z_out.resize(len(y_out),len(x_out))
    return x_out, y_out, z_out

def return_q_profile(mk,file_name='PROFEQ_PEST', n=2):
    dataq = num.loadtxt(open(file_name,'r'))
    s = dataq[:,0]
    q = dataq[:,1]
    mq = num.arange(num.ceil(num.min(q)*abs(n)),max(mk.flatten())+1)
    qq = mq/abs(n)

    def FindX(x,y,yy):
        xn = []
        yn = []

        for k in range(0,len(yy)):
            I = num.nonzero((y[0:-2]-yy[k])*(y[1:-1]-yy[k]) <= 0);

            for m in range(0,len(I)):
                J = I[m]
                if len(J)>=1:
                    xn.append(x[J] + (x[J+1]-x[J])*(yy[k]-y[J])/(y[J+1]-y[J]))
                    yn.append(yy[k])
        return num.array(xn),num.array(yn)

    sq,qn = FindX(s,q,qq)
    mq = qn*num.abs(n)

    return num.array(qn), num.array(sq), num.array(q), num.array(s),num.array(mq)




def pest_plot(directory, fig_name,title):
    os.chdir(directory)
    print directory
    Nchi = 513
    chi = num.linspace(num.pi*-1,num.pi,Nchi)
    chi.resize(1,len(chi))

    phi = num.linspace(0,2.*num.pi,Nchi)
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
#    print num.sum(num.abs(R_upper-R))
#    print num.sum(num.abs(Z_upper-Z))
#    print num.sum(num.abs(B1_upper-B1))
#    print num.sum(num.abs(B2_upper-B2))
#    print num.sum(num.abs(B3_upper-B3))
#    print num.sum(num.abs(Bn_upper-Bn))
#    print num.sum(num.abs(BMn_upper-BMn))

    II=num.arange(1,Ns1+21,dtype=int)-1
    BnEQAC = Bn[II,:]
    R_EQAC = R[II,:]
    Z_EQAC = Z[II,:]

    Rs = R_EQAC[-1,:]
    Zs = Z_EQAC[-1,:]
    Rc = (num.min(Rs)+num.max(Rs))/2
    Zc = (num.min(Zs)+num.max(Zs))/2
    Tg = num.arctan2(Zs-Zc,Rs-Rc)
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

    expmchi = num.exp(num.dot(-chi.transpose(),mk)*1j)
    BMnPEST = num.dot(BnPEST,expmchi)*(chi[0,1]-chi[0,0])/2./num.pi


    mm = num.arange(-29,29+1,dtype=int)
    mm2 = num.arange(-29,29+1,dtype=int)

    II = mm - mk[0,0] + 1 - 1
    II.resize(len(II),1)

    mk = mk[0,II]

    mk.resize(1,len(mk))
    facn = num.pi/2.

    BnPEST  = BMnPEST[:,II.flatten()]/facn

    BnPEST[0,:] = BnPEST[1,:]
    BnPEST[-1,:] = BnPEST[-2,:]


    fig = pt.figure()
    ax = fig.add_subplot(111) 
    II=num.arange(1,Ns1+1,dtype=int)
    ax.plot(s[II],num.real(BMn[II,:]))
    ax.set_ylabel('Re[B_n^{(m)}]')
    ax.set_xlabel('\psi_p^{1/2}')
#    fig.savefig('hello_pic1.png')
    #fig.canvas.draw()
    #fig.show()


    MM = num.arange(-15-Mm[0]+1,15+1+1-Mm[0],dtype=int)# - Mm[0] + 1;
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
    os.chdir(directory)
    Nchi = 513
    chi = num.linspace(num.pi*-1,num.pi,Nchi)
    chi.resize(1,len(chi))
    phi = num.linspace(0,2.*num.pi,Nchi)
    phi.resize(len(phi),1)

    file_name = 'RMZM_F'
    RM, ZM, Ns,Ns1,Ns2, Nm0, R0EXP, B0EXP, s = readRMZM(file_name)
    print 'R0EXP', R0EXP
    Nm2 = Nm0
    R,Z =  GetRZ(RM,ZM,Nm0,Nm2,chi,phi)
    FEEDI = get_FEEDI('log_mars')
    BNORM = calc_BNORM(FEEDI, R0EXP)

    file_name = 'BPLASMA'
    BM1,BM2,BM3,Mm = ReadBPLASMA(file_name, BNORM, Ns)
    dRds,dZds,dRdchi,dZdchi,jacobian = GetUnitVec(R, Z, s, chi)
    B1,B2,B3,Bn,BMn = GetB123(BM1, BM2, BM3, R, Mm, chi, dRdchi, dZdchi)
    return R, Z, B1,B2,B3,Bn,BMn

def combine_data(directory_upper, directory_lower,phasing):
    R_upper,Z_upper,B1_upper,B2_upper,B3_upper,Bn_upper,BMn_upper = extract_data_temp(directory_upper)
    R_lower,Z_lower,B1_lower,B2_lower,B3_lower,Bn_lower,BMn_lower = extract_data_temp(directory_lower)
    phasing = phasing/180.*num.pi
    print '**********I-coil Phasing******************'
    print phasing
    print num.sum(num.abs(R_upper-R_lower)), num.max(num.abs(R_upper-R_lower))
    print num.sum(num.abs(Z_upper-Z_lower)), num.max(num.abs(Z_upper-Z_lower))
    print '****************************'
    phasor = (num.cos(phasing)+1j*num.sin(phasing))
    print '%.5f,%.5f'%(num.cos(phasing),num.sin(phasing))
    phasor = -0.5000 + 0.86603*1j
    B1 = num.array(B1_upper) + num.array(B1_lower)*phasor
    B2 = num.array(B2_upper) + num.array(B2_lower)*phasor
    B3 = num.array(B3_upper) + num.array(B3_lower)*phasor
    Bn = num.array(Bn_upper) + num.array(Bn_lower)*phasor
    BMn = num.array(BMn_upper) + num.array(BMn_lower)*phasor

    return R_upper, Z_upper, B1, B2, B3, Bn, BMn


def plot_error(R,Z,error):
    rw=[1,1.1999]
    file_name = 'RMZM_F'
    RM_tmp, ZM_tmp, Ns_tmp,Ns1_tmp,Ns2_tmp, Nm0_tmp, R0EXP_tmp, B0EXP_tmp, s_tmp = readRMZM(file_name)
    fig = pt.figure()
    ax = fig.add_subplot(111)
    color_ax = ax.pcolor(R[1:-2,:]*R0EXP_tmp,Z[1:-2,:]*R0EXP_tmp,error)
    II = num.zeros([1,len(rw)+1],dtype=int)
    for j in range(0,len(rw)):
        II[0,j]=round(num.argmin(num.abs(s_tmp - rw[j])))
        smin=num.min(num.abs(s_tmp - rw[j]))
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

