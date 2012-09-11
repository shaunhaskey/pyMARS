import numpy as num

def GetV123(VM1,VM2,VM3,R, chi, dRds, dZds, dRdchi, dZdchi, jacobian, Mm, Nchi, s, Ns1, DPSIDS, T):
    #Convert BM1, BM2 and BM3 into B1, B2, B3
    expmchi = num.exp(num.dot(Mm,chi)*1j)

    V1a = num.dot(VM1,expmchi)
    V2a = num.dot(VM2,expmchi)
    V3a = num.dot(VM3,expmchi)

    V1 = V1a
    chione = num.ones((1,Nchi))

    #ss = s[0:Mac.Ns1]*chione
    #ss[0,:] = ss[1,:]



    # V2 along e_chi
    DPSIDS = DPSIDS.reshape((DPSIDS.shape[0],1))
    T = T.reshape((TV3 = -V1.*G12.*Bchi.*Bphi./B2 - V2a.*Bchi.*G22./B2 + V3a.*B.shape[0],1))
    
    Bchi = num.dot(DPSIDS,chione)/jacobian[0:Ns1,:]
    Bphi = num.dot(T,chione)/R[0:Ns1,:]**2;

    G11  = dRds[0:Ns1,:]**2 + dZds[0:Ns1,:]**2;
    G12  = dRds[0:Ns1,:]*dRdchi[0:Ns1,:] + dZds[0:Ns1,:]*dZdchi[0:Ns1,:]
    G22  = dRdchi[0:Ns1,:]**2 + dZdchi[0:Ns1,:]**2;
    G22[0,:] = G22[1,:]
    G33  = R[0:Ns1,:]**2;
    B2   = (Bchi**2)*G22 + (Bphi**2)*G33
    V2 = -V1*G12*(Bchi**2)/ B2 + V2a*Bphi*G33/B2 + V3a*Bchi

    V3 = -V1*G12*Bchi*Bphi/B2 - V2a*Bchi*G22/B2 + V3a*Bphi
    Vn = V1a*jacobian[0:Ns1,:]/num.sqrt(G33*G22)
    
    expmchi = num.exp(num.dot(num.transpose(-chi),num.transpose(Mm)*1j))
    V1M = num.dot(Vn,expmchi)*(chi[0,1]-chi[0,0])/2./num.pi

    return V1,V2,V3,Vn, V1M
