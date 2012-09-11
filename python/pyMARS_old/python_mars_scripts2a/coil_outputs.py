import numpy as num
from PythonMARS_funcs import *
import scipy.interpolate as interpolate

dir = '/u/haskeysr/mars/templates/'

def coil_responses(r_array,z_array,Br,Bz,Bphi):
    probe  = [ '67A', '66M', '67B', 'ESL', 'ISL','UISL','LISL']
    # probe type 1: poloidal field, 2: radial field
    type   = num.array([     1,     1,     1,     0,     0,     0,     0])
    # Poloidal geometry
    Rprobe = num.array([ 2.265, 2.413, 2.265, 2.477, 2.431, 2.300, 2.300])
    Zprobe = num.array([ 0.755,   0.0,-0.755, 0.000, 0.000, 0.714,-0.714])
    tprobe = num.array([ -67.5, -90.0,-112.5, 0.000, 0.000,  22.6, -22.6])*2*num.pi/360  #DTOR # poloidal inclination
    lprobe = num.array([ 0.155, 0.140, 0.155, 1.194, 0.800, 0.680, 0.680])  # Length of probe
    Nprobe = len(probe)

    Navg = 41    # points along probe to interpolate
    Bprobem = [] # final output
    for k in range(0, Nprobe):
        #depending on poloidal/radial
        if type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*num.sin(tprobe[k])*num.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*num.cos(tprobe[k])*num.linspace(-1,1,num = Navg)

        r_filt = [];z_filt = [];Br_filt = [];Bz_filt = []
        Rprobek_min = num.min(Rprobek)
        Rprobek_max = num.max(Rprobek)
        Zprobek_min = num.min(Zprobek)
        Zprobek_max = num.max(Zprobek)

        #search box for interpolation to minimise computation
        Rprobek_min = Rprobek_min - num.abs(Rprobek_min)*0.05
        Rprobek_max = Rprobek_max + num.abs(Rprobek_max)*0.05
        Zprobek_min = Zprobek_min - num.abs(Zprobek_min)*0.05
        Zprobek_max = Rprobek_max + num.abs(Zprobek_max)*0.05

        #whittle down to poindts near the coil
        for iii in range(0,r_array.shape[0]):
            for jjj in range(0,r_array.shape[1]):
                if (((Rprobek_min <= r_array[iii,jjj]) and (r_array[iii,jjj] <= Rprobek_max)) and ((Zprobek_min <= z_array[iii,jjj]) and (z_array[iii,jjj] <= Zprobek_max))):
                    r_filt.append(r_array[iii,jjj])
                    z_filt.append(z_array[iii,jjj])
                    Br_filt.append(Br[iii,jjj])
                    Bz_filt.append(Bz[iii,jjj])

        r_filt_array = num.array(r_filt)
        z_filt_array = num.array(z_filt)
        Br_filt_array = num.array(Br_filt)
        Bz_filt_array = num.array(Bz_filt)

        print r_filt_array.shape
        print z_filt_array.shape
        print Br_filt_array.shape

        #Create interpolation functions
        newfuncBrr = interpolate.Rbf(r_filt_array, z_filt_array, num.real(Br_filt_array), function='linear') 
        newfuncBri = interpolate.Rbf(r_filt_array, z_filt_array, num.imag(Br_filt_array),function='linear')

        newfuncBzr = interpolate.Rbf(r_filt_array, z_filt_array, num.real(Bz_filt_array), function='linear')
        newfuncBzi = interpolate.Rbf(r_filt_array, z_filt_array, num.imag(Bz_filt_array), function='linear')

        #Create interpolated values
        Brprobek = newfuncBrr(Rprobek,Zprobek) + newfuncBri(Rprobek,Zprobek)*1j
        Bzprobek = newfuncBzr(Rprobek,Zprobek) + newfuncBzi(Rprobek,Zprobek)*1j

        #Find perpendicular components
        Bprobek  =  (num.sin(tprobe[k])*num.real(Bzprobek) + num.cos(tprobe[k])*num.real(Brprobek)) + 1j * (num.sin(tprobe[k])*num.imag(Bzprobek) +num.cos(tprobe[k])*num.imag(Brprobek))

        Bprobem.append(num.average(Bprobek))
    return Bprobem

r_array, z_array = post_mars_r_z(dir)
Br = extractB(dir,'Br')
Bz = extractB(dir,'Bz')
Bphi = extractB(dir,'Bphi')


print 'new data'
Answers = coil_responses(r_array,z_array,Br,Bz,Bphi)
print 'finished'
print len(Answers)
print Answers
