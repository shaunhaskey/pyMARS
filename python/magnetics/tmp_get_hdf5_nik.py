import h5py
import numpy as num

def get_hdf5values(f, sensor_name, coil_name, nd, debug=0):
    '''
    return sz, sp, sk (zero, pole, gain) between sensor_name and coil_name
    f is a h5py object containing the couplings
    
    I'm not sure what nd is about - this came from Jeremy
    '''
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


single_transfers = '/home/srh112/NAMP_datafiles/tf2012_single.h5'
vac_coupling = h5py.File(single_transfers, 'r')
sensor_coil = 'MPI66M157'
i_coil = 'IU270'
nd = 16
sk, sp, sz, Afit, Bfit, np, nz = get_hdf5values(vac_coupling, sensor_coil, i_coil, nd, debug=0)
