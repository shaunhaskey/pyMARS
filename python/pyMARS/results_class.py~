import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interp
import matplotlib.mlab as mlab
from scipy.interpolate import *
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import os, copy,time
from pyMARS.RZfuncs import *
import pyMARS.RZfuncs
import matplotlib


def combine_fields_displacement(input_data, attr_name, theta = 0, field_type='plas'):
    '''Used for combining the displacement fields
    Should replace with a single phasing function to reduce duplication!

    SRH: 31Jan2014
    '''
    print 'combining property : ', attr_name
    if len(input_data)==2:
        if field_type=='plas':
            answer = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
        elif field_type == 'total':
            answer = getattr(input_data[0], attr_name)
        elif field_type == 'vac':
            answer = getattr(input_data[1], attr_name)
        print 'already combined, ', field_type
    elif len(input_data) == 4:
        if field_type=='plas':
            lower_data = getattr(input_data[0], attr_name) - getattr(input_data[1], attr_name)
            upper_data = getattr(input_data[2], attr_name) - getattr(input_data[3], attr_name)
        elif field_type=='total':
            lower_data = getattr(input_data[0], attr_name)
            upper_data = getattr(input_data[2], attr_name)
        elif field_type=='vac':
            lower_data = getattr(input_data[1], attr_name)
            upper_data = getattr(input_data[3], attr_name)
        answer = upper_data + lower_data*(np.cos(theta)+1j*np.sin(theta))
        print 'combine %s, theta : %.2f'%(field_type, theta)
    return answer

def disp_calcs(run_data, n_zones = 20, phasing_vals = None, ul= True):
    '''This function finds the average sum of displacement amplitudes between certain values to check for x-point peaking etc...
    It returns the regions on the LFS, HFS, above and below the midplane.

    run_data is the list of input data results classes [lower_tot,lower_vac,upper_tot,upper_vac]
    n_zones is the number of regions a measurement is found for
    phasing_vals is the list of phasings the calculations are performed for
    ul - not really implemented yet, but says if its upper lower data
    SRH : 31Jan2014
    '''
    plot_field = 'Vn'; field_type = 'total'
    #run_data = extract_data(base_dir, I0EXP, ul=ul, Nchi=Nchi, get_VPLASMA=1, plas_vac = False)

    output_dict = {}
    if phasing_vals == None: phasing_vals = [0]
    grid_r = run_data[0].R*run_data[0].R0EXP
    grid_z = run_data[0].Z*run_data[0].R0EXP

    #remove grid points outside the plasma
    plas_r = grid_r[0:run_data[0].Vn.shape[0],:]
    plas_z = grid_z[0:run_data[0].Vn.shape[0],:]

    upper_values = np.linspace(0,np.max(plas_z[-1,:]),n_zones/2,endpoint = True)
    lower_values = np.linspace(0,np.min(plas_z[-1,:]),n_zones/2,endpoint = True)
    output_dict['upper_values'] = upper_values
    output_dict['lower_values'] = lower_values

    r_vals = plas_r[-1,:]
    z_vals = plas_z[-1,:]
    dz = np.diff(z_vals)
    dl = np.sqrt(np.diff(z_vals)**2 + np.diff(r_vals)**2)
    angle = np.arctan2(plas_z[-1,:], plas_r[-1,:]-run_data[0].R0EXP)
    z_vals_red = z_vals[:-1]
    r_vals_red = r_vals[:-1]

    for theta_deg in phasing_vals:
        cur_dict = {}
        for side in ['HFS','LFS']:
            for ab in ['above','below']:
                cur_dict['disp_{}_{}'.format(ab,side)] = []
                cur_dict['ang_{}_{}'.format(ab,side)] = []
        print '===== %d ====='%(theta_deg)
        theta = float(theta_deg)/180*np.pi;
        plot_quantity = combine_fields_displacement(run_data, plot_field, theta=theta, field_type=field_type)
        plot_quantity_red = plot_quantity[-1,:-1]
        for i in range(1,len(upper_values)):
            truth = (z_vals_red>=upper_values[i-1])*(z_vals_red<upper_values[i])*(dz<0)
            cur_dict['disp_above_HFS'].append(np.sum(np.abs(plot_quantity_red[truth]))/np.sum(truth))
            cur_dict['ang_above_HFS'].append(np.mean(angle[truth]))
            #ax.plot(r_vals_red[truth1], z_vals_red[truth1],'--')
            truth = (z_vals_red>=upper_values[i-1])*(z_vals_red<upper_values[i])*(dz>0)
            cur_dict['disp_above_LFS'].append(np.sum(np.abs(plot_quantity_red[truth]))/np.sum(truth))
            cur_dict['ang_above_LFS'].append(np.mean(angle[truth]))
            #ax.plot(r_vals_red[truth2], z_vals_red[truth2],'-')
        for i in range(1,len(lower_values)):
            truth = (z_vals_red<lower_values[i-1])*(z_vals_red>=lower_values[i])*(dz<0)
            #print upper_values[i-1], upper_values[i], np.sum(truth)
            cur_dict['disp_below_HFS'].append(np.sum(np.abs(plot_quantity_red[truth]))/np.sum(truth))
            cur_dict['ang_below_HFS'].append(np.mean(angle[truth]))
            #ax.plot(r_vals_red[truth1], z_vals_red[truth1],'--')
            truth = (z_vals_red<lower_values[i-1])*(z_vals_red>=lower_values[i])*(dz>0)
            cur_dict['disp_below_LFS'].append(np.sum(np.abs(plot_quantity_red[truth]))/np.sum(truth))
            cur_dict['ang_below_LFS'].append(np.mean(angle[truth]))
            #ax.plot(r_vals_red[truth2], z_vals_red[truth],'-')
        output_dict[theta_deg]=copy.deepcopy(cur_dict)
    print output_dict[0]['disp_below_HFS'], output_dict[45]['disp_below_HFS']
    return output_dict




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

def extract_plotk_results():
    file_handle = file('plotk/bnisldo','r')
    file_lines = file_handle.readlines()
    output_dict = {}
    output_dict['REAL']={}
    output_dict['IMAG']={}
    for i in range(3,len(file_lines)):
        current_line = file_lines[i]
        #print current_line
        if current_line=='\n':
            pass
            #print i, 'blank'
        elif current_line.find('REAL')!=(-1):
            component = 'REAL'
            mode = int(current_line.split('=')[1])
            output_dict[component][mode] = []
            #print component, mode
        elif current_line.find('IMAG')!=(-1):
            component = 'IMAG'
            mode = int(current_line.split('=')[1])
            output_dict[component][mode] = []
            #print component, mode
        else:
            line_list = filter(None,current_line.split(' '))
            for j in line_list:
                output_dict[component][mode].append(float(j))
    m = np.array(output_dict['REAL'].keys())
    m.sort()
    start = 1
    mode_amps = np.ones((m.shape[0], len(output_dict['REAL'][m[0]])),dtype='complex')
    for i in range(0,m.shape[0]):
        mode_amps[i,:] = (np.array(output_dict['REAL'][m[i]])+1j*(np.array(output_dict['IMAG'][m[i]])))
    return m, mode_amps


class data():
    def __init__(self, directory,Nchi=513,link_RMZM=1, I0EXP=1.0e+3 * 3./np.pi, spline_B23=2, getpest = False):
        self.directory = copy.deepcopy(directory)
        self.Nchi = copy.deepcopy(Nchi)
        #self.link_RMZM = copy.deepcopy(link_RMZM)
        self.I0EXP = copy.deepcopy(I0EXP)
        self.spline_B23 = spline_B23
        self.extract_single()
        if getpest: self.get_PEST(facn =1 )
        print '-----',  os.getcwd(), os.getpid(), np.sum(np.abs((self.Bn))), np.sum(np.abs((self.BnPEST)))

    def extract_single(self):
        #os.chdir(self.directory) 
        self.chi = np.linspace(np.pi*-1,np.pi,self.Nchi)
        self.chi.resize(1,len(self.chi))
        self.phi = np.linspace(0,2.*np.pi,self.Nchi)
        self.phi.resize(len(self.phi),1)

        file_name = self.directory + '/../../cheaserun/RMZM_F'
        self.RM, self.ZM, self.Ns, self.Ns1, self.Ns2, self.Nm0, self.R0EXP, self.B0EXP, self.s = readRMZM(file_name)
        print self.Ns, self.Ns1, self.Ns2
        #print 'R0EXP', self.R0EXP
        self.Nm2 = self.Nm0
        self.R, self.Z =  GetRZ(self.RM,self.ZM,self.Nm0,self.Nm2,self.chi,self.phi)
        self.FEEDI = get_FEEDI(self.directory + '/FEEDI')
        self.BNORM = calc_BNORM(self.FEEDI, self.R0EXP, I0EXP = self.I0EXP)
        print self.directory, self.BNORM, self.FEEDI, self.R0EXP, self.I0EXP
        file_name = self.directory + '/BPLASMA'
        #Extract geometry related stuff
        self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian = GetUnitVec(self.R, self.Z, self.s, self.chi)

        #Extract all the magnetic field stuff
        self.BM1, self.BM2, self.BM3, self.Mm = ReadBPLASMA(file_name, self.BNORM, self.Ns, self.s, spline_B23=self.spline_B23)
        self.B1,self.B2,self.B3,self.Bn,self.BMn = GetB123(self.BM1, self.BM2, self.BM3, self.R, self.Mm, self.chi, self.dRdchi, self.dZdchi)
        self.Br,self.Bz,self.Bphi = MacGetBphysC(self.R,self.Z,self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian,self.B1,self.B2,self.B3)
        self.Brho,self.Bchi,self.Bphi2 = MacGetBphysT(self.R,self.Z,self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian,self.B1,self.B2,self.B3,self.B0EXP)
        self.NW = int(round(float(pyMARS_funcs.extract_value(self.directory + '/../../cheaserun/log_chease','NW',' '))))

    def get_VPLASMA(self, VNORM=1.0):
        #os.chdir(self.directory)
        self.VNORM = calc_VNORM(self.FEEDI, self.B0EXP, I0EXP = self.I0EXP)
        self.VM1, self.VM2, self.VM3, self.DPSIDS, self.T = ReadVPLASMA(self.directory + '/VPLASMA',self.Ns, self.Ns1, self.s, VNORM=self.VNORM)
        self.V1,self.V2,self.V3,self.Vn, self.V1m = GetV123(self.VM1,self.VM2,self.VM3,self.R, self.chi, self.dRds, self.dZds, self.dRdchi, self.dZdchi, self.jacobian, self.Mm, self.Nchi, self.s, self.Ns1, self.DPSIDS, self.T)
        self.Vr, self.Vz, self.Vphi = MacGetVphys(self.R,self.Z,self.dRds,self.dZds,self.dRdchi,self.dZdchi,self.jacobian,self.V1,self.V2,self.V3, self.Ns1)


    def plot_Bn_surface(self,ax = None, surfaces = None, multiplier = 0.0074):
        self.plot_Vn_surface(ax = ax, surfaces = surfaces, multiplier = multiplier, plot_attribute = 'Bn')
        print 'Bn'

    def plot_Vn_surface(self,ax = None, surfaces = None, multiplier = 10, plot_attribute = 'Vn'):
        if surfaces == None: surfaces = [-1]
        grid_r = self.R*self.R0EXP
        grid_z = self.Z*self.R0EXP
        plas_r = grid_r[0:self.Vn.shape[0],:]
        plas_z = grid_z[0:self.Vn.shape[0],:]
        no_ax = True if (ax == None) else False
        if no_ax: fig,ax = pt.subplots(ncols = 2, sharex = True, sharey = True)
        for i in surfaces: 
            ax.plot(plas_r[i,:],plas_z[i,:],'-')
            norm_z = np.diff(plas_r[i,:])
            norm_r = -np.diff(plas_z[i,:])
            dl = np.sqrt(norm_r**2+norm_z**2)
            norm_r/=dl
            norm_z/=dl
            disp_quant = (getattr(self, plot_attribute))[0:self.Vn.shape[0],:]
            disp_quant = disp_quant[i,:]
            print 'mean abs {:.4e}'.format(np.mean(np.abs(disp_quant)))
            ax.plot(plas_r[i,:-1]+multiplier*norm_r*np.real(disp_quant[:-1]),plas_z[i,:-1]+multiplier*norm_z*np.real(disp_quant[:-1]))
            ax.plot(plas_r[i,:-1]+multiplier*norm_r*np.imag(disp_quant[:-1]),plas_z[i,:-1]+multiplier*norm_z*np.imag(disp_quant[:-1]))

            #angle = np.arctan2(plas_z[i,:], plas_r[i,:]-run_data[0].R0EXP)
            #angle = np.rad2deg(angle)
            #ax2.plot(angle, disp_quant)
        if no_ax: fig.canvas.draw(); fig.show()

    
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
        #os.chdir(self.directory) 
        #os.system('ln -f -s ../../cheaserun_PEST/RMZM_F RMZM_F_PEST')
        II=np.arange(1,self.Ns1+21,dtype=int)-1
        BnEQAC = copy.deepcopy(self.Bn[II,:])
        R_EQAC = copy.deepcopy(self.R[II,:])
        Z_EQAC = copy.deepcopy(self.Z[II,:])
        
        print '--- in pest calc eqac orig',  os.getcwd(), np.sum(np.abs((BnEQAC)))
        Rs = copy.deepcopy(R_EQAC[-1,:])
        Zs = copy.deepcopy(Z_EQAC[-1,:])
        Rc = (np.min(Rs)+np.max(Rs))/2.
        Zc = (np.min(Zs)+np.max(Zs))/2.
        Tg = np.arctan2(Zs-Zc,Rs-Rc)
        BnEDGE = copy.deepcopy(BnEQAC[-1,:])

        file_name = self.directory +'/../../cheaserun_PEST/RMZM_F'
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
        #this section is to make it work inthe griddata function
        print '--- in pest calc eqac',  os.getpid(), os.getcwd(), np.sum(np.abs((BnEQAC)))
        import scipy.interpolate as interp
        BnPEST  = interp.griddata(R_Z_EQAC,BnEQAC.flatten(),R_Z_PEST,method='linear')
        print '--- in pest calc nan', os.getpid(), np.sum(np.isnan(R_Z_EQAC)), np.sum(np.isnan(R_Z_PEST))
        print '--- in pest calc eqac*',  os.getpid(), os.getcwd(), np.sum(np.abs((BnPEST))), np.sum(np.isnan((BnPEST))), np.sum(np.isnan((BnPEST))) >0

        #if_pass = 'fail' if np.sum(np.isnan((BnPEST))) > 0 else 'pass'
        #np.savetxt('{}_RZPEST_{}'.format(os.getpid(), if_pass), R_Z_PEST)
        #np.savetxt('{}_BnPEST_{}'.format(os.getpid(), if_pass), BnPEST)
        #np.savetxt('{}_RZEQAC_{}'.format(os.getpid(), if_pass), R_Z_EQAC)
        BnPEST.resize(BnEQAC.shape)
        BnPEST = BnPEST*np.sqrt(G22_PEST)*R_PEST
        print '--- in pest calc eqac**',  os.getpid(), os.getcwd(), np.sum(np.abs((BnPEST)))


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
        print '--- in pest calc',  os.getcwd(), os.getcwd(), np.sum(np.abs((self.Bn))), np.sum(np.abs((self.BnPEST)))

        new_area = []
        for i in range(0, self.R.shape[0]):
            dR = (self.R[i,1:]-self.R[i,:-1])#*self.R0EXP
            dZ = (self.Z[i,1:]-self.Z[i,:-1])#*self.R0EXP
            R_ave = (self.R[i,1:]+self.R[i,:-1])/2.#*self.R0EXP
            new_area.append(np.sum(np.sqrt(dR**2+dZ**2)*R_ave*2*np.pi))
            #self.new_area.append(np.sum(np.sqrt(dR**2+dZ**2))*np.pi*2.)
        if new_area[0]==0:new_area[0]=new_area[1] #so that nothing blows up if dividing by it
        self.A = np.array(new_area)
        tmp = self.A[:self.BnPEST.shape[0]]
        tmp.resize((self.BnPEST.shape[0],1))
        self.BnPEST_SURF = self.BnPEST*4.*(np.pi)**2/(tmp)
        self.ss_plas_edge = np.argmin(np.abs(self.ss-1.0))


    def resonant_strength(self, min_s = 0, power = 1, n=2, SURFMN_coords = 0):
        '''
        Calculate dBres
        Finds the amplitude at each of the resonant harmonics (temp_discrete)
        Finds the integral of the resonant line
        SH : 26Feb2013
        '''
        #os.chdir(self.directory) 

        self.extract_q_profile_information(n=n, file_name=self.directory + '/PROFEQ.OUT')
        mk_grid, ss_grid = np.meshgrid(self.mk.flatten(), self.ss.flatten())
        #qn_grid, s_grid = np.meshgrid(self.q_profile*n, self.s.flatten())
        if SURFMN_coords:
            temp_qn  = griddata((mk_grid.flatten(),ss_grid.flatten()),np.abs(self.BnPEST_SURF.flatten()),(self.q_profile*n, self.q_profile_s.flatten()),method='linear')
            temp_discrete  = griddata((mk_grid.flatten(),ss_grid.flatten()),self.BnPEST_SURF.flatten(),(self.qn.flatten()*n, self.sq.flatten()),method='linear')
        else:
            temp_qn  = griddata((mk_grid.flatten(),ss_grid.flatten()),np.abs(self.BnPEST.flatten()),(self.q_profile*n, self.q_profile_s.flatten()),method='linear')
            temp_discrete  = griddata((mk_grid.flatten(),ss_grid.flatten()),self.BnPEST.flatten(),(self.qn.flatten()*n, self.sq.flatten()),method='linear')
        #print s.shape,len(s), temp_qn.shape, len(temp_qn)
        total_integral = 0
        min_location = np.argmin(np.abs(self.q_profile_s-min_s))
        print min_s, min_location, self.q_profile_s[min_location]
        for i in range(min_location,len(self.q_profile_s)-1):
            total_integral += temp_qn[i]*((self.q_profile_s[i+1]-self.q_profile_s[i])**power)

        #ax2.plot(temp_qn, s,'.-')
        #ax2.plot(mq*0, sq,'o')
        return total_integral, temp_discrete

    def extract_q_profile_information(self, n=2, file_name='PROFEQ.OUT'):
        '''Extract the q profile from file_name
        qn: q at the resonant surfaces
        sq: s at the resonant surfaces
        mq: m at the resonant surfaces
        i.e mq=n qn (sq)
        also returns
        q value on the s grid, q_profile, and q_profile_s (sgrid locations)
        SH : 26Feb2013
        '''
        #os.chdir(self.directory) 
        qn, sq, q, s, mq = return_q_profile(self.mk, file_name=file_name, n=n)
        self.qn = qn
        self.sq = sq
        self.q_profile = q
        self.q_profile_s = s
        self.mq = mq
        

    def kink_amp(self, s_surface, q_range, n = 2, SURFMN_coords = 0):
        '''Calculate dBkink
        Returns the harmonics that are between n q_range[0] < m < n q_range[1]
        at surface s= s_surface

        if SURFMN_coords=1, the calculation is done in BnPEST in SURFMN
        coordinates instaed of MARS-F coordinates

        returns m values, s value, amplitudes, q value at the surface of interest
        '''

        #Get q profile information, and find the surface we are interested in
        self.extract_q_profile_information(n=n, file_name=self.directory + '/PROFEQ.OUT')
        s_loc = np.argmin(np.abs(self.ss-s_surface))

        #q value at the relevant surface
        relevant_q = self.q_profile[s_loc]
        print np.max(self.mk), q_range[0]*relevant_q, q_range[1]*relevant_q

        #indices of the minimum m and maximum m that we are interested in
        lower_bound = np.argmin(np.abs(self.mk.flatten() - q_range[0]*relevant_q))
        upper_bound = np.argmin(np.abs(self.mk.flatten() - q_range[1]*relevant_q))
        print 'kink_amp: s_loc: %d, self.ss_val: %.2f, self.q_profile_s: %.2f'%(s_loc, self.ss[s_loc], self.q_profile[s_loc])
        #make sure upper bound isn't larger than the values available
        upper_bound_new = np.min([upper_bound, len(self.mk.flatten())-1])
        print lower_bound, upper_bound, upper_bound_new
        print 'relevant_q: %.2f, bounds: %d %d, values: %d, %d'%(relevant_q, lower_bound, upper_bound_new, self.mk.flatten()[lower_bound], self.mk.flatten()[upper_bound_new])

        #extract the relevant values
        if SURFMN_coords:
            relevant_values = self.BnPEST_SURF[s_loc,lower_bound:upper_bound_new]
        else:
            relevant_values = self.BnPEST[s_loc,lower_bound:upper_bound_new]

        print relevant_values
        print 'sum', np.abs(np.sum(np.abs(relevant_values)))
        return self.mk.flatten()[lower_bound:upper_bound_new], self.ss[s_loc], relevant_values, relevant_q


    def plot_SURFMN_MARS_F_comparison(self, mk, ss, tmp_plot_quantity, ss_plas_edge, n):
        '''MARS-F SURFMN comparison
        '''

        import matplotlib.pyplot as pt
        fig_tmp, ax_tmp = pt.subplots(nrows = 2, sharex=True, sharey=True)
        image2 = ax_tmp[1].pcolor(mk.flatten(),ss[:ss_plas_edge].flatten(), tmp_plot_quantity, cmap='hot', rasterized=True)
        ax_tmp[1].contour(mk.flatten(),ss[:ss_plas_edge].flatten(), tmp_plot_quantity, colors='white')
        ax_tmp[1].set_title('MARS-F, n=%d'%(n,))
        ax_tmp[1].set_ylabel(r'$\sqrt{\psi_N}$',fontsize=14)
        ax_tmp[1].set_xlabel('m')
        ax_tmp[1].plot(self.mq,self.sq,'wo')
        ax_tmp[1].plot(self.q_profile*n,self.q_profile_s,'w--') 

        image2.set_clim([0, np.max(tmp_plot_quantity)])
        image2.set_clim([0, 1.5])
        image1 = ax_tmp[0].pcolor(self.SURFMN_xdat,self.SURFMN_ydat,self.SURFMN_zdat, cmap = 'hot', rasterized=True)
        ax_tmp[0].contour(self.SURFMN_xdat,self.SURFMN_ydat,self.SURFMN_zdat, colors='white')
        #ax_tmp[0].set_title('SURFMN, n=%d, maximum: %.2f G/kA'%(n, np.max(np.abs(zdat))))
        ax_tmp[0].set_title('SURFMN, n=%d'%(n,))
        ax_tmp[0].plot(self.mq,self.sq,'wo')
        ax_tmp[0].plot(self.q_profile*n,self.q_profile_s,'w--') 
        ax_tmp[0].set_xlim([-29,29])
        ax_tmp[0].set_ylabel(r'$\sqrt{\psi_N}$',fontsize=14)
        image1.set_clim([0,np.max(np.abs(self.SURFMN_zdat))])
        image1.set_clim([0, 1.5])
        cb = pt.colorbar(mappable=image1, ax = ax_tmp[0])
        #from matplotlib.ticker import MaxNLocator
        #cb.locator = MaxNLocator( nbins = 6)
        cb.ax.set_ylabel('G/kA')
        cb = pt.colorbar(mappable=image2, ax = ax_tmp[1])
        cb.ax.set_ylabel('G/kA')
        fig_tmp.canvas.draw(); fig_tmp.show()
        #fig_tmp10, ax_tmp10 = pt.subplots()

    def plot_SURFMN_MARS_F_horizontal_cut(self, BnPEST, ss, ss_plas_edge, mk,psi_list=[0.92]):
        import matplotlib.pyplot as pt
        fig_tmp10, ax_tmp10 = pt.subplots()
        for tmp_psi in [0.92]:
            tmp_SURFMN_loc = np.argmin(np.abs(self.SURFMN_ydat[0,:]-tmp_psi))
            tmp_MARSF_loc = np.argmin(np.abs(ss[:ss_plas_edge] - tmp_psi))
            print tmp_SURFMN_loc, tmp_MARSF_loc, self.SURFMN_ydat[0,tmp_SURFMN_loc],ss[tmp_MARSF_loc]
            ax_tmp10.plot(self.SURFMN_xdat[:,tmp_SURFMN_loc], self.SURFMN_zdat[:,tmp_SURFMN_loc],'bx-', label = 'SURFMN s=%.2f'%(tmp_psi))
            print mk.shape, np.abs(BnPEST[tmp_MARSF_loc, : ]).shape
            ax_tmp10.plot(mk.flatten(), np.abs(BnPEST[tmp_MARSF_loc, : ]), 'kx-', label = 'MARS-F s=%.2f'%(tmp_psi))
        ax_tmp10.legend(loc='best')
        ax_tmp10.set_ylabel('mode amplitude')
        ax_tmp10.set_xlabel('m')
        fig_tmp10.canvas.draw(); fig_tmp10.show()

    def plot_compare_with_matlab(self, BnPEST, cmap='hot',RZ_dir = '/home/srh112/Desktop/Test_Case/matlab_outputs/'):
        '''Plot a comparison with Matlab output - useful for checking 
        for errors
        SH : 14Feb2013
        '''
        import matplotlib.pyplot as pt
        print 'reading in data from ', RZ_dir
        BnPEST_matlab = np.loadtxt(RZ_dir+'PEST_BnPEST_real.txt',delimiter=',')+np.loadtxt(RZ_dir+'PEST_BnPEST_imag.txt',delimiter=',')*1j
        ss_matlab = np.loadtxt(RZ_dir+'PEST_ss.txt',delimiter=',')
        mk_matlab = np.loadtxt(RZ_dir+'PEST_mk.txt',delimiter=',')

        mat_fig, mat_ax = pt.subplots()
        mat_image = mat_ax.pcolor(mk_matlab, (ss_matlab)**2, np.abs(BnPEST_matlab),cmap = cmap)
        mat_image.set_clim([0,1.2])
        mat_ax.set_xlim([-30,30])
        mat_ax.set_ylim([0,1])
        mat_fig.canvas.draw();mat_fig.show()

        print 'finished reading in data from ', RZ_dir

        fig_tmp, ax_tmp = pt.subplots(nrows = 2)
        diff = np.abs(BnPEST-BnPEST_matlab)
        self.diff = diff
        #self.BnPEST = BnPEST
        print 'max diff :'
        print np.max(diff), np.mean(np.max(diff)), np.max(diff).shape
        print 'as a percent:'
        self.diff_percent = diff/np.abs(BnPEST)*100.
        print np.max(self.diff_percent), np.mean(self.diff_percent)

        diff_image = ax_tmp[0].pcolor(mk_matlab, ss_matlab, diff, cmap = cmap)
        diff_image2 = ax_tmp[1].pcolor(mk_matlab, ss_matlab, (diff/np.abs(BnPEST))*100, cmap = cmap)
        diff_image.set_clim([0,0.1])
        diff_image2.set_clim([0,5])
        fig_tmp.canvas.draw(); fig_tmp.show()
        print diff_image.get_clim(), diff_image2.get_clim()

    def plot_SURFMN_MARS_F_radial_comparison(self, top_s, top_z, top_m, single_mode_plots, plot_quantity_top, n, single_mode_plots2, SURFMN_coords, surfmn_file):
        '''
        Plot lots of radial harmonic profiles
        SH : 14Feb2013
        '''
        import matplotlib.pyplot as pt
        if SURFMN_coords == 1:
            tmp_coords = 'SURFMN coords'
        elif SURFMN_coords == 0:
            tmp_coords = 'MARS-F coords'

        sqrt_modes = int(np.sqrt(len(single_mode_plots)))

        single_m_fig, single_m_ax = pt.subplots(nrows = sqrt_modes, ncols = sqrt_modes, sharey=True, sharex = True)
        single_m_fig2, single_m_ax2 = pt.subplots()
        for tmp_ax_label in single_m_ax[:,0]:
            tmp_ax_label.set_ylabel('Mode Amp G/kA')
        for tmp_ax_label in single_m_ax[-1,:]:
            tmp_ax_label.set_xlabel(r'$\sqrt{\psi_N}$',fontsize = 14)
        single_m_ax2.set_xlabel(r'$\sqrt{\psi_N}$',fontsize = 14)
        single_m_ax2.set_ylabel('Amplitude G/kA',fontsize = 14)
        single_m_ax2.set_title('Radial mode shapes - SURFMN (dashed), MARS-F (solid)',fontsize = 14)
        single_m_ax = single_m_ax.flatten()

        for tmp_m in single_mode_plots:
            tmp_single_loc = single_mode_plots.index(tmp_m)
            x_shift = 0.00
            tmp_top_loc = np.argmin(np.abs(top_m-tmp_m))
            #tmp_bottom_loc = np.argmin(np.abs(bottom_m-tmp_m))

            single_m_ax[tmp_single_loc].plot(top_s+x_shift, top_z[tmp_top_loc,:], '-b',label = '%s, n=%d'%(plot_quantity_top,n))
            if (tmp_m in single_mode_plots2):
                single_m_ax2.plot(top_s+x_shift, top_z[tmp_top_loc,:], '-b',label = '%s, n=%d'%(plot_quantity_top,n))

            #single_m_ax[tmp_single_loc].plot(bottom_s+x_shift, bottom_z[tmp_bottom_loc,:], label = '%s'%(plot_quantity_bottom,))
            include_surfmn_tmp = 1
            if include_surfmn_tmp:
                tmp_styles = ['--k','--r']
                for tmp_i, tmp_n in enumerate([2,4]):
                    tmp, xdat_tmp, ydat_tmp, zdat_tmp  = pyMARS_funcs.extract_surfmn_data(surfmn_file, tmp_n)
                    tmp_surfmn_m = xdat_tmp[:,0]
                    tmp_surfmn_s = ydat_tmp[0,:].flatten()
                    tmp_surfmn_loc = np.argmin(np.abs(tmp_surfmn_m-tmp_m))
                    if SURFMN_coords:
                        tmp_surfmn_z = zdat_tmp
                    else:
                        tmp_surfmn_z = zdat_tmp*self.SURFMN_A/((2*np.pi)**2)
                    single_m_ax[tmp_single_loc].plot(tmp_surfmn_s, tmp_surfmn_z[tmp_surfmn_loc,:], tmp_styles[tmp_i], label = '%s'%('SURFMN, n=%d'%(tmp_n),))
                    if (tmp_n == n) & (tmp_m in single_mode_plots2):
                        single_m_ax2.plot(tmp_surfmn_s, tmp_surfmn_z[tmp_surfmn_loc,:], tmp_styles[tmp_i], label = '%s'%('SURFMN, n=%d'%(tmp_n),))
            single_m_ax[tmp_single_loc].grid(b=True)
            single_m_ax2.grid(b=True)
            #single_m_ax[tmp_single_loc].set_xlabel('s')
            #single_m_ax[tmp_single_loc].set_xlabel('amp')
            single_m_ax[tmp_single_loc].set_title('m=%d'%(tmp_m,))
            if tmp_single_loc == (len(single_mode_plots)-1):
                print 'making legend'
                tmp_leg = single_m_ax[tmp_single_loc].legend(loc='best', fancybox = True)
                pt.setp(tmp_leg.get_texts(), fontsize=10)
            single_m_ax[tmp_single_loc].set_ylim([0,1.6])
            single_m_ax2.set_ylim([0,1.6])
        single_m_fig.suptitle('%s, %s'%(self.directory, tmp_coords))
        single_m_fig.canvas.draw(); single_m_fig.show()
        single_m_fig2.canvas.draw(); single_m_fig2.show()

    def get_SURFMN_data(self, surfmn_file, n):
        '''Extracts the SURFMN data from a surfmn file
        Only gets the data related to toroidal mode number n
        SH : 14Feb2013
        '''
        import h5py
        #fig_tmp, ax_tmp = pt.subplots(nrows = 2, sharex=True, sharey=True)
        if surfmn_file[-2:] == 'h5':
            print 'getting data from h5'
            tmp_file = h5py.File(surfmn_file)
            stored_data = tmp_file.get('1')
            zdat = stored_data[0][0]; xdat = stored_data[0][1]; ydat = stored_data[0][2]
        else:
            print 'getting surfmn natively, n = ', n
            tmp, xdat, ydat, zdat  = pyMARS_funcs.extract_surfmn_data(surfmn_file, n)
        #min_loc = np.argmin(np.abs(xdat[:,0]-(-30.)))
        #max_loc = np.argmin(np.abs(xdat[:,0]-(30.)))
        print 'ss, mk values :', min(self.ss), max(self.ss), min(self.mk), max(self.mk)
        #max_mk = np.argmin(np.abs(self.mk.flatten()-30))
        #min_mk = np.argmin(np.abs(self.mk.flatten()+30))
        #calculate flux area for conversion to SURFMN
        self.SURFMN_ydat = ydat; self.SURFMN_xdat = xdat; self.SURFMN_zdat = zdat

        self.SURFMN_A = griddata(self.ss[:181],self.A.flatten()[:181], self.SURFMN_ydat[0,:],method='linear')

    def pick_plot_quantity(self, plot_quantity_top, mk, ss_plas_edge, BnPEST):
        '''Determine which plot quantity to get... choices are SURFMN, MARS-F or PLOTK
        SH : 14Feb2013
        '''
        if plot_quantity_top == 'SURFMN':
            top_z_S = self.SURFMN_zdat
            top_m = self.SURFMN_xdat[:,0]
            top_s = self.SURFMN_ydat[0,:].flatten()
            top_z_M = self.SURFMN_zdat*self.SURFMN_A/((2*np.pi)**2)
        elif plot_quantity_top == 'MARS-F':
            top_m = mk.flatten()
            top_s = self.ss[:ss_plas_edge].flatten()
            top_z_M = np.abs(BnPEST[:ss_plas_edge,:]).transpose()
            top_z_S = top_z_M * 1./(np.array(self.A)[:ss_plas_edge]/((2.*np.pi)**2))
        elif plot_quantity_top == 'PLOTK':
            top_z_S = np.abs(self.plotk_mode_amps[:,:])
            #skip the first value in the file otherwise the shape looks weird
            #Note this is what Matt does in marsplot3d here : dumy = dumy[*,1:imax]
            top_s = np.loadtxt('PROFEQ.OUT')[1:self.plotk_mode_amps.shape[1]+1,0].flatten()
            #top_s = self.s[:self.plotk_mode_amps.shape[1]].flatten() 
            top_m = self.plotk_m[:]
            top_z_M = top_z_S * np.array(self.A)[0:self.plotk_mode_amps.shape[1]]/((2*np.pi)**2)#*self.SURFMN_DPSIDS
        return top_z_S, top_m, top_s, top_z_M

    def get_plotk_data(self, make_plot=1):
        '''Extract data from plotk routines and generate a plot
        SH : 14Feb2013
        '''
        self.plotk_m, self.plotk_mode_amps = extract_plotk_results()
        self.plotk_mode_amps = self.plotk_mode_amps*self.BNORM*2.*np.pi
        if make_plot:
            plotk_fig, plotk_ax = pt.subplots()
            color_plot = plotk_ax.pcolor(self.plotk_m, (s[1:self.plotk_mode_amps.shape[1]+1]), np.abs(self.plotk_mode_amps).transpose(),cmap='hot')
            pt.colorbar(color_plot,ax = plotk_ax)
            plotk_ax.set_title('PLOT_K, max : %.2f'%(np.max(np.abs(self.plotk_mode_amps)),))
            plotk_fig.canvas.draw(); plotk_fig.show()


    def plot_BnPEST_phase(self, inc_phase, BnPEST, mk, ss, tmp_cmap, phase_correction, clim_value, title, temp_qn, mk_grid, ss_grid, ss_squared, n, suptitle, fig_show, fig_name):
        import matplotlib.pyplot as pt
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
            ax.plot(self.mq,self.sq**2,'wo')
            ax.plot(self.q_profile*n,self.q_profile_s**2,'w--') 
        else:
            print 'not ss_squared'
            color_ax = ax.pcolor(mk,ss,np.abs(BnPEST),cmap='hot')#tmp_cmap)
            ax.plot(self.mq,self.sq,'wo')
            #ax.plot(tmp_mk_range, tmp_mk_range*0+tmp_ss,'ko')
            #tmp_relevant_values = self.kink_amp(0.92, [2,4])
            ax.plot(self.q_profile*n,self.q_profile_s,'w--') 

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
            ax3.plot(self.mq,self.sq,'bo')
            ax3.plot(self.q_profile*n,self.q_profile_s,'b--') 

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


        ax2.plot(temp_qn, self.q_profile_s,'.-')
        ax2.plot(self.mq*0, self.sq,'o')
        for i in range(0,len(self.mq)):
            ax2.plot([0, np.max(temp_qn)],[self.sq[i], self.sq[i]],'--')
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])

        if inc_phase!=0:
            temp_qn_angle  = griddata((mk_grid.flatten(),ss_grid.flatten()),angles.flatten(), (self.q_profile*n, self.q_profile_s.flatten()),method='linear')
            ax4.plot(temp_qn_angle*180./np.pi, s,'.-')
            for i in range(0,len(self.mq)):
                ax4.plot([lower_limit*180./np.pi, upper_limit*180./np.pi],[self.sq[i], self.sq[i]],'--')
                ax4.set_xlim([lower_limit*180./np.pi,upper_limit*180./np.pi])
                ax4.set_ylim([0,1])

        fig.suptitle(suptitle)
        if fig_show==1:
            fig.canvas.draw()
            fig.show()
        if fig_name != '':
            print 'saving figure'
            fig.savefig(fig_name, dpi=200)
            fig.clf()
            pt.close('all')


    def plot_radial_lines_overlay(self, start_mode, end_mode, top_m, top_s, top_z, plot_quantity_top, bottom_m, bottom_s, bottom_z, plot_quantity_bottom, mk, ss, ss_plas_edge, BnPEST):
        import matplotlib.pyplot as pt

        fig_tmp10, ax_tmp10 = pt.subplots(nrows = 2, sharex = True)
        #start_mode = -20; end_mode = 25
        colormap = pt.cm.jet
        tmp_color = ax_tmp10[0].pcolor(np.array([[start_mode,end_mode],[start_mode,end_mode]]),cmap=colormap)
        ax_tmp10[0].cla()
        tmp_cbar = pt.colorbar(mappable = tmp_color, ax=ax_tmp10[0])
        tmp_cbar.ax.set_title('m')
        tmp_cbar = pt.colorbar(mappable = tmp_color, ax=ax_tmp10[1])
        tmp_cbar.ax.set_title('m')

        ax_tmp10[1].set_color_cycle([colormap(i) for i in np.linspace(0, 1.0, start_mode + (end_mode))])
        ax_tmp10[0].set_color_cycle([colormap(i) for i in np.linspace(0, 1.0, start_mode + (end_mode))])
        fig_tmp100, ax_tmp100 = pt.subplots(nrows = 2, sharex = True)

        for tmp_m in range(start_mode,end_mode):
            correction_factor = 1.
            tmp_top_loc = np.argmin(np.abs(top_m-tmp_m))
            tmp_bottom_loc = np.argmin(np.abs(bottom_m-tmp_m))

            ax_tmp100[0].plot(top_s, top_z[tmp_top_loc,:], '-x',label = '%s s=%.2f'%(plot_quantity_top, tmp_m))
            ax_tmp100[1].plot(top_s, top_z[tmp_top_loc,:]/correction_factor, 'k-',label = '%s s=%.2f'%(plot_quantity_top, tmp_m))

            #print tmp_bottom_loc, bottom_s.shape, bottom_z[tmp_bottom_loc,:].flatten().shape
            ax_tmp100[1].plot(bottom_s, bottom_z[tmp_bottom_loc,:], label = '%s s=%.2f'%(plot_quantity_bottom, tmp_m))
            ax_tmp100[1].set_title('%s (colour), %s / %.2f (black)'%(plot_quantity_bottom, plot_quantity_top, correction_factor))
            ax_tmp100[0].set_title('%s'%(plot_quantity_top))

            for j in range(0,len(top_s), 5):
                #ax_tmp10[0].text(ydat[tmp_SURFMN_loc,j], zdat[tmp_SURFMN_loc,j], str(tmp_m), fontsize = 8.5)
                ax_tmp100[0].text(top_s[j], top_z[tmp_top_loc, j], str(tmp_m), fontsize = 8.5)

            for j in range(0,len(bottom_s), 5):
                #ax_tmp10[0].text(ydat[tmp_SURFMN_loc,j], zdat[tmp_SURFMN_loc,j], str(tmp_m), fontsize = 8.5)
                #print bottom_s[j], bottom_z[tmp_bottom_loc, j], str(tmp_m)
                ax_tmp100[1].text(bottom_s[j], bottom_z[tmp_bottom_loc, j], str(tmp_m), fontsize = 8.5)
                pass
            compare_MARS = 1
            correct_SURFMN = 0
            tmp_SURFMN_loc = np.argmin(np.abs(self.SURFMN_xdat[:,0]-tmp_m))
            if correct_SURFMN:
                plot_quantity = self.SURFMN_zdat[tmp_SURFMN_loc,:]*self.SURFMN_A/((2*np.pi)**2)#*self.SURFMN_DPSIDS
            else:
                plot_quantity = self.SURFMN_zdat[tmp_SURFMN_loc,:]
            ax_tmp10[0].plot(self.SURFMN_ydat[tmp_SURFMN_loc,:], plot_quantity, label = 'SURFMN s=%.2f'%(tmp_m))
            ax_tmp10[1].plot(self.SURFMN_ydat[tmp_SURFMN_loc,:], plot_quantity/correction_factor, 'k-',label = 'SURFMN s=%.2f'%(tmp_m))
            for j in range(0,len(self.SURFMN_ydat[tmp_SURFMN_loc,:]), 5):
                #ax_tmp10[0].text(ydat[tmp_SURFMN_loc,j], zdat[tmp_SURFMN_loc,j], str(tmp_m), fontsize = 8.5)
                ax_tmp10[0].text(self.SURFMN_ydat[tmp_SURFMN_loc,j], plot_quantity[j], str(tmp_m), fontsize = 8.5)
            if compare_MARS:
                tmp_MARSF_loc = np.argmin(np.abs(mk.flatten() - tmp_m))
                ax_tmp10[1].plot(ss[:ss_plas_edge].flatten(), np.abs(BnPEST[:ss_plas_edge,tmp_MARSF_loc]), label = 'MARS-F s=%.2f'%(tmp_m))
                ax_tmp10[1].set_title('MARS-F (colour), SURFMN * A / (2pi)^2 /%.2f (black)'%(correction_factor))
                for j in range(0,len(ss[:ss_plas_edge]), 5):
                    ax_tmp10[1].text(ss[j].flatten(), np.abs(BnPEST[j,tmp_MARSF_loc]), str(tmp_m), fontsize = 8.5)

            else:
                tmp_plotk_loc = np.argmin(np.abs(self.plotk_m[:]-tmp_m))
                plot_quantity = np.abs(self.plotk_mode_amps[tmp_plotk_loc,:]).transpose()
                x_axis = self.s[1:self.plotk_mode_amps.shape[1]+1]
                ax_tmp10[1].plot(x_axis, plot_quantity, label = 'plotk s=%.2f'%(tmp_m))
                for j in range(0,len(self.SURFMN_ydat[tmp_SURFMN_loc,:]), 5):
                    ax_tmp10[1].text(x_axis[j], plot_quantity[j], str(tmp_m), fontsize = 8.5)
                ax_tmp10[1].set_title('plotk')

        ax_tmp10[1].set_ylabel('mode amplitude')
        ax_tmp10[0].set_ylabel('mode amplitude')
        ax_tmp10[0].set_title('SURFMN * A / (2pi)^2')
        ax_tmp10[1].set_xlabel('s')
        fig_tmp10.canvas.draw(); fig_tmp10.show()
        fig_tmp100.canvas.draw(); fig_tmp100.show()

    def load_SURFMN_data(self, surfmn_file, n, horizontal_comparison=0, PEST_comparison=0, single_radial_mode_plots=0, all_radial_mode_plots=0):
        self.extract_q_profile_information(n=n, file_name=self.directory + 'PROFEQ.OUT')
        self.get_SURFMN_data(surfmn_file, n)

        #This is if we want to plot SURFMN data in MARS-F coords
        self.SURFMN_A = griddata(self.ss[:181],self.A.flatten()[:181], self.SURFMN_ydat[0,:],method='linear')
        tmp_plot_quantity = (np.abs(self.BnPEST[:self.ss_plas_edge,:]).transpose()/(self.A[:self.ss_plas_edge]/(4*np.pi**2))).transpose()
        print 'tmp plot shape', tmp_plot_quantity.shape, self.mk.flatten().shape, self.ss[:self.ss_plas_edge].flatten().shape
        if PEST_comparison:
            self.plot_SURFMN_MARS_F_comparison(self.mk, self.ss, tmp_plot_quantity, self.ss_plas_edge, n)
        #horizontal cut of comparison between MARS-F and SURFMN
        if horizontal_comparison:
            self.plot_SURFMN_MARS_F_horizontal_cut(self.BnPEST, self.ss, self.ss_plas_edge, self.mk, psi_list=[0.92])


        plot_quantity_top = 'MARS-F'
        plot_quantity_bottom = 'SURFMN'
        #make the radial comparison plots that I sent to Yueqiang when trying to get co-ordinates right
        top_z_S, top_m, top_s, top_z_M = self.pick_plot_quantity(plot_quantity_top, self.mk, self.ss_plas_edge, self.BnPEST)
        bottom_z_S, bottom_m, bottom_s, bottom_z_M = self.pick_plot_quantity(plot_quantity_bottom, self.mk, self.ss_plas_edge, self.BnPEST)

        SURFMN_coords = 1
        if SURFMN_coords:
            top_z = top_z_S
            bottom_z = bottom_z_S
        else:
            top_z = top_z_M
            bottom_z = bottom_z_M
        start_mode = -20; end_mode = 25
        single_mode_plots = range(1,3**2+1)
        single_mode_plots2 = [1,3,9]
        if single_radial_mode_plots:
            self.plot_SURFMN_MARS_F_radial_comparison(top_s, top_z, top_m, single_mode_plots, plot_quantity_top, n, single_mode_plots2, SURFMN_coords, surfmn_file)
        if all_radial_mode_plots:
            self.plot_radial_lines_overlay(start_mode, end_mode, top_m, top_s, top_z, plot_quantity_top, bottom_m, bottom_s, bottom_z, plot_quantity_bottom, self.mk, self.ss, self.ss_plas_edge, self.BnPEST)



    def plot1(self, suptitle='', title='', fig_name = '', fig_show = 1,clim_value=[0,1],inc_phase=1, phase_correction=None, cmap = 'gist_rainbow_r', ss_squared = 0, surfmn_file = None, n=2, increase_grid_BnPEST = 0, single_mode_plots = range(1,3**2+1), single_mode_plots2 = [1,5,9]):
        #os.chdir(self.directory) 
        self.extract_q_profile_information(n=n, file_name=self.directory + '/PROFEQ.OUT')

        mk_grid, ss_grid = np.meshgrid(self.mk.flatten(), self.ss.flatten())
        #qn_grid, s_grid = np.meshgrid(self.q_profile*n, self.s.flatten())

        #print q.shape, s.shape, mk_grid.shape, ss_grid.shape, self.BnPEST.shape
        temp_qn  = griddata((mk_grid.flatten(),ss_grid.flatten()),np.abs(self.BnPEST.flatten()),(self.q_profile*n, self.q_profile_s.flatten()),method='linear')
        if increase_grid_BnPEST:
            mk,ss,BnPEST=RZfuncs.increase_grid(self.mk.flatten(),self.ss.flatten(),abs(self.BnPEST),number=200)
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

        #tmp_mk_range, tmp_ss, tmp_relevant_values, relevant_q_values = self.kink_amp(0.92, [2,4], n = n)

        if surfmn_file != None:
            self.get_SURFMN_data(surfmn_file, n)
            #This is if we want to plot SURFMN data in MARS-F coords
            self.SURFMN_A = griddata(self.ss[:181],self.A.flatten()[:181], self.SURFMN_ydat[0,:],method='linear')
            tmp_plot_quantity = (np.abs(BnPEST[:ss_plas_edge,:]).transpose()/(self.A[:ss_plas_edge]/(4*np.pi**2))).transpose()
            print 'tmp plot shape', tmp_plot_quantity.shape, mk.flatten().shape, ss[:ss_plas_edge].flatten().shape
            #color comparison between MARS-F and SURFMN
            self.plot_SURFMN_MARS_F_comparison(mk, ss, tmp_plot_quantity, ss_plas_edge, n)
            #horizontal cut of comparison between MARS-F and SURFMN
            self.plot_SURFMN_MARS_F_horizontal_cut(BnPEST, ss, ss_plas_edge, mk, psi_list=[0.92])
            #tmp hack trying to see difference between SURFMN and MARS-F
            inc_DPSIDS = 0
            if inc_DPSIDS:
                self.DPSIDS_1 = np.loadtxt('OUT.dat')
                print 'DPSIDS shape', self.DPSIDS_1.shape
                print 'ydat shape', self.SURFMN_ydat[1,:].shape
                #dchi = self.chi[0,1]-self.chi[0,0]
                #self.A = 2.*np.pi*np.sum(self.jacobian * dchi, axis = 1)
                self.SURFMN_DPSIDS = griddata(self.ss[:181],self.DPSIDS_1, self.SURFMN_ydat[0,:],method='linear')
            #self.SURFMN_A = griddata(self.ss[:181],self.A.flatten()[:181], self.SURFMN_ydat[0,:],method='linear')
            get_plotk_results = 0
            if get_plotk_results:
                self.get_plotk_data(make_plot=1)
            start_mode = -20; end_mode = 25
            #fig_tmp100, ax_tmp100 = pt.subplots(nrows = 2, sharex = True)
            plot_quantity_top = 'MARS-F'
            plot_quantity_bottom = 'SURFMN'
            
            top_z_S, top_m, top_s, top_z_M = self.pick_plot_quantity(plot_quantity_top, mk, ss_plas_edge, BnPEST)
            bottom_z_S, bottom_m, bottom_s, bottom_z_M = self.pick_plot_quantity(plot_quantity_bottom, mk, ss_plas_edge, BnPEST)

            SURFMN_coords = 1
            if SURFMN_coords:
                top_z = top_z_S
                bottom_z = bottom_z_S
            else:
                top_z = top_z_M
                bottom_z = bottom_z_M

            #make the radial comparison plots that are in the paper
            self.plot_SURFMN_MARS_F_radial_comparison(top_s, top_z, top_m, single_mode_plots, plot_quantity_top, n, single_mode_plots2, SURFMN_coords, surfmn_file)

            #make the radial comparison plots that I sent to Yueqiang when trying to get co-ordinates right
            start_mode = -20; end_mode = 25
            self.plot_radial_lines_overlay(start_mode, end_mode, top_m, top_s, top_z, plot_quantity_top, bottom_m, bottom_s, bottom_z, plot_quantity_bottom, mk, ss, ss_plas_edge, BnPEST)

            #Compare with Matlab RZplot routines output
            compare_with_matlab=0
            if compare_with_matlab:
                self.plot_compare_with_matlab(BnPEST, cmap='hot', RZ_dir = '/home/srh112/Desktop/Test_Case/matlab_outputs/')

        self.plot_BnPEST_phase(inc_phase, BnPEST, mk, ss, tmp_cmap, phase_correction, clim_value, title, temp_qn, mk_grid, ss_grid, ss_squared, n, suptitle, fig_show, fig_name)

    def plot_BnPEST(self, ax, n=2, inc_contours = 1, contour_levels = None, phase = 0, increase_grid_BnPEST = 0, min_phase = -130, max_ss = 1.0, interp_points = 100, gauss_filter = [0,0.5], cmap = 'hot'):
        ss_plas_edge = np.argmin(np.abs(self.ss-max_ss))
        if phase==1:
            tmp_plot_quantity = np.angle(self.BnPEST[:ss_plas_edge,:], deg = True)
            tmp_plot_quantity[tmp_plot_quantity<min_phase] += 360
        else:
            tmp_plot_quantity = np.abs((np.abs(self.BnPEST[:ss_plas_edge,:]).transpose()/(self.A[:ss_plas_edge]/(4*np.pi**2))).transpose())
        if increase_grid_BnPEST:
            tmp_plot_mk, tmp_plot_ss, tmp_plot_quantity = RZfuncs.increase_grid(self.mk.flatten(), self.ss.flatten()[:ss_plas_edge], tmp_plot_quantity, increase_y = 1, increase_x = 0, new_y_lims = [0,0.99],number=interp_points)
            tmp_plot_quantity =  ndimage.gaussian_filter(tmp_plot_quantity,gauss_filter)
        else:
            tmp_plot_mk = self.mk.flatten()
            tmp_plot_ss = self.ss[:ss_plas_edge].flatten()


        #color_ax = ax.pcolor(self.mk.flatten(),self.ss[:ss_plas_edge].flatten(), tmp_plot_quantity, cmap=cmap, rasterized=True)
        if increase_grid_BnPEST:
            color_ax = ax.imshow(tmp_plot_quantity, cmap=cmap, aspect='auto', interpolation='bicubic', rasterized=True, extent = [np.min(tmp_plot_mk), np.max(tmp_plot_mk), np.min(tmp_plot_ss), np.max(tmp_plot_ss)], origin='lower')
        else:
            color_ax = ax.pcolor(tmp_plot_mk,tmp_plot_ss, tmp_plot_quantity, cmap=cmap, rasterized=True)
        self.tmp_plot_quantity = tmp_plot_quantity
        if inc_contours:
            if contour_levels==None:
                ax.contour(tmp_plot_mk,tmp_plot_ss, tmp_plot_quantity, colors='white')
            else:
                ax.contour(tmp_plot_mk,tmp_plot_ss, tmp_plot_quantity, contour_levels, colors='white')
                #ax.contour(self.mk.flatten(),self.ss[:ss_plas_edge].flatten(), tmp_plot_quantity, contour_levels, colors='white')

        #color_ax = ax.pcolor(self.mk,self.ss,np.abs(self.BnPEST),cmap=cmap)
        file_name = self.directory + '/PROFEQ.OUT'
        qn, sq, q, s, mq = return_q_profile(self.mk,file_name=file_name, n=n)
        ax.plot(mq,sq,'wo')
        ax.plot(q*n,s,'w--') 
        return color_ax


    

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
