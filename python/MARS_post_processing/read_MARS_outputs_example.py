import pickle
import pyMARS.results_class as results_class
import pickle,sys
import numpy as np
import pyMARS.PythonMARS_funcs as pyMARS_funcs
import pyMARS.RZfuncs as RZfuncs
import matplotlib.pyplot as plt

# This is a simulation where the upper and lower I-coils were simulated separately
upper_and_lower = True

# pickle file that has all the run information in it
fname = '/home/weisbergd/mars/test_runk/test_runk_post_processing_PEST.pickle'
with file(fname,'r') as filehandle:
    project_dict = pickle.load(filehandle)

# Select a particular simulation in that run file that we will look at
sim_id = 1

# Get the probe location details which are also stored in the input file
probe = project_dict['details']['pickup_coils']['probe']
probe_type = project_dict['details']['pickup_coils']['probe_type']
Rprobe = project_dict['details']['pickup_coils']['Rprobe']
Zprobe = project_dict['details']['pickup_coils']['Zprobe']
tprobe = project_dict['details']['pickup_coils']['tprobe']
lprobe = project_dict['details']['pickup_coils']['lprobe']
link_RMZM = 0

# toroidal mode number and normalization factor
n = np.abs(project_dict['sims'][sim_id]['MARS_settings']['<<RNTOR>>'])
I0EXP = RZfuncs.I0EXP_calc_real(n, project_dict['details']['I-coils']['I_coil_current'])
Nchi = project_dict['sims'][sim_id]['CHEASE_settings']['<<NCHI>>']

# Loop through upper/lower and plasma/vacuum combinations and get the results for each case
locs = ['upper','lower'] if upper_and_lower else ['']
types = ['plasma','vacuum']
outputs = {}
for loc in locs:
    outputs[loc] = {}
    for type in types:
        directory = project_dict['sims'][sim_id]['dir_dict']['mars_{}_{}_dir'.format(loc,type)]
        print directory
        new_data = results_class.data(directory,Nchi=240,link_RMZM=0, I0EXP=I0EXP, spline_B23=2)
        new_data_R = new_data.R*new_data.R0EXP
        new_data_Z = new_data.Z*new_data.R0EXP

        # Here you can get probe outputs for a particular case
        coil_outputs = pyMARS_funcs.coil_responses6(new_data_R,new_data_Z,new_data.Br,new_data.Bz,new_data.Bphi,probe, probe_type, Rprobe,Zprobe,tprobe,lprobe)
        facn = 1.

        # Get the outputs in PEST co-ordinates so that we can make m vs rho type plots 
        new_data.get_PEST(facn = facn)

        # Store this particular output so we can do things with it later
        outputs[loc][type] = new_data

# ---------
# Some plotting examples

# Plot Bn for the upper I-coil including plasma response in real space and in m vs rho space in PEST coordinates (straight field line)
tmp = outputs['upper']['plasma']
fig, ax = plt.subplots(ncols = 2)
tmp.plot_BnPEST(ax[0])

# Note this is called plot_Bn, but you actually pass it what you want it to plot.... bad historical naming...
tmp.plot_Bn(np.abs(tmp.Bn),axis=ax[1],cmap='plasma',plot_coils_switch = True)
fig.canvas.draw();fig.show()

# Example where we superpose the upper and lower fields with a phase shift between them
# We also extract out the plasma generated component of the magnetic field

fig, ax = plt.subplots(ncols = 4,nrows=2,sharex = True, sharey=True)
axf =ax.flatten()

# Phase shifts to apply between the coils
phases = np.linspace(0,2.*np.pi,8,endpoint = False)

# Loop through the phase shifts and plot each one
for ind, (ph, tmp_ax) in enumerate(zip(phases,axf)):
    # Include the phase shift in one of the arrays, and subtract vacuum from plasma to get the plasma only contribution
    plasma_only_upper_field = outputs['upper']['plasma'].Bn*np.exp(1j*ph) - outputs['upper']['vacuum'].Bn*np.exp(1j*ph)

    # Keep the phase of the lower field fixed
    plasma_only_lower_field = outputs['lower']['plasma'].Bn - outputs['lower']['vacuum'].Bn

    # Combine the fields from the upper and lower arrays
    plasma_field = plasma_only_upper_field + plasma_only_lower_field

    # Plot the magnitude of the plasma contribution to the field
    # Note that the method is being called from outputs['upper']['plasma'] but that doesn't matter 
    # because each one has the same R x Z grid, and we pass the quantity we want to plot on that grid
    im = outputs['upper']['plasma'].plot_Bn(np.abs(plasma_field),axis=tmp_ax,cmap='viridis',plot_coils_switch = True)

    # Set all on the same color limits
    if ind==0:
        clim = im.get_clim()
    im.set_clim(clim)
    tmp_ax.set_title('{:.3f}deg'.format(np.rad2deg(ph)))
fig.canvas.draw();fig.show()
