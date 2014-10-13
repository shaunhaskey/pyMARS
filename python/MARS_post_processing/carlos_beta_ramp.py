import numpy as np
import matplotlib.pyplot as pt
import pyMARS.dBres_dBkink_funcs as dBres_dBkink
import pyMARS.generic_funcs as gen_func

file_name = '/home/srh112/NAMP_datafiles/mars/shot_142614_expt_scan_NC_const_eqV6/shot_142614_expt_scan_NC_const_eqV6_post_processing_PEST.pickle'

file_name = '/home/srh112/NAMP_datafiles/mars/156746_02113_betaNramp/156746_02113_betaNramp_post_processing.pickle'

print file_name
phasing = 0
n = 2
phase_machine_ntor = 0
s_surface = 0.92
fixed_harmonic = 3
reference_dB_kink = 'plas'
reference_offset = [4,0]
sort_name = 'time_list'
a = dBres_dBkink.post_processing_results(file_name, s_surface, phasing, phase_machine_ntor, fixed_harmonic = fixed_harmonic, reference_offset = reference_offset, reference_dB_kink = reference_dB_kink, sort_name = sort_name, try_many_phasings = False)
#dBres = dBres_dBkink.dBres_calculations(a, mean_sum = 'sum')
#dBkink = dBres_dBkink.dBkink_calculations(a)
probe = dBres_dBkink.magnetic_probe(a,'66M')
#xpoint = dBres_dBkink.x_point_displacement_calcs(a, phasing)
fig, ax = pt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)

plot_style = {'marker':'x'}
multiplier = 1
probe.plot_single_phasing(phasing, 'BNLI', field = 'plasma', plot_kwargs = plot_style, amplitude = True, ax = ax, multiplier = multiplier)
fig.canvas.draw();fig.show()
