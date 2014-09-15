import os
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--shot', type = int, help = 'Shot number', required = True)
parser.add_argument('--time', type = int, help = 'Shot time', required = True)
parser.add_argument('--dir', type=str, help = 'efit dir', required = True)
parser.add_argument('--name', type=str, help = 'identifier in mars dir', required = True)
parser.add_argument('--n', type=int, help = 'mode number')
parser.add_argument('--rote', type=float, help = 'rotation, 0 no rotation, 1 rotation', )
parser.add_argument('--eta', type=float, help = 'resistivity, 0 no resist, 1 resist', )
parser.add_argument('--simul', type=str, help = 'Type of simulation - single or betaN_ramp', )
parser.add_argument('--freq', type=float, help = 'I-coil freq (Hz)', )

args = vars(parser.parse_args())
print args

#defaults
for name, val in zip(['eta','rote','simul','n','freq'], [0, 0, 'single', -2, 10.]):
    if args[name] == None:
        args[name] = val

print args
#Required args
shot = args['shot']
time = args['time']
original_dir = args['dir']
suffix = args['name']
n = args['n']
simul = args['simul']
#simul = 'single' #'single', 'betaN_ramp'

eta_val = -1; rote_val = -1

single_run = 1 if simul == 'single' else 0
start_dir = os.getcwd()
#shot = 157312
#time = 4575
#shot = 156746
#time = 4617
#shot = 158115
#time = 4780
#n = -2
#suffix = 'raffi2_ideal'
#original_dir = '/u/nazikian/CARLOS_2014/158115/shot158115_kinetic_RN/'
#original_dir = '/u/paz-soldan/efit/156746/'
#original_dir = '/u/nazikian/CARLOS_2014/158115_rn02/shot158115_kinetic_RN_2/'
if original_dir[-1]!='/': original_dir +='/'
mars_ind_dir = 'shot{}_{:05d}_{}_{}'.format(shot, time, simul, suffix)
#efit_dir = '/u/haskeysr/efit/shot158115_04780/'
efit_dir = '/u/haskeysr/efit/shot{}_{:05d}/'.format(shot, time)
efit_dir = '/u/haskeysr/efit/{}/'.format(mars_ind_dir,)
mars_dir = '/u/haskeysr/mars/{}/'.format(mars_ind_dir)
if efit_dir[-1]!='/':efit_dir+='/'
print efit_dir
os.system('mkdir {}'.format(efit_dir))
orig_list = os.listdir(original_dir)
filt_list = []
for i in orig_list:
    if i.find(str(shot))>=0 and i.find(str(time))>=0:
        filt_list.append(i)
for i in filt_list: 
    j = i
    if i.find('_')>i.find(str(time)):
        j = i[:i.find('_')]  
        ms = i[i.find('_'):]
    else:
        j = i
        ms = ''
    copy_string = 'cp {}{} {}{}'.format(original_dir, i, efit_dir,j)
    print copy_string
    os.system(copy_string)

#Extract the profiles - need some kind of if in here
extra = ''
filename = '{}/p{}.{:05d}{}'.format(efit_dir, shot, time, extra)
final_names_link = ['PROFDEN', 'PROFTE', 'PROFTI', 'PROFROT']
final_names = ['dne', 'dte', 'dti', 'dtrot']
multipliers = [10**20,1,1,1]
format = ['{:.4e}','{:.4f}','{:.4f}','{:.4f}']
final_names = ['{}{}.{:05d}.dat'.format(i, shot, time) for i in final_names]
search_terms = ['ne','te','ti','omeg']
with file(filename,'r') as handle:lines = handle.readlines()
os.chdir(efit_dir)
def mod_data(in_list, mult, form):
    out_list = [' {}        {}\n'.format(len(in_list), '2')]
    for i in in_list:
        tmp = " ".join(i.split())
        tmp = tmp.split()
        val = form.format(float(tmp[1])*mult)
        out_list.append('{:.4f}        {}\n'.format((float(tmp[0]))**0.5, val))
    return out_list

for fname, s_term, fname_link, mult, form in zip(final_names, search_terms, final_names_link, multipliers, format):
    print fname, s_term
    success = 0
    for i in range(len(lines)):
        if lines[i].find(s_term)>=0:
            success = 1
            break
    if success:
        n_terms = int(lines[i].split(' ')[0])
        data = mod_data(lines[i+1:i+1+n_terms], mult, form)
        with file(efit_dir + fname,'w') as handle:handle.writelines(data)
        os.system('ln -sf {} {}'.format(fname, fname_link))
os.system('mkdir {}'.format(mars_dir))
input_template = '/u/haskeysr/mars/templates/BetaRampTemplate.cfg'

#input_template = '/u/haskeysr/mars/raffi_157312_n3RMP/input.cfg'
with file(input_template,'r') as file_handle: inp_temp = file_handle.readlines()
mods = [['project_name','project_name = {}\n'.format(mars_ind_dir)],
        ['efit_file_location','efit_file_location = {}\n'.format(efit_dir)],
        ['profile_file_location','profile_file_location = {}\n'.format(efit_dir)],
        ['<<RNTOR>>', '<<RNTOR>> : {}\n'.format(n)],
        ['<<ROTE>>', '<<ROTE>> : {}\n'.format(args['rote'])],
        ['<<ETA>>', '<<ETA>> : {}\n'.format(args['eta'])],
        ['I_coil_frequency', 'I_coil_frequency = {}\n'.format(args['freq'])],
        ['single_runthrough','single_runthrough = {}\n'.format(single_run)]]

for i in mods:
    print i
    for j in range(len(inp_temp)):
        if inp_temp[j].find(i[0])>=0:
            print 'found', inp_temp[j], i[0]
            inp_temp[j] = i[1]

with file(mars_dir + 'input.cfg','w') as file_handle: file_handle.writelines(inp_temp)
print mars_dir
print mars_dir + 'input.cfg'
print 'cd {};'.format(mars_dir) + 'nohup unbuffer run_pyMARS input.cfg &>log &'
os.chdir(start_dir)
