'''

/u/paz-soldan/efit/158117/MARS/
Example for single simulation

#This one works fine
python extract_profiles_p_file.py --shot 158117 --time 03556 --dir /u/paz-soldan/efit/158117/MARS/ --name test_mar25 --n 2 --rote 1 --eta 1 --simul single --freq 10


#This one works fine
python extract_profiles_p_file.py --shot 156746 --time 02113 --dir /u/haskeysr/tmp/shot156746_02113_betaN_ramp_carlos_prlV2/ --name test2_mar --n 2 --rote 1 --eta 1 --simul single --freq 10

works with a,g,m,p files
a,g,p - works
g,p - doesn't work
m,g,p - doesn't work

python extract_profiles_p_file.py --shot 161198 --time 03550 --dir /u/haskeysr/tmp/161198/MARS/ --name carlos_hicol --n 2 --rote 1 --eta 1 --simul betaN_ramp  --freq 10

python extract_profiles_p_file.py --shot 161205 --time 03215 --dir /u/haskeysr/tmp/161205/MARS/ --name carlos_lmode2 --n -2 --rote -1 --eta -1 --simul betaN_ramp --freq 10

python extract_profiles_p_file.py --shot 161198 --time 03550 --dir /u/haskeysr/tmp/161198/MARS/ --name carlos_hicol2 --n -2 --rote -1 --eta -1 --simul betaN_ramp --freq 10

python extract_profiles_p_file.py --shot 158103 --time 03796 --dir /u/haskeysr/tmp/158103/MARS/ --name carlos_2 --n -2 --rote -1 --eta -1 --simul betaN_ramp --freq 10

Modify so that rote=-1 and eta=-1, and n=-2... does it matter if n=-2 or 2?!!!
python extract_profiles_p_file.py --shot 158115 --time 04780 --dir /u/haskeysr/efit/shot158115_04780/ --name retestV2 --n -2 --rote -1 --eta -1 --simul betaN_ramp --freq 10

python extract_profiles_p_file.py --shot 157308 --time 04200 --dir /u/nazikian/IAEA/157308/shot157308_kinetic_RN/ --name raffi_n3_jan2016 --n -3 --rote -1 --eta -1 --simul q95_scan --freq 10



16June2015 test that the results are the same wih the old data : 
python extract_profiles_p_file.py --shot 158115 --time 04780 --dir /u/haskeysr/efit/shot158115_04780/ --name retest_carlos_prl_case --n -2 --rote -1 --eta -1 --simul betaN_ramp --freq 10
* ends up giving slightly different results.... try again:
python extract_profiles_p_file.py --shot 158115 --time 04780 --dir /u/haskeysr/efit/shot158115_04780_betaN_ramp_retest2/ --name retest_carlos_prl_caseV2 --n -2 --rote -1 --eta -1 --simul betaN_ramp --freq 10

Multi efit example:
python extract_profiles_p_file.py --shot 158103 --time 12 --dir /u/paz-soldan/efit/158103/VARYPED_5x5/ --name varyped_test --n -2 --rote -1 --eta -1 --simul multi_efit --freq 10

python extract_profiles_p_file.py --shot 158103 --time 12 --dir /u/paz-soldan/efit/158103/VARYPED_BIGJBOOT/ --name varyped_test2 --n -2 --rote -1 --eta -1 --simul multi_efit --freq 10

python extract_profiles_p_file.py --shot 158103 --time 12 --dir /u/paz-soldan/efit/158103/VARYPED_PPED/ --name varyped_IAEA --n -2 --rote -1 --eta -1 --simul multi_efit --freq 10

python/pyMARS
/u/haskeysr/tmp/shot156746_02113_betaN_ramp_carlos_prlV2
'''

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
parser.add_argument('--simul', type=str, help = 'Type of simulation - single, betaN_ramp, q95_scan, multi_efit', )
parser.add_argument('--freq', type=float, help = 'I-coil freq (Hz)', )

args = vars(parser.parse_args())
print args
arg_string = ' '.join(['--{} {}'.format(i,j) for i,j in args.iteritems()])
orig_call = "python extract_profiles_p_file.py {}".format(arg_string)

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

single_run = 1 if simul in ['single','multi_efit'] else 0
start_dir = os.getcwd()
if original_dir[-1]!='/': original_dir +='/'
mars_ind_dir = 'shot{}_{:05d}_{}_{}'.format(shot, time, simul, suffix)
efit_dir = '/u/haskeysr/efit/shot{}_{:05d}/'.format(shot, time)
efit_dir = '/u/haskeysr/efit/{}/'.format(mars_ind_dir,)
mars_dir = '/u/haskeysr/mars/{}/'.format(mars_ind_dir)
if efit_dir[-1]!='/':efit_dir+='/'
print efit_dir
os.system('mkdir {}'.format(efit_dir))
orig_list = os.listdir(original_dir)

if simul=='multi_efit':
    list_of_times = []
    for i in orig_list:
        if i.find(str(shot))>=0 and i.find('g')==0:
            list_of_times.append(int(i[i.find('.')+1:]))
            print i, list_of_times[-1]
else:
    list_of_times = [time]
print list_of_times


for cur_time in list_of_times:
    filt_list = []
    for i in orig_list:
        if i.find(str(shot))>=0 and i.find('{:05d}'.format(cur_time))>=0:
            filt_list.append(i)
    for i in filt_list: 
        j = i
        if i.find('_')>i.find(str(cur_time)):
            j = i[:i.find('_')]  
            ms = i[i.find('_'):]
        else:
            j = i
            ms = ''
        copy_string = 'cp {}{} {}{}'.format(original_dir, i, efit_dir,j)
        print copy_string
        os.system(copy_string)

#Extract the profiles - need some kind of if in here
for cur_time in list_of_times:
    extra = ''
    filename = '{}/p{}.{:05d}{}'.format(efit_dir, shot, cur_time, extra)
    final_names_link = ['PROFDEN', 'PROFTE', 'PROFTI', 'PROFROT']
    final_names = ['dne', 'dte', 'dti', 'dtrot']
    multipliers = [10**20,1,1,1]
    format = ['{:.4e}','{:.4f}','{:.4f}','{:.4f}']
    final_names = ['{}{}.{:05d}.dat'.format(i, shot, cur_time) for i in final_names]
    search_terms = ['ne','te','ti','omeg']
    def mod_data(in_list, mult, form):
        out_list = [' {}        {}\n'.format(len(in_list), '2')]
        for i in in_list:
            tmp = " ".join(i.split())
            tmp = tmp.split()
            val = form.format(float(tmp[1])*mult)
            out_list.append('{:.4f}        {}\n'.format((float(tmp[0]))**0.5, val))
        return out_list

    if os.path.exists(filename): 
        print 'using p file'
        use_p_file = True
    else:
        print 'using profile files'
        for i in final_names_link:
            if os.path.exists('{}/{}'.format(original_dir, i)):
                pass
            else:
                print 'profile file: {} does not exist'.format(i)
                raise ValueError()
        use_p_file = False

    if use_p_file: 
        with file(filename,'r') as handle:lines = handle.readlines()
        os.chdir(efit_dir)
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
    else:
        for i in final_names_link:
            os.system('cp {}/{} {}/'.format(original_dir, i, efit_dir))

if abs(int(n))==2:
    I_coil_current_string = '1.,-0.5,-0.5,1.,-0.5,-0.5'
elif abs(int(n))==3:
    I_coil_current_string = '1.,-1.,1.,-1.,1.,-1.'
else:
    raise(ValueError('Unknown current string'))

os.system('mkdir {}'.format(mars_dir))
input_template = '/u/haskeysr/mars/templates/BetaRampTemplate.cfg'
with file(input_template,'r') as file_handle: inp_temp = file_handle.readlines()
mods = [['project_name','project_name = {}\n'.format(mars_ind_dir)],
        ['efit_file_location','efit_file_location = {}\n'.format(efit_dir)],
        ['profile_file_location','profile_file_location = {}\n'.format(efit_dir)],
        ['<<RNTOR>>', '<<RNTOR>> : {}\n'.format(n)],
        ['<<ROTE>>', '<<ROTE>> : {}\n'.format(args['rote'])],
        ['<<ETA>>', '<<ETA>> : {}\n'.format(args['eta'])],
        ['I_coil_frequency', 'I_coil_frequency = {}\n'.format(args['freq'])],
        ['single_runthrough','single_runthrough = {}\n'.format(single_run)],
        ['I_coil_current','I_coil_current = {}\n'.format(I_coil_current_string)]]
if simul=='betaN_ramp':
    print 'Adding new mods for betaN_ramp'
    mods.append(['CORSICA_template_name', 'CORSICA_template_name = equal_spacing_pt1.bas\n'])
    mods.append(['CORSICA_template_name2', 'CORSICA_template_name2 = equal_spacing_pt2.bas\n'])
    mods.append(['q_mult_min', 'q_mult_min = 1\n'])
    mods.append(['q_mult_max', 'q_mult_max = 1\n'])
    mods.append(['q_mult_number', 'q_mult_number = 1\n'])
    mods.append(['p_mult_number', 'p_mult_number = 20\n'])
    mods.append(['<<stab_mode>>', '<<stab_mode>> : {}\n'.format(abs(n))])
    mods.append(['<<call_dcon>>', '<<call_dcon>> : 1\n'])
    mods.append(['CORSICA_workers','CORSICA_workers = 5\n'])
elif simul=='q95_scan':
    mods.append(['CORSICA_template_name', 'CORSICA_template_name = q95_scan.bas\n'])
    mods.append(['q_mult_number', 'q_mult_number = 10\n'])
    mods.append(['<<q95_min>>', '<<q95_min>> : 3.\n'])
    mods.append(['<<q95_max>>', '<<q95_max>> : 5.\n'])
    mods.append(['<<min_bn_li>>', '<<min_bn_li>> : 0\n'])
    mods.append(['<<stab_mode>>', '<<stab_mode>> : {}\n'.format(abs(n))])
elif simul=='multi_efit':
    # Need to check which CORSICA template it is using....
    mods.append(['multiple_efits', 'multiple_efits = 1\n'])

for i in mods:
    print i
    for j in range(len(inp_temp)):
        if inp_temp[j].find(i[0])>=0:
            print 'found', inp_temp[j], i[0]
            inp_temp[j] = i[1]
            break
with file(mars_dir + 'input.cfg','w') as file_handle: file_handle.writelines(inp_temp)
with file(mars_dir + 'extract_profiles_p_file_call.txt','w') as file_handle: file_handle.writelines([orig_call])
print mars_dir
print mars_dir + 'input.cfg'
print 'cd {} && '.format(mars_dir) + ' nohup unbuffer run_pyMARS input.cfg &>log &'
os.chdir(start_dir)
