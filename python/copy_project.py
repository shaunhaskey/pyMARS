'''
This script is supposed to copy pyMARS project into a new directory so that it can be run with some different settings. Assumes that you want to run fron CHEASE onwards. Should update it so that you can select whichever part of hte run you want to start from.

More or less done, however what is going on with MARS????

SRH: 7Oct2015
'''
import pickle, os

base_location = r'/u/haskeysr/mars/'
old_name = r'shot158103_03796_q95_scan_carlos_thetac0-003_100'
new_name = r'shot158103_03796_q95_scan_carlos_thetac0-003_100_high_res'
input_name = 'input.cfg'
old_loc = '{}{}/'.format(base_location,old_name)
new_loc = '{}{}/'.format(base_location,new_name)

execute = False
def run_cmd(cmd, execute = True):
    print cmd
    if execute:os.system(cmd)

#Make directory
cmd = 'mkdir {}'.format(new_loc)
run_cmd(cmd, execute = execute)

#copy input file and relevant pickle file from the last completed stage
new_input_cfg = '{}{}'.format(new_loc,input_name)
cmd = 'cp -a {}{} {}'.format(old_loc,input_name, new_input_cfg)
run_cmd(cmd, execute = execute)


a = '{}{}_setup_directories.pickle'.format(old_loc, old_name)
b = '{}{}_setup_directories.pickle'.format(new_loc, new_name)
new_pickle = b
cmd = 'cp -a {} {}'.format(a, b)
run_cmd(cmd, execute = execute)


#efit directory
a = '{}efit'.format(old_loc)
b = '{}efit'.format(new_loc)
cmd = 'cp -a {} {}'.format(a, b)
run_cmd(cmd, execute = execute)

#corsica directory
a = '{}corsica'.format(old_loc)
b = '{}corsica'.format(new_loc)
cmd = 'cp -a {} {}'.format(a, b)
run_cmd(cmd, execute = execute)

with file(new_input_cfg,'r') as filehandle:
    a = filehandle.readlines()
for i in range(len(a)):
    if a[i].find('project_name')>=0:
        a[i] = 'project_name = {}\n'.format(new_name)
    elif a[i].find('start_from_step')>=0:
        a[i] = 'start_from_step = 3\n'
with file(new_input_cfg,'w') as filehandle:
    filehandle.writelines(a)

project_dict = pickle.load(file(new_pickle,'r'))
for i in ['base_dir','shot_dir','thetac_dir']:
    print project_dict['details'][i]
    project_dict['details'][i] = new_loc
    print project_dict['details'][i]
print project_dict['details']['efit_dir']
project_dict['details']['efit_dir'] = new_loc + 'efit/'
print project_dict['details']['efit_dir']
for i in project_dict['sims'].keys():
    print i
    tmp = project_dict['sims'][i]['dir_dict']
    for j in tmp.keys():
        print tmp[j]
        tmp[j] = tmp[j].replace(old_name, new_name)
        print tmp[j]
    #print project_dict['sims'][1]['dir_dict']
pickle.dump(project_dict,file(new_pickle,'w'))
