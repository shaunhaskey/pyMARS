#!/usr/bin/env Python
import os

os.chdir('/u/haskeysr/matlab/RZplot3/')
file = open('mat_commands.txt','w')

mat_commands = 'close all;clear all;\n'
mat_commands += 'cd /u/haskeysr/matlab/RZplot3/\n'
mat_commands += "diary('testing_output')\n"
mat_commands += "diary on\n"
mat_commands += 'Run MacMain_Shaun2\n'
mat_commands += "diary off\n"
mat_commands += 'quit\n'
file.write(mat_commands)
file.close

os.system('matlab -nodesktop -nodisplay < mat_commands.txt')


print 'hello'
