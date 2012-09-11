import time
import pickle


#function to read in a stab_results file and generate a dictionary containing
#information about the equilibria

def read_stab_results(file_location):
    stab_setup_contents = open(file_location).read()
    stab_lines = stab_setup_contents.split('\n')
    line = 0

    dictionary_list = {}
    
    #skip lines at the start with ;
    while stab_lines[line][0] == ';':
        line += 1

    var_names = []

    #extract variable names
    stab_lines[line] = stab_lines[line].lstrip(' ').rstrip('\n').rstrip(' ')
    stab_lines[line] = stab_lines[line]+ ' '

    while len(stab_lines[line]) >= 1:
        end = stab_lines[line].find(' ')
        var_names.append(stab_lines[line][0:end])
        stab_lines[line] = stab_lines[line].lstrip(var_names[-1]).lstrip(' ')

    #extract equilibrium run values
    line += 1
    values = []
    item = 1
    while (line< len(stab_lines)) and (len(stab_lines[line])>1):
        current_value = []
        stab_lines[line] = stab_lines[line].lstrip(' ').rstrip('\n').rstrip(' ')
        stab_lines[line] = stab_lines[line] + ' ' #pad so while loop works
        while len(stab_lines[line]) >= 1:
            end = stab_lines[line].find(' ')
            current_value.append(stab_lines[line][0:end])
            stab_lines[line] = stab_lines[line].lstrip(current_value[-1]).lstrip(' ')
        #index each eq by a tuple (pmult,qmult)
        #item = (current_value[0],current_value[1])

        dictionary_list[item]={}
        for i in range(0,len(current_value)):
            dictionary_list[item][var_names[i]]=current_value[i]
        values.append(current_value)
        line += 1
        item += 1
    return dictionary_list
