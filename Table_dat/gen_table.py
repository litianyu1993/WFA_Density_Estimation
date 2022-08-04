import numpy as np

with open('wfa_table.dat', 'a') as the_file:
    for exp_data in ['covType', 'elec' ,'outdoor', 'poker',
                     'rialto', 'weather', 'covTypetwoclasses', 'sea', 'mixeddrift', 'hyperplane', 'chess', 'outdoor', 'rialtotwoclasses',
                     'pokertwoclasses', 'interRBF', 'movingRBF', 'border', 'COIL', 'overlap']:
        command = f'python stream_sgd_wfa_windowed.py --exp_data {exp_data} --method wfa\n'
        the_file.write(command)

with open('gru_table.dat', 'a') as the_file:
    for exp_data in ['covType', 'elec' ,'outdoor', 'poker',
                     'rialto', 'weather', 'covTypetwoclasses', 'sea', 'mixeddrift', 'hyperplane', 'chess', 'outdoor', 'rialtotwoclasses',
                     'pokertwoclasses', 'interRBF', 'movingRBF', 'border', 'COIL', 'overlap']:
        command = f'python stream_sgd_wfa_windowed.py --exp_data {exp_data} --method gru\n'
        the_file.write(command)

with open('lstm_table.dat', 'a') as the_file:
    for exp_data in ['covType', 'elec' ,'outdoor', 'poker',
                     'rialto', 'weather', 'covTypetwoclasses', 'sea', 'mixeddrift', 'hyperplane', 'chess', 'outdoor', 'rialtotwoclasses',
                     'pokertwoclasses', 'interRBF', 'movingRBF', 'border', 'COIL', 'overlap']:
        command = f'python stream_sgd_wfa_windowed.py --exp_data {exp_data} --method lstm\n'
        the_file.write(command)