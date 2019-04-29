import os
import numpy as np


files = [f for f in os.listdir('.') if f.endswith('csv') and f not in ['_N_datetimes.csv', '_N_imaps.csv', '_N_simu_ids.csv']]
assert len(files) == 10, 'expected 10 files'

os.makedirs('output/')


def replace_values(filename, new_values):
    array = np.genfromtxt(filename, delimiter=';', dtype=None)
    array[1:] = list(map(str, new_values))
    np.savetxt('output/' + filename, array, delimiter=';', fmt='%s')
    np.savetxt('output/' + filename[:-4] + '_planned.csv', array, delimiter=';', fmt='%s')


replace_values('_N_prods_p.csv', [232.4, 40, 0, 0, 0])
replace_values('_N_prods_v.csv', [106, 104.5, 101, 107, 109])
replace_values('_N_loads_p.csv', [21.7, 94.2, 47.8, 7.6, 11.2, 29.5, 9, 3.5, 6.1, 13.5, 14.9])
replace_values('_N_loads_q.csv', [12.7, 19, -3.9, 1.6, 7.5, 16.6, 5.8, 1.8, 1.6, 5.8, 5])
replace_values('hazards.csv', [0]*20)
replace_values('maintenance.csv', [0]*20)
