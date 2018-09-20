__author__ = 'marvinler'
import os
import sys
import numpy as np
from oct2py import octave
from pypownet import ARTIFICIAL_NODE_STARTING_STRING
import copy

if len(sys.argv) != 2:
    print('usage: python -m pypownet.%s caseZZZ.m' % os.path.basename(__file__)[:-3])
input_file = sys.argv[1]
grid_case = os.path.basename(input_file)[4:-2]  # Seek for pattern casexxxxxx.m
#case_grid = '/home/marvin/Documents/pro/stagemaster_inria/pypownet/input/case%d.m' % grid_case

mpc = octave.loadcase(input_file, verbose=False)
mpc = {k: mpc[k] for k in ['bus', 'gen', 'branch', 'baseMVA', 'version']}  # Shrink useless parts

# Sort prods, buses, lines
mpc['gen'] = mpc['gen'][mpc['gen'][:, 0].argsort()]
mpc['bus'] = mpc['bus'][mpc['bus'][:, 0].argsort()]
mpc['branch'] = mpc['branch'][mpc['branch'][:, 1].argsort()]
mpc['branch'] = mpc['branch'][mpc['branch'][:, 0].argsort()]
# Rename ids such that they are all consecutive
substations_ids = copy.deepcopy(mpc['bus'][:, 0])
mpc['bus'][:, 0] = [i + 1 for i, _ in enumerate(substations_ids)]
mpc['gen'][:, 0] = [np.where(substations_ids == sub_id)[0][0] + 1 for sub_id in mpc['gen'][:, 0]]
mpc['branch'][:, 0] = [np.where(substations_ids == sub_id)[0][0] + 1 for sub_id in mpc['branch'][:, 0]]
mpc['branch'][:, 1] = [np.where(substations_ids == sub_id)[0][0] + 1 for sub_id in mpc['branch'][:, 1]]


# Add artificial nodes
artificial_buses = copy.deepcopy(mpc['bus'])
artificial_buses[:, 0] = list(map(float,
                                  list(map(lambda x: ARTIFICIAL_NODE_STARTING_STRING+str(x), artificial_buses[:, 0]))))
artificial_buses[:, 1] = 4
artificial_buses[:, 2] = 0.
artificial_buses[:, 3] = 0.
mpc['bus'] = np.concatenate((mpc['bus'], artificial_buses), axis=0)

# Ensure all prods and lines on
mpc['gen'][:, 7] = 1
mpc['branch'][:, 10] = 1

# Put all angles to 0
mpc['bus'][:, 8] = 0

# Change baseKV to default matpower = 100 if 0
if np.all(mpc['bus'][:, 9] == 0):
    mpc['bus'][:, 9] = 100

print(mpc['bus'].shape)

output_file = os.path.join(os.path.dirname(input_file), 'reference_grid%s.m' % grid_case)
octave.savecase(output_file, mpc)
print('created file', output_file)
