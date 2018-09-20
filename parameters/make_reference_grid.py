__author__ = 'marvinler'
import os
import sys
import numpy as np
from oct2py import octave
from pypownet import ARTIFICIAL_NODE_STARTING_STRING
import copy


def main(grid_path, output_file=None):
    mpc = octave.loadcase(grid_path, verbose=False)
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

    output_file = os.path.join(os.path.dirname(input_file), 'reference_grid.m') if output_file is None else output_file
    octave.savecase(output_file, mpc)
    return output_file

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python -m pypownet.%s caseZZZ.m' % os.path.basename(__file__)[:-3])
    input_file = sys.argv[1]
    output_file = main(input_file)
    print('created file', output_file)
