__author__ = 'marvinler'
import os
import sys
import numpy as np
from pypownet import ARTIFICIAL_NODE_STARTING_STRING
import copy


def main(grid_path, output_file=None):
    if grid_path.endswith('.mat') or grid_path.endswith('.m'):
        from oct2py import octave as matpower
        loadflow_backend = matpower
        mpc = matpower.loadcase(grid_path, verbose=False)
    elif grid_path.endswith('.py'):
        import importlib
        pypower = importlib.import_module('pypower.api')
        loadflow_backend = pypower
        mpc = pypower.loadcase(grid_path, expect_gencost=False)
    else:
        raise ValueError('Loadflow computation backend with extension .{} is not supported; '
                         'options: matpower (.m), pypower (.py)'.format(grid_path.split('.')[-1]))

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

    preartificial_mpc = copy.deepcopy(mpc)

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

    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(input_file), 'reference_grid.' + grid_path.split('.')[-1])
            # os.path.basename('.'.join(grid_path.split('.')[:-1])+'_referenced.'+grid_path.split('.')[-1]))
    loadflow_backend.savecase(output_file, mpc)
    return output_file, preartificial_mpc


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python -m pypownet.%s filecase[.m|.mat|.py]' % os.path.basename(__file__)[:-3])
    input_file = sys.argv[1]
    output_file, mpc = main(input_file)
    print('created file', output_file)
