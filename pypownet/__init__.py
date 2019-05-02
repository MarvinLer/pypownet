__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import os
import logging

logger = logging.getLogger('pypownet.' + __file__)

ARTIFICIAL_NODE_STARTING_STRING = '666'

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def configure_matpower():
    from oct2py import octave

    # Add matpower to octave path
    mp_path_config = os.path.join(root_path, 'matpower_path.config')
    if not os.path.exists(mp_path_config):
        raise FileNotFoundError('The matpower path configuration file is not found at', mp_path_config)
    with open(mp_path_config, 'r') as f:
        relative_path = f.read().splitlines()[0]

    matpower_path = os.path.abspath(os.path.join(root_path, relative_path))
    # Check that the matpower path does exist
    if not os.path.exists(matpower_path):
        raise FileNotFoundError('Matpower folder %s not found' % matpower_path)

    # Add matpower and some of its subfolders to octave workspace
    octave.addpath(os.path.abspath(matpower_path))
    #octave.addpath(os.path.abspath(os.path.join(matpower_path, 't/')))
    octave.addpath(os.path.abspath(os.path.join(matpower_path, 'most/')))
    #octave.addpath(os.path.abspath(os.path.join(matpower_path, os.path.join('most/', 't/'))))
    octave.addpath(os.path.abspath(os.path.join(matpower_path, 'lib/')))


#from gym.envs.registration import register

# Seek for parameters folders
#parameters_folders = [os.path.abspath(os.path.join('parameters/', f)) for f in os.listdir('parameters/')
#                      if not os.path.isfile(f)]
#for parameters_folder in parameters_folders:
#    register(
#        id='pypownet_%s-v1' % os.path.basename(parameters_folder),
#        entry_point='pypownet.environment:RunEnv',
#        kwargs={'parameters_folder': parameters_folder}
#    )


def register_gym_envs():
    try:
        from gym.envs.registration import register

        # Seek for parameters folders
        parameters_folders = [os.path.abspath(os.path.join('parameters/', f)) for f in os.listdir('parameters/')
                              if not os.path.isfile(f)]
        for parameters_folder in parameters_folders:
            register(
                id='pypownet_%s-v1' % os.path.basename(parameters_folder),
                entry_point='pypownet.environment:RunEnv',
                kwargs={'parameters_folder': parameters_folder}
            )
    except ImportError:
        pass
    except:
        raise
