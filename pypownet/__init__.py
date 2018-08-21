__author__ = 'marvinler'
from oct2py import octave
import os


ARTIFICIAL_NODE_STARTING_STRING = '666'

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#os.chdir(root_path)

# Add matpower to octave path
mp_path_config = os.path.join(root_path, 'matpower_path.config')
if not os.path.exists(mp_path_config):
    raise FileNotFoundError('The matpower path configuration file is not found at', mp_path_config)
with open(mp_path_config, 'r') as f:
    relative_path = f.read()

matpower_path = os.path.abspath(relative_path)
# Check that the matpower path does exist
if not os.path.exists(matpower_path):
    raise FileNotFoundError('Matpower folder %s not found')

# Add matpower and some of its subfolders to octave workspace
octave.addpath(os.path.abspath(matpower_path))
#octave.addpath(os.path.abspath(os.path.join(matpower_path, 't/')))
octave.addpath(os.path.abspath(os.path.join(matpower_path, 'most/')))
#octave.addpath(os.path.abspath(os.path.join(matpower_path, os.path.join('most/', 't/'))))
octave.addpath(os.path.abspath(os.path.join(matpower_path, 'lib/')))

try:
    from gym.envs.registration import register

    register(
        id='PowNet118-v1',
        entry_point='pypownet.env:RunEnv',
        kwargs={'grid_case': 118}
    )

    register(
        id='PowNet14-v1',
        entry_point='pypownet.env:RunEnv',
        kwargs={'grid_case': 14}
    )
except ImportError:
    pass
