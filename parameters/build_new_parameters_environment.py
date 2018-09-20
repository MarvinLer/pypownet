__author__ = 'marvinler'
import os
import shutil
from parameters.make_reference_grid import main as make_ref_grid

ieee_path = input('enter the path to the IEEE grid case (.m) to be processed into a reference grid:')
if not os.path.exists(ieee_path):
    raise FileNotFoundError
grid_case = os.path.basename(ieee_path)[4:-2]
env_name = input('enter the name of the parameters environment (press ENTER for "case%s"):' % grid_case)
if env_name == '':
    env_name = 'case%s' % grid_case
dest_folder = input('enter destination folder (press ENTER for "%s")' % os.path.abspath('parameters/'))
if dest_folder == '':
    dest_folder = 'parameters/'
dest_folder = os.path.abspath(dest_folder)
if not os.path.exists(dest_folder):
    raise FileNotFoundError

env_path = os.path.join(dest_folder, env_name)
if os.path.exists(env_path):
    raise ValueError('Parameters environment %s already exists' % env_path)

os.makedirs(env_path)
n_levels = input('enter the number of levels (press ENTER for 1; you can create more later, see doc)')
if n_levels == '':
    n_levels = 1
else:
    n_levels = int(n_levels)

custom_reward = input('do you plan to use a custom reward ? (y/N)')
if custom_reward.lower() == 'y':
    custom_reward = True
else:
    custom_reward = False
if custom_reward:
    reward_signal_template = '''from pypownet.reward_signal import RewardSignal


class CustomRewardSignal(RewardSignal):
    def __init__(self):
        super().__init__()

    def compute_reward(self, observation, action, flag):
        return 0.
'''
    open(os.path.join(env_path, 'reward_signal.py'), 'w').write(reward_signal_template)

configuration_file_template = '''{
    "game_mode"                                       : "soft",

    "loadflow_mode"                                   : "AC",
    "max_seconds_per_timestep"                        : 1.0,

    "hard_overflow_coefficient"                       : 1.5,
    "n_timesteps_hard_overflow_is_broken"             : 10,

    "n_timesteps_consecutive_soft_overflow_breaks"    : 3,
    "n_timesteps_soft_overflow_is_broken"             : 5,

    "n_timesteps_horizon_maintenance"                 : 20,

    "max_number_prods_game_over"                      : 1,
    "max_number_loads_game_over"                      : 0
}
'''

# Construct each level
of = None
for i in range(n_levels):
    level_path = os.path.join(env_path, 'level%d' % i)
    os.makedirs(level_path)
    open(os.path.join(level_path, 'configuration.json'), 'w').write(configuration_file_template)
    os.makedirs(os.path.join(level_path, 'chronics/'))
    if i == 0:
        of = make_ref_grid(ieee_path, output_file=os.path.join(level_path, 'reference_grid.m'))
    else:
        shutil.copy(of, os.path.join(level_path, 'reference_grid.m'))

print()
print('Successively converted IEEE into reference grid')
print('Create new environment parameters at', env_path)
print('Chronics need to be added by hand. Please refer to the documentation for further information.')
