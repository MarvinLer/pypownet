__author__ = 'marvinler'
import os
import sys
import yaml
import logging
import importlib
from pypownet.reward_signal import RewardSignal


class Parameters(object):
    def __init__(self, parameters_folder, game_level):
        self.__parameters_path = os.path.abspath(parameters_folder)
        self.logger = logging.getLogger('pypownet.' + __name__)

        if not os.path.exists(self.__parameters_path):
            print('Parameters path %s does not exit' % self.__parameters_path)
            print('Located parameters folders:')
            available_folders = [f for f in os.listdir('parameters/') if not os.path.isfile(f)]
            if not available_folders:
                print('  no parameters environment found')
            else:
                print('  ' + '\n  '.join([os.path.join('parameters/', f) for f in available_folders]))
                print('Use -p PARAM_FOLDER with PARAM_FOLDER as one of the previous located folders; see their '
                      'configuration.json for more info\n\n')
            raise FileNotFoundError('folder %s does not exist' % os.path.abspath(parameters_folder))
        sys.path.append(self.__parameters_path)
        self.level_folder = os.path.join(self.__parameters_path, game_level)
        if not os.path.exists(self.level_folder):
            # Seek existing levels folders
            level_folders = [os.path.join(self.__parameters_path, d) for d in os.listdir(self.__parameters_path)
                             if not os.path.isfile(d)]
            raise FileNotFoundError('Game level folder %s does not exist; level folders found in %s: %s' % (
                game_level, self.__parameters_path, '[' + ', '.join(level_folders) + ']'))

        mandatory_files = ['configuration.yaml',  # Simulator parameters config file
                           'reference_grid.m']  # Reference (and initial starting) grid
        mandatory_folders = ['chronics/']
        for f in mandatory_files + mandatory_folders:
            if not os.path.exists(os.path.join(self.level_folder, f)):
                raise FileNotFoundError('Mandatory file/folder %s not found within %s' % (f, self.level_folder))

        format_path = lambda f: os.path.join(self.level_folder, f)
        self.reference_grid_matpower_path = format_path('reference_grid.m')
        self.reference_grid_pypower_path = format_path('reference_grid.py')
        self.chronics_path = format_path('chronics/')
        # Seek and read configuration file
        # self.configuration_path = format_path('configuration.json')
        # with open(self.configuration_path, 'r') as f:
        #     self.simulator_configuration = json.load(f)
        self.configuration_path = format_path('configuration.yaml')
        with open(self.configuration_path, 'r') as stream:
            try:
                self.simulator_configuration = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self.logger.error(exc)

        # Seek for custom reward signal file
        reward_signal_expected_path = os.path.join(self.__parameters_path, 'reward_signal.py')
        if not os.path.exists(reward_signal_expected_path):
            self.logger.error('/!\ Using default reward signal, as reward_signal.py file is not found')
            self.reward_signal_class = RewardSignal
        else:
            sys.path.append(os.path.dirname(reward_signal_expected_path))
            try:
                self.reward_signal_class = getattr(importlib.import_module('reward_signal'), 'CustomRewardSignal')
                self.logger.warning('Using custom reward signal CustomRewardSignal of file %s' %
                                 reward_signal_expected_path)
            except ImportError:
                self.logger.error('/!\ Using default reward signal, as reward_signal.py file is not found')
                self.reward_signal_class = RewardSignal

    def get_reward_signal_class(self):
        return self.reward_signal_class

    def get_reference_grid_path(self, loadflow_backend):
        if loadflow_backend == 'pypower':
            return self.reference_grid_pypower_path
        elif loadflow_backend == 'matpower':
            return self.reference_grid_matpower_path
        else:
            raise ValueError('should not happen')

    def get_chronics_path(self):
        return self.chronics_path

    def get_parameters_path(self):
        return self.__parameters_path

    def get_loadflow_backend(self):
        backend = self.simulator_configuration['loadflow_backend'].lower()
        if backend not in ['matpower', 'pypower']:
            raise ValueError('loadflow_backend %s is not currently supported; supported backend: "matpower", "pypower"')
        return backend

    def _get_game_mode(self):
        mode = self.simulator_configuration['game_mode'].lower()
        if mode not in ['hard', 'soft']:
            raise ValueError('loadflow_mode value in configuration file should be either "hard" or "soft"')
        return mode

    def _get_loadflow_mode(self):
        mode = self.simulator_configuration['loadflow_mode'].lower()
        if mode not in ['ac', 'dc']:
            raise ValueError('loadflow_mode value in configuration file should be either "AC" or "DC"')
        return mode

    def is_dc_mode(self):
        return self._get_loadflow_mode() == 'DC'.lower()

    def get_max_seconds_per_timestep(self):
        return self.simulator_configuration['max_seconds_per_timestep']

    def get_hard_overflow_coefficient(self):
        return self.simulator_configuration['hard_overflow_coefficient']

    def get_n_timesteps_hard_overflow_is_broken(self):
        return self.simulator_configuration['n_timesteps_hard_overflow_is_broken']

    def get_n_timesteps_consecutive_soft_overflow_breaks(self):
        return self.simulator_configuration['n_timesteps_consecutive_soft_overflow_breaks']

    def get_n_timesteps_soft_overflow_is_broken(self):
        return self.simulator_configuration['n_timesteps_soft_overflow_is_broken']

    def get_n_timesteps_horizon_maintenance(self):
        return self.simulator_configuration['n_timesteps_horizon_maintenance']

    def get_max_number_prods_game_over(self):
        return self.simulator_configuration['max_number_prods_game_over']

    def get_max_number_loads_game_over(self):
        return self.simulator_configuration['max_number_loads_game_over']

    def get_n_timesteps_actionned_line_reactionable(self):
        return self.simulator_configuration['n_timesteps_actionned_line_reactionable']

    def get_n_timesteps_actionned_node_reactionable(self):
        return self.simulator_configuration['n_timesteps_actionned_node_reactionable']

    def get_n_timesteps_pending_line_reactionable_when_overflowed(self):
        return self.simulator_configuration['n_timesteps_pending_line_reactionable_when_overflowed']

    def get_n_timesteps_pending_node_reactionable_when_overflowed(self):
        return self.simulator_configuration['n_timesteps_pending_node_reactionable_when_overflowed']

    def get_max_number_actionned_substations(self):
        return self.simulator_configuration['max_number_actionned_substations']

    def get_max_number_actionned_lines(self):
        return self.simulator_configuration['max_number_actionned_lines']

    def get_max_number_actionned_total(self):
        return self.simulator_configuration['max_number_actionned_total']

    def __str__(self):
        params_str = ['    ' + k + ': ' + str(v) for k, v in self.simulator_configuration.items()]
        max_length = max(list(map(len, params_str)))
        params_str = '\n'.join(params_str)

        hdrs = '  ' + '=' * max_length + '\n' + \
               ' ' * (max_length // 2 - 5) + 'GAME PARAMETERS\n' + \
               '  ' + '=' * max_length
        ftrs = '  ' + '=' * max_length
        return '\n'.join([hdrs, params_str, ftrs])
