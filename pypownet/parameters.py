__author__ = 'marvinler'
import os
import sys
import json


class Parameters(object):
    def __init__(self, parameters_folder):
        self.__parameters_path = os.path.abspath(parameters_folder)
        if not os.path.exists(self.__parameters_path):
            print('Parameters path %s does not exit' % self.__parameters_path)
            print('Located parameters folders:')
            print('  ' + '\n  '.join(
                os.path.join('parameters/', f) for f in os.listdir('parameters/') if not os.path.isfile(f)))
            print('Use -p PARAM_FOLDER with PARAM_FOLDER as one of the previous located folders; see their '
                  'configuration.json for more info\n\n')
            raise FileNotFoundError('folder %s does not exist' % os.path.abspath(parameters_folder))
        sys.path.append(self.__parameters_path)

        mandatory_files = ['configuration.json',  # Simulator parameters config file
                           'reference_grid.m']  # Reference (and initial starting) grid
        mandatory_folders = ['chronics/']
        for f in mandatory_files + mandatory_folders:
            if not os.path.exists(os.path.join(self.__parameters_path, f)):
                raise FileNotFoundError('Mandatory file/folder %s not found within %s' % (f, self.__parameters_path))

        format_path = lambda f: os.path.join(self.__parameters_path, f)
        self.reference_grid_path = format_path('reference_grid.m')
        self.configuration_path = format_path('configuration.json')
        self.chronics_path = format_path('chronics/')

        with open(self.configuration_path, 'r') as f:
            self.simulator_configuration = json.load(f)

        # Seek for custom reward signal file
        reward_signal_expected_path = os.path.join(self.__parameters_path, 'reward_signal.py')
        if os.path.exists(reward_signal_expected_path):
            sys.path.append(os.path.dirname(reward_signal_expected_path))
            try:
                from reward_signal import CustomRewardSignal
            except ImportError:
                raise ImportError('Expected reward_signal.py to have CustomRewardSignal class but not found.')
            self.reward_signal = CustomRewardSignal()
        else:
            self.reward_signal = None

    def get_reward_signal(self):
        return self.reward_signal

    def get_reference_grid_path(self):
        return self.reference_grid_path

    def get_chronics_path(self):
        return self.chronics_path

    def get_parameters_path(self):
        return self.__parameters_path

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

    def get_grid_case(self):
        return self.simulator_configuration['grid_case']

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
