__author__ = 'marvinler'
import os
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

        mandatory_files = ['configuration.json',  # Simulator parameters config file
                           'reference_grid.m']  # Reference (and initial starting) grid
        mandatory_folders = ['planned_chronics/', 'realized_chronics/']
        for f in mandatory_files + mandatory_folders:
            if not os.path.exists(os.path.join(self.__parameters_path, f)):
                raise FileNotFoundError('Mandatory file/folder %s not found within %s' % (f, self.__parameters_path))

        format_path = lambda f: os.path.join(self.__parameters_path, f)
        self.reference_grid_path = format_path('reference_grid.m')
        self.configuration_path = format_path('configuration.json')
        self.planned_chronics_path = format_path('planned_chronics/')
        self.realized_chronics_path = format_path('realized_chronics/')

        with open(self.configuration_path, 'r') as f:
            self.simulator_configuration = json.load(f)

    def get_reference_grid_path(self):
        return self.reference_grid_path

    def get_planned_chronics_path(self):
        return self.planned_chronics_path

    def get_realized_chronics_path(self):
        return self.realized_chronics_path

    def _get_mode(self):
        mode = self.simulator_configuration['mode'].lower()
        if mode not in ['ac', 'dc']:
            raise ValueError('mode value in configuration file should be either "AC" or "DC"')
        return mode

    def is_dc_mode(self):
        return self._get_mode() == 'DC'.lower()

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

    def get_grid_case(self):
        return self.simulator_configuration['grid_case']
