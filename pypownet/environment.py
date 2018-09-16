__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import numpy as np

import pypownet.game
import pypownet.grid


class IllegalActionException(Exception):
    def __init__(self, text, illegal_lines_reconnections, *args):
        super(IllegalActionException, self).__init__(*args)
        self.text = text
        self.illegal_lines_reconnections = illegal_lines_reconnections


class ActionSpace(object):
    def __init__(self, number_generators, number_consumers, number_power_lines):
        self._n_prods = number_generators
        self._n_loads = number_consumers
        self._n_lines = number_power_lines

        self.topological_subaction_length = self._n_prods + self._n_loads + 2 * self._n_lines
        self.lines_status_subaction_length = self._n_lines

        # In an environment, the actions have fixed shape: self.action_length is expected action list size
        self.action_length = self.topological_subaction_length + self.lines_status_subaction_length

    def get_do_nothing_action(self):
        """ Creates and returns an action equivalent to a do-nothing: all of the activable switches are 0 i.e.
        not activated.

        :return: an instance of Action that is equivalent to an action doing nothing
        """
        return pypownet.game.Action(topological_subaction=np.zeros(self.topological_subaction_length),
                                    lines_status_subaction=np.zeros(self.lines_status_subaction_length))

    def array_to_action(self, array):
        """ Converts and returns an Action from a array-object (e.g. list, numpy arrays).

        :param array: array-style object
        :return: an instance of Action equivalent to input action :raise ValueError: the input array is not of the
        same length than the expected action (self.action_length)
        """
        if len(array) != self.action_length:
            raise ValueError('Expected binary array of length %d, got %d' % (self.action_length, len(array)))

        topological_subaction = array[:self.topological_subaction_length]
        lines_status_subaction = array[self.topological_subaction_length:]

        return pypownet.game.Action(topological_subaction, lines_status_subaction)

    def verify_action_shape(self, action):
        # Action of None is equivalent to no action
        if action is None:
            raise ValueError('Expected binary array of length %d, got None' % self.action_length)

        # If the input action is not of class Action, try to format it into Action (action must be array-like)
        if not isinstance(action, pypownet.game.Action):
            try:
                formatted_action = self.array_to_action(action)
            except ValueError as e:
                raise e
        else:
            formatted_action = action

        topological_subaction_length, lines_status_subaction_length = formatted_action.__len__(do_sum=False)

        if topological_subaction_length and topological_subaction_length != self.topological_subaction_length:
            raise ValueError('Expected topological subaction of size %d, got %d' % (
                self.topological_subaction_length, topological_subaction_length))

        if lines_status_subaction_length and lines_status_subaction_length != self.lines_status_subaction_length:
            raise ValueError('Expected lines status subaction of size %d, got %d' % (
                self.lines_status_subaction_length, lines_status_subaction_length))

        return formatted_action


class ObservationSpace(object):
    def __init__(self, number_generators, number_consumers, number_power_lines):
        self.number_productions = number_generators
        self.number_loads = number_consumers
        self.number_power_lines = number_power_lines

        self.grid_number_of_elements = self.number_productions + self.number_loads + 2 * self.number_power_lines


class Observation(object):
    """ The class State is a container for all the values representing the state of a given grid at a given time. It
    contains the following values:
    * The active and reactive power values of the loads
    * The active power values and the voltage setpoints of the productions
    * The values of the power through the lines: the active and reactive values at the origin/extremity of the
    lines as well as the lines capacity usage
    * The exhaustive topology of the grid, as a stacked vector of one-hot vectors
    """

    def __init__(self, active_loads, reactive_loads, voltage_loads, active_productions, reactive_productions,
                 voltage_productions, active_flows_origin, reactive_flows_origin, voltage_flows_origin,
                 active_flows_extremity, reactive_flows_extremity, voltage_flows_extremity, ampere_flows,
                 thermal_limits, lines_status, are_isolated_loads, are_isolated_prods, loads_substations_ids,
                 prods_substations_ids, lines_or_substations_ids, lines_ex_substations_ids,
                 timesteps_before_lines_reconnectable, timesteps_before_planned_maintenance, planned_active_loads,
                 planned_reactive_loads, planned_active_productions, planned_voltage_productions, date,
                 prods_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes):
        # Loads related state values
        self.active_loads = active_loads
        self.reactive_loads = reactive_loads
        self.voltage_loads = voltage_loads
        self.are_loads_cut = are_isolated_loads
        self.loads_substations_ids = loads_substations_ids
        self.loads_nodes = loads_nodes

        # Productions related state values
        self.active_productions = active_productions
        self.reactive_productions = reactive_productions
        self.voltage_productions = voltage_productions
        self.are_productions_cut = are_isolated_prods
        self.productions_substations_ids = prods_substations_ids
        self.productions_nodes = prods_nodes

        # Origin flows related state values
        self.active_flows_origin = active_flows_origin
        self.reactive_flows_origin = reactive_flows_origin
        self.voltage_flows_origin = voltage_flows_origin
        self.lines_or_substations_ids = lines_or_substations_ids
        self.lines_or_nodes = lines_or_nodes
        # Extremity flows related state values
        self.active_flows_extremity = active_flows_extremity
        self.reactive_flows_extremity = reactive_flows_extremity
        self.voltage_flows_extremity = voltage_flows_extremity
        self.lines_ex_substations_ids = lines_ex_substations_ids
        self.lines_ex_nodes = lines_ex_nodes

        # Ampere flows and thermal limits
        self.ampere_flows = ampere_flows
        self.thermal_limits = thermal_limits
        self.lines_status = lines_status

        # Per-line timesteps to wait before the line is full repaired, after being broken by cascading failure,
        # random hazards, or shut down for maintenance (e.g. painting)
        self.timesteps_before_lines_reconnectable = timesteps_before_lines_reconnectable
        self.timesteps_before_planned_maintenance = timesteps_before_planned_maintenance

        # Planned injections for the next timestep
        self.planned_active_loads = planned_active_loads
        self.planned_reactive_loads = planned_reactive_loads
        self.planned_active_productions = planned_active_productions
        self.planned_voltage_productions = planned_voltage_productions

        self.datetime = date

    def as_dict(self):
        return self.__dict__

    def as_array(self):
        return np.concatenate((
            self.loads_substations_ids, self.active_loads, self.reactive_loads, self.voltage_loads, self.are_loads_cut,
            self.planned_active_loads, self.planned_reactive_loads, self.loads_nodes,

            self.productions_substations_ids, self.active_productions, self.reactive_productions,
            self.voltage_productions, self.are_productions_cut,
            self.planned_active_productions, self.planned_voltage_productions, self.productions_nodes,

            self.lines_or_substations_ids, self.active_flows_origin, self.reactive_flows_origin,
            self.voltage_flows_origin, self.lines_or_nodes,

            self.lines_ex_substations_ids, self.active_flows_extremity, self.reactive_flows_extremity,
            self.voltage_flows_extremity, self.lines_ex_nodes,

            self.ampere_flows, self.thermal_limits, self.lines_status, self.timesteps_before_lines_reconnectable,
            self.timesteps_before_planned_maintenance))

    def __str__(self):
        date_str = 'date:' + self.datetime.strftime("%Y-%m-%d %H:%M")

        def _tabular_prettifier(matrix, formats, column_widths):
            """ Used for printing well shaped tables within terminal and log files
            """
            res = ''

            matrix_str = [[fmt.format(v) for v, fmt in zip(line, formats)] for line in matrix]
            for line in matrix_str:
                line_str = ' |' + ' |'.join(' ' * (w - 1 - len(v)) + v for v, w in zip(line, column_widths)) + ' |\n'
                res += line_str

            return res

        # Prods
        headers = ['Sub. #', 'Node #', 'OFF', 'P', 'Q', 'V', 'P', 'V']
        content = np.vstack((self.productions_substations_ids,
                             self.productions_nodes,
                             self.are_productions_cut,
                             self.active_productions,
                             self.reactive_productions,
                             self.voltage_productions,
                             self.planned_active_productions,
                             self.planned_voltage_productions)).T
        # Format header then add matrix as string
        n_symbols = 67
        column_widths = [8, 8, 5, 8, 7, 7, 8, 7]
        prods_header = ' ' + '=' * n_symbols + '\n' + \
                       ' |' + ' ' * ((n_symbols - 13) // 2) + 'PRODUCTIONS' + ' ' * (
                           (n_symbols - 12) // 2) + '|' + '\n' + \
                       ' ' + '=' * n_symbols + '\n'
        prods_header += ' |                 | is  |         Current        | Previsions t+1 |\n'
        prods_header += ' |' + ' |'.join(
            ' ' * (w - 1 - len(v)) + v for v, w in zip(headers, column_widths)) + ' |\n'
        prods_str = prods_header + _tabular_prettifier(content,
                                                       formats=['{:.0f}', '{:.0f}', '{:.0f}', '{:.1f}', '{:.2f}',
                                                                '{:.2f}', '{:.2f}', '{:.2f}'],
                                                       column_widths=column_widths)

        # Loads
        n_symbols = 68
        column_widths = [8, 8, 5, 8, 7, 7, 8, 8]
        headers = ['Sub. #', 'Node #', 'OFF', 'P', 'Q', 'V', 'P', 'Q']
        content = np.vstack((self.loads_substations_ids,
                             self.loads_nodes,
                             self.are_loads_cut,
                             self.active_loads,
                             self.reactive_loads,
                             self.voltage_loads,
                             self.planned_active_loads,
                             self.planned_reactive_loads)).T
        loads_header = ' ' + '=' * n_symbols + '\n' + \
                       ' |' + ' ' * ((n_symbols - 6) // 2) + 'LOADS' + ' ' * ((n_symbols - 7) // 2) + '|' + '\n' + \
                       ' ' + '=' * n_symbols + '\n'
        loads_header += ' |                 | is  |         Current        | Previsions t+1  |\n'
        loads_header += ' |' + ' |'.join(
            ' ' * (w - 1 - len(v)) + v for v, w in zip(headers, column_widths)) + ' |\n'
        loads_str = loads_header + _tabular_prettifier(content,
                                                       formats=['{:.0f}', '{:.0f}', '{:.0f}', '{:.1f}', '{:.2f}',
                                                                '{:.2f}', '{:.1f}', '{:.2f}'],
                                                       column_widths=column_widths)
        # Concatenate both strings horizontally
        prods_lines = prods_str.splitlines()
        loads_lines = loads_str.splitlines()
        injections_str = ''
        for prod_line, load_line in zip(prods_lines, loads_lines[:len(prods_lines)]):
            injections_str += load_line + '          ' + prod_line + '\n'
        injections_str += '\n'.join(loads_lines[len(prods_lines):]) + '\n'

        # Lines
        headers = ['sub. #', 'node #', 'sub. #', 'node #', 'ON', 'P', 'Q', 'V', 'P', 'Q', 'V', 'Ampere', 'limits ',
                   'maintenance',
                   'reconnectable']
        column_widths = [8, 8, 8, 8, 4, 8, 7, 6, 8, 7, 6, 8, 9, 13, 15]
        content = np.vstack((self.lines_or_substations_ids,
                             self.lines_or_nodes,
                             self.lines_ex_substations_ids,
                             self.lines_ex_nodes,
                             self.lines_status,
                             self.active_flows_origin,
                             self.reactive_flows_origin,
                             self.voltage_flows_origin,
                             self.active_flows_extremity,
                             self.reactive_flows_extremity,
                             self.voltage_flows_extremity,
                             self.ampere_flows,
                             self.thermal_limits,
                             self.timesteps_before_planned_maintenance,
                             self.timesteps_before_lines_reconnectable)).T
        n_symbols = 139
        lines_header = ' ' + '=' * n_symbols + '\n' + \
                       ' |' + ' ' * ((n_symbols - 7) // 2) + 'LINES' + ' ' * ((n_symbols - 7) // 2) + '|' + '\n' + \
                       ' ' + '=' * n_symbols + '\n'
        lines_header += ' |       From      |       To        | is |    From injections    |     To injections     | ' \
                        'Flows  | Thermal |      Timesteps before       |\n'

        lines_header += ' |' + ' |'.join(
            ' ' * (w - 1 - len(v)) + v for v, w in zip(headers, column_widths)) + ' |\n'
        lines_str = lines_header + _tabular_prettifier(content,
                                                       formats=['{:.0f}', '{:.0f}', '{:.0f}', '{:.0f}', '{:.0f}',
                                                                '{:.1f}', '{:.1f}', '{:.2f}', '{:.1f}', '{:.1f}',
                                                                '{:.2f}', '{:.1f}', '{:.0f}', '{:.0f}', '{:.0f}'],
                                                       column_widths=column_widths)

        return '\n\n'.join([date_str, injections_str, lines_str])


class RunEnv(object):
    def __init__(self, parameters_folder, start_id=0, latency=None):
        """ Instantiate the game Environment based on the specified parameters. """
        # Instantiate game & action space
        self.game = pypownet.game.Game(parameters_folder=parameters_folder, start_id=start_id, latency=latency)
        self.action_space = ActionSpace(*self.game.get_number_elements())
        self.observation_space = ObservationSpace(*self.game.get_number_elements())

        # Reward hyperparameters
        self.multiplicative_factor_line_usage_reward = -1.  # Mult factor for line capacity usage subreward
        self.additive_factor_distance_initial_grid = -.02  # Additive factor for each differed node in the grid
        self.additive_factor_load_cut = -self.observation_space.grid_number_of_elements / 10.  # Additive factor for each isolated load
        self.additive_factor_prod_cut = .5 * self.additive_factor_load_cut
        self.connexity_exception_reward = -self.observation_space.grid_number_of_elements  # Reward when the grid is not connexe
        # (at least two islands)
        self.loadflow_exception_reward = -self.observation_space.grid_number_of_elements  # Reward in case of loadflow software error

        self.illegal_action_exception_reward = -self.observation_space.grid_number_of_elements / 100.  # Reward in case of bad action shape/form

        # Action cost reward hyperparameters
        self.cost_line_switch = .1  # 1 line switch off or switch on
        self.cost_node_switch = 0.  # Changing the node on which an element is directly wired

        self.last_rewards = []

    def _get_obs(self):
        return self.game.export_observation()

    def _get_distance_reference_grid(self, observation):
        # Reference grid distance reward
        """ Computes the distance of the current observation with the reference grid (i.e. initial grid of the game).
        The distance is computed as the number of different nodes on which two identical elements are wired. For
        instance, if the production of first current substation is wired on the node 1, and the one of the first initial
        substation is wired on the node 0, then their is a distance of 1 (there are different) between the current and
        reference grid (for this production). The total distance is the sum of those values (0 or 1) for all the
        elements of the grid (productions, loads, origin of lines, extremity of lines).

        :return: the number of different nodes between the current topology and the initial one
        """
        initial_topology = np.asarray(self.game.get_initial_topology())
        current_topology = np.concatenate((observation.productions_nodes, observation.loads_nodes,
                                           observation.lines_or_nodes, observation.lines_ex_nodes))

        return np.sum((initial_topology != current_topology))  # Sum of nodes that are different

    def _get_action_cost(self, action):
        # Action cost reward: compute the number of line switches, node switches, and return the associated reward
        """ Compute the >=0 cost of an action. We define the cost of an action as the sum of the cost of node-splitting
        and the cost of lines status switches. In short, the function sums the number of 1 in the action vector, since
        they represent activation of switches. The two parameters self.cost_node_switch and self.cost_line_switch
        control resp the cost of 1 node switch activation and 1 line status switch activation.

        :param action: an instance of Action or a binary numpy array of length self.action_space.n
        :return: a >=0 float of the cost of the action
        """
        if action is None:
            return 0.

        # Computes the number of activated switches of the action
        number_node_switches = np.sum(action.get_lines_status_subaction())
        number_line_switches = np.sum(action.get_topological_subaction())
        action_cost = self.cost_node_switch * number_node_switches + self.cost_line_switch * number_line_switches
        return action_cost

    def _get_lines_capacity_usage(self, observation):
        ampere_flows = observation.ampere_flows
        thermal_limits = observation.thermal_limits
        lines_capacity_usage = np.divide(ampere_flows, thermal_limits)
        return lines_capacity_usage

    def get_reward(self, observation, action, flag=None):
        # First, check for flag raised during step, as they indicate errors from grid computations
        if flag is not None:
            if isinstance(flag, pypownet.game.NoMoreScenarios):
                reward_aslist = [0, 0, 0, 0, 0]
            elif isinstance(flag, pypownet.grid.DivergingLoadflowException):
                reward_aslist = [0., 0., -self._get_action_cost(action), self.loadflow_exception_reward, 0.]
            elif isinstance(flag, IllegalActionException):
                # If some broken lines are attempted to be switched on, put the switches to 0, and add penalty to
                # the reward consequent to the newly submitted action
                reward_aslist = self.get_reward(observation, action, flag=None)
                reward_aslist[2] += self.illegal_action_exception_reward
            else:  # Should not happen
                raise flag
        else:
            # Load cut reward
            number_cut_loads = sum(observation.are_loads_cut)
            load_cut_reward = self.additive_factor_load_cut * number_cut_loads

            # Prod cut reward
            number_cut_prods = sum(observation.are_productions_cut)
            prod_cut_reward = self.additive_factor_prod_cut * number_cut_prods

            # Reference grid distance reward
            reference_grid_distance = self._get_distance_reference_grid(observation)
            reference_grid_distance_reward = self.additive_factor_distance_initial_grid * reference_grid_distance

            # Action cost reward: compute the number of line switches, node switches, and return the associated reward
            action_cost_reward = -self._get_action_cost(action)

            # The line usage subreward is the sum of the square of the lines capacity usage
            lines_capacity_usage = self._get_lines_capacity_usage(observation)
            line_usage_reward = self.multiplicative_factor_line_usage_reward * np.sum(np.square(lines_capacity_usage))

            # Format reward
            reward_aslist = [load_cut_reward, prod_cut_reward, action_cost_reward, reference_grid_distance_reward,
                             line_usage_reward]

        self.last_rewards = reward_aslist

        return reward_aslist

    def step(self, action, do_sum=True):
        """ Performs a game step given an action. The as list pattern is:
        load_cut_reward, prod_cut_reward, action_cost_reward, reference_grid_distance_reward, line_usage_reward
        """
        # First verify that the action is in expected condition: one array (or list) of expected size of 0 or 1
        try:
            submitted_action = self.action_space.verify_action_shape(action)
        except IllegalActionException as e:
            raise e

        observation, reward_flag, done = self.game.step(submitted_action)

        reward_aslist = self.get_reward(observation=observation, action=action, flag=reward_flag)

        return observation, sum(reward_aslist) if do_sum else reward_aslist, done, reward_flag

    def simulate(self, action=None, do_sum=True):
        """ Computes the reward of the simulation of action to the current grid. """
        # First verify that the action is in expected condition: one array (or list) of expected size of 0 or 1
        try:
            to_simulate_action = self.action_space.verify_action_shape(action)
        except IllegalActionException as e:
            raise e

        observation, reward_flag, done = self.game.simulate(to_simulate_action)

        reward_aslist = self.get_reward(observation=observation, action=action, flag=reward_flag)

        return sum(reward_aslist) if do_sum else reward_aslist

    def reset(self, restart=True):
        # Reset the grid overall topology
        self.game.reset(restart=restart)
        return self._get_obs()

    def render(self, mode='human', close=False, game_over=False):
        if mode == 'human':
            self.game._render(self.last_rewards, close, game_over=game_over)
        else:
            raise ValueError("Unsupported render mode: " + mode)

    def get_current_scenario_id(self):
        return self.game.get_current_timestep_id()


OBSERVATION_MEANING = {
    'active_productions': 'Real power produced by the generators of the grid (MW).',
    'active_loads': 'Real power consumed by the demands of the grid (MW).',
    'active_flows_origin': 'Real power flowing through the origin part of the lines (MW).',
    'active_flows_extremity': 'Real power flowing through the extremity part of the lines (MW).',

    'reactive_productions': 'Reactive power produced by the generators of the grid (Mvar).',
    'reactive_loads': 'Reactive power consumed by the demands of the grid (Mvar).',
    'reactive_flows_origin': 'Reactive power flowing through the origin part of the lines (Mvar).',
    'reactive_flows_extremity': 'Reactive power flowing through the extremity part of the lines (Mvar).',

    'voltage_productions': 'Voltage magnitude of the generators of the grid (per-unit V).',
    'voltage_loads': 'Voltage magnitude of the demands of the grid (per-unit V).',
    'voltage_flows_origin': 'Voltage magnitude of the origin part of the lines (per-unit V).',
    'voltage_flows_extremity': 'Voltage magnitude of the extremity part of the lines (per-unit V).',

    'ampere_flows': 'Current value of the flow within lines (A); fixed throughout a line.',
    'thermal_limits': 'Nominal thermal limit of the power lines (actually A).',
    'are_loads_cut': 'Mask whether the consumers are isolated (1) from the rest of the network.',
    'are_prods_cut': 'Mask whether the productors are isolated (1) from the rest of the network.',

    'prods_substations_ids': 'ID of the substation on which the productions (generators) are wired.',
    'loads_substations_ids': 'ID of the substation on which the loads (consumers) are wired.',
    'lines_or_substations_ids': 'ID of the substation on which the lines origin are wired.',
    'lines_ex_substations_ids': 'ID of the substation on which the lines extremity are wired.',

    'lines_status': 'Mask whether the lines are switched ON (1) or switched OFF (0).',
    'timesteps_before_lines_reconnectable': 'Number of timesteps to wait before a line is switchable ON.',
    'timesteps_before_planned_maintenance': 'Number of timesteps to wait before a line will be switched OFF for'
                                            'maintenance',

    'topology': 'The ID of the subnode, within a substation, on which the elements of the system are '
                'directly wired (0 or 1).',
}

# TODO
ACTION_MEANING = {

}
