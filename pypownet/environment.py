__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import numpy as np
from enum import Enum

import pypownet.game
import pypownet.grid
import pypownet.reward_signal


class IllegalActionException(Exception):
    def __init__(self, text, illegal_lines_reconnections, *args):
        super(IllegalActionException, self).__init__(*args)
        self.text = text
        self.illegal_lines_reconnections = illegal_lines_reconnections


class ElementType(Enum):
    PRODUCTION = "production"
    CONSUMPTION = "consumption"
    ORIGIN_POWER_LINE = "origin of power line"
    EXTREMITY_POWER_LINE = "extremity of power line"


class ActionSpace(object):
    def __init__(self, number_generators, number_consumers, number_power_lines, substations_ids, prods_subs_ids,
                 loads_subs_ids, lines_or_subs_id, lines_ex_subs_id):
        self.prods_switches_subaction_length = number_generators
        self.loads_switches_subaction_length = number_consumers
        self.lines_or_switches_subaction_length = number_power_lines
        self.lines_ex_switches_subaction_length = number_power_lines
        self.lines_status_subaction_length = number_power_lines

        # In an environment, the actions have fixed shape: self.action_length is expected action list size
        self.action_length = self.prods_switches_subaction_length + self.loads_switches_subaction_length + \
                             self.lines_or_switches_subaction_length + self.lines_ex_switches_subaction_length + \
                             self.lines_status_subaction_length

        self.substations_ids = substations_ids
        self.prods_subs_ids = prods_subs_ids
        self.loads_subs_ids = loads_subs_ids
        self.lines_or_subs_id = lines_or_subs_id
        self.lines_ex_subs_id = lines_ex_subs_id

        # Computes en saves the number of topological switches per substations (sum of prod + load + lines or. and ext.)
        self._substations_n_elements = [len(self.get_topological_switches_of_substation(self.get_do_nothing_action(),
                                                                                        sub_id)[1]) for sub_id in
                                        self.substations_ids]

    def get_do_nothing_action(self):
        """ Creates and returns an action equivalent to a do-nothing: all of the activable switches are 0 i.e.
        not activated.

        :return: an instance of Action that is equivalent to an action doing nothing
        """
        return pypownet.game.Action(prods_switches_subaction=np.zeros(self.prods_switches_subaction_length),
                                    loads_switches_subaction=np.zeros(self.loads_switches_subaction_length),
                                    lines_or_switches_subaction=np.zeros(self.lines_or_switches_subaction_length),
                                    lines_ex_switches_subaction=np.zeros(self.lines_ex_switches_subaction_length),
                                    lines_status_subaction=np.zeros(self.lines_status_subaction_length))

    def array_to_action(self, array):
        """ Converts and returns an Action from a array-object (e.g. list, numpy arrays).

        :param array: array-style object
        :return: an instance of Action equivalent to input action :raise ValueError: the input array is not of the
        same length than the expected action (self.action_length)
        """
        if len(array) != self.action_length:
            raise ValueError('Expected binary array of length %d, got %d' % (self.action_length, len(array)))

        offset = 0
        prods_switches_subaction = array[:self.prods_switches_subaction_length]
        offset += self.prods_switches_subaction_length
        loads_switches_subaction = array[offset:offset + self.loads_switches_subaction_length]
        offset += self.loads_switches_subaction_length
        lines_or_switches_subaction = array[offset:offset + self.lines_or_switches_subaction_length]
        offset += self.lines_or_switches_subaction_length
        lines_ex_switches_subaction = array[offset:offset + self.lines_ex_switches_subaction_length]
        lines_status_subaction = array[-self.lines_status_subaction_length:]

        return pypownet.game.Action(prods_switches_subaction=prods_switches_subaction,
                                    loads_switches_subaction=loads_switches_subaction,
                                    lines_or_switches_subaction=lines_or_switches_subaction,
                                    lines_ex_switches_subaction=lines_ex_switches_subaction,
                                    lines_status_subaction=lines_status_subaction)

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

        prods_switches_subaction_length, loads_switches_subaction_length, lines_or_switches_subaction_length, \
        lines_ex_subaction_length, lines_status_subaction_length = formatted_action.__len__(do_sum=False)

        if prods_switches_subaction_length and prods_switches_subaction_length != self.prods_switches_subaction_length:
            raise ValueError('Expected prods_switches_subaction subaction of size %d, got %d' % (
                self.prods_switches_subaction_length, prods_switches_subaction_length))
        if loads_switches_subaction_length and loads_switches_subaction_length != self.loads_switches_subaction_length:
            raise ValueError('Expected loads_switches_subaction subaction of size %d, got %d' % (
                self.loads_switches_subaction_length, loads_switches_subaction_length))
        if lines_or_switches_subaction_length and lines_or_switches_subaction_length != \
                self.lines_or_switches_subaction_length:
            raise ValueError('Expected lines_or_switches_subaction subaction of size %d, got %d' % (
                self.lines_or_switches_subaction_length, lines_or_switches_subaction_length))
        if lines_ex_subaction_length and lines_ex_subaction_length != self.lines_ex_switches_subaction_length:
            raise ValueError('Expected lines_ex_subaction subaction of size %d, got %d' % (
                self.lines_ex_switches_subaction_length, lines_ex_subaction_length))

        return formatted_action

    def get_number_elements_of_substation(self, substation_id):
        return self._substations_n_elements[np.where(self.substations_ids == substation_id)[0][0]]

    def get_topological_switches_of_substation(self, action, substation_id, do_concatenate=True):
        """ From the current action, retrieves the list of value of the switch (0 or 1) of the switches on which each
        element of the substation with input id. This function also computes the type of element associated to each
        switch value of the returned switches-value list.

        :param substation_id: an integer of the id of the substation to retrieve the switches of its elements in the
        input action
        :return: a switch-values list (binary list) in the order: production (<=1), loads (<=1), lines origins, lines
        extremities; also returns a ElementType list of same size, where each value indicates the type of element
        associated to each first-returned list values.
        """
        assert substation_id in self.substations_ids, 'Substation with id %d does not exist' % substation_id

        # Save the type of each elements in the returned switches list
        elements_type = []

        # Retrieve switches associated with resp. production (max 1 per substation), consumptions (max 1 per substation),
        # origins of lines, extremities of lines; also saves each type inserted within the switches-values list
        prod_switches = action.prods_switches_subaction[
            np.where(self.prods_subs_ids == substation_id)] if substation_id in self.prods_subs_ids else []
        elements_type.extend([ElementType.PRODUCTION] * len(prod_switches))
        load_switches = action.loads_switches_subaction[
            np.where(self.loads_subs_ids == substation_id)] if substation_id in self.loads_subs_ids else []
        elements_type.extend([ElementType.CONSUMPTION] * len(load_switches))
        lines_origins_switches = action.lines_or_switches_subaction[
            np.where(self.lines_or_subs_id == substation_id)] if substation_id in self.lines_or_subs_id else []
        elements_type.extend([ElementType.ORIGIN_POWER_LINE] * len(lines_origins_switches))
        lines_extremities_switches = action.lines_ex_switches_subaction[
            np.where(self.lines_ex_subs_id == substation_id)] if substation_id in self.lines_ex_subs_id else []
        elements_type.extend([ElementType.EXTREMITY_POWER_LINE] * len(lines_extremities_switches))

        assert len(elements_type) == len(prod_switches) + len(load_switches) + len(lines_origins_switches) + len(
            lines_extremities_switches), "Mistmatch lengths for elements type and switches-value list; should not happen"

        return np.concatenate((prod_switches, load_switches, lines_origins_switches,
                               lines_extremities_switches)) if do_concatenate else \
                   (prod_switches, load_switches, lines_origins_switches, lines_extremities_switches), \
               np.asarray(elements_type)

    def set_switches_configuration_of_substation(self, action, substation_id, new_configuration):
        new_configuration = np.asarray(new_configuration)

        _, elements_type = self.get_topological_switches_of_substation(action, substation_id, do_concatenate=False)
        expected_configuration_size = len(elements_type)
        assert expected_configuration_size == len(new_configuration), 'Expected configuration of size %d for' \
                                                                      ' substation %d, got %d' % (
                                                                          expected_configuration_size, substation_id,
                                                                          len(new_configuration))

        action.prods_switches_subaction[self.prods_subs_ids == substation_id] = new_configuration[
            elements_type == ElementType.PRODUCTION]
        action.loads_switches_subaction[self.loads_subs_ids == substation_id] = new_configuration[
            elements_type == ElementType.CONSUMPTION]
        action.lines_or_switches_subaction[self.lines_or_subs_id == substation_id] = new_configuration[
            elements_type == ElementType.ORIGIN_POWER_LINE]
        action.lines_ex_switches_subaction[self.lines_ex_subs_id == substation_id] = new_configuration[
            elements_type == ElementType.EXTREMITY_POWER_LINE]

        assert np.all(self.get_topological_switches_of_substation(action, substation_id)[0] == new_configuration), \
            "Should not happen"

    def get_lines_status_switches_of_substation(self, action, substation_id):
        assert substation_id in self.substations_ids, 'Substation with id %d does not exist' % substation_id

        if substation_id in self.lines_or_subs_id or substation_id in self.lines_ex_subs_id:
            lines_status_switches = action.lines_status_subaction[
                np.where(np.logical_or((self.lines_or_subs_id == substation_id,
                                        self.lines_ex_subs_id == substation_id)))]
        else:
            lines_status_switches = []

        assert len(lines_status_switches) == len(self.lines_ex_subs_id == substation_id) + len(
            self.lines_or_subs_id == substation_id)

        return lines_status_switches

    def set_lines_status_switches_of_substation(self, action, substation_id, new_configuration):
        new_configuration = np.asarray(new_configuration)

        lines_status_switches = self.get_topological_switches_of_substation(action, substation_id)
        expected_configuration_size = len(lines_status_switches)
        assert expected_configuration_size == len(new_configuration), 'Expected configuration of size %d for' \
                                                                      ' substation %d, got %d' % (
                                                                          expected_configuration_size, substation_id,
                                                                          len(new_configuration))

        action.lines_status_subaction[np.where(np.logical_or(
            (self.lines_or_subs_id == substation_id, self.lines_ex_subs_id == substation_id)))] = new_configuration

        assert np.all(self.get_lines_status_switches_of_substation(action, substation_id) == new_configuration), \
            "Should not happen"


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

    def __init__(self, substations_ids, active_loads, reactive_loads, voltage_loads, active_productions,
                 reactive_productions,
                 voltage_productions, active_flows_origin, reactive_flows_origin, voltage_flows_origin,
                 active_flows_extremity, reactive_flows_extremity, voltage_flows_extremity, ampere_flows,
                 thermal_limits, lines_status, are_isolated_loads, are_isolated_prods, loads_substations_ids,
                 prods_substations_ids, lines_or_substations_ids, lines_ex_substations_ids,
                 timesteps_before_lines_reconnectable, timesteps_before_planned_maintenance, planned_active_loads,
                 planned_reactive_loads, planned_active_productions, planned_voltage_productions, date,
                 prods_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes):
        self.substations_ids = substations_ids

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

    def get_nodes_of_substation(self, substation_id):
        """ From the current observation, retrieves the list of value of the nodes on which each element of the
        substation with input id. This function also computes the type of element associated to each node value of the
        returned nodes-value list.

        :param substation_id: an integer of the id of the substation to retrieve the nodes on which its elements are
        wired
        :return: a nodes-values list in the order: production (<=1), loads (<=1), lines origins, lines extremities;
        also returns a ElementType list of same size, where each value indicates the type of element associated to
        each first-returned list values.
        """
        assert substation_id in self.substations_ids, 'Substation with id %d does not exist' % substation_id

        # Save the type of each elements in the returned nodes list
        elements_type = []

        # Retrieve nodes associated with resp. production (max 1 per substation), consumptions (max 1 per substation),
        # origins of lines, extremities of lines; also saves each type inserted within the nodes-values list
        prod_nodes = self.productions_nodes[np.where(
            self.productions_substations_ids == substation_id)] if substation_id in \
                                                                   self.productions_substations_ids else []
        elements_type.extend([ElementType.PRODUCTION] * len(prod_nodes))
        load_nodes = self.loads_nodes[np.where(
            self.loads_substations_ids == substation_id)] if substation_id in self.loads_substations_ids else []
        elements_type.extend([ElementType.CONSUMPTION] * len(load_nodes))
        lines_origin_nodes = self.lines_or_nodes[np.where(
            self.lines_or_substations_ids == substation_id)] if substation_id in self.lines_or_substations_ids else []
        elements_type.extend([ElementType.ORIGIN_POWER_LINE] * len(lines_origin_nodes))
        lines_extremities_nodes = self.lines_ex_nodes[np.where(
            self.lines_ex_substations_ids == substation_id)] if substation_id in self.lines_ex_substations_ids else []
        elements_type.extend([ElementType.EXTREMITY_POWER_LINE] * len(lines_extremities_nodes))

        assert len(elements_type) == len(prod_nodes) + len(load_nodes) + len(lines_origin_nodes) + len(
            lines_extremities_nodes), "Mistmatch lengths for elements type and nodes-value list; should not happen"

        return np.concatenate((prod_nodes, load_nodes, lines_origin_nodes, lines_extremities_nodes)), elements_type

    def __str__(self):
        date_str = 'date:' + self.datetime.strftime("%Y-%m-%d %H:%M")

        def _tabular_prettifier(matrix, formats, column_widths):
            """ Used for printing well shaped tables within terminal and log files
            """
            res = ' |' + ' |'.join('-' * (w - 1) for w in column_widths) + ' |\n'

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
        lines_header += ' |      Origin     |    Extremity    | is |         Origin        |        Extremity      | ' \
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
        self.action_space = ActionSpace(*self.game.get_number_elements(),
                                        substations_ids=self.game.get_substations_ids(),
                                        prods_subs_ids=self.game.get_substations_ids_prods(),
                                        loads_subs_ids=self.game.get_substations_ids_loads(),
                                        lines_or_subs_id=self.game.get_substations_ids_lines_or(),
                                        lines_ex_subs_id=self.game.get_substations_ids_lines_ex())
        self.observation_space = ObservationSpace(*self.game.get_number_elements())

        self.reward_signal = pypownet.reward_signal.DefaultRewardSignal(grid_case=self.game.get_grid_case(),
                                                                        initial_topology=self.game.get_initial_topology())

        self.last_rewards = []

    def _get_obs(self):
        return self.game.export_observation()

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

        reward_aslist = self.reward_signal.compute_reward(observation=observation, action=action, flag=reward_flag)
        self.last_rewards = reward_aslist

        return observation, sum(reward_aslist) if do_sum else reward_aslist, done, reward_flag

    def simulate(self, action=None, do_sum=True):
        """ Computes the reward of the simulation of action to the current grid. """
        # First verify that the action is in expected condition: one array (or list) of expected size of 0 or 1
        try:
            to_simulate_action = self.action_space.verify_action_shape(action)
        except IllegalActionException as e:
            raise e

        observation, reward_flag, done = self.game.simulate(to_simulate_action)

        reward_aslist = self.reward_signal.compute_reward(observation=observation, action=action, flag=reward_flag)
        self.last_rewards = reward_aslist

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
