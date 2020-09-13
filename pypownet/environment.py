__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import numpy as np
from copy import deepcopy
from enum import Enum
from collections import OrderedDict
from gym.spaces import MultiBinary, Box, Dict, Discrete

import pypownet.game


class IllegalActionException(pypownet.game.IllegalActionException):
    def __init__(self, text, illegal_lines_reconnections, illegal_unavailable_lines_switches,
                 illegal_oncoolown_substations_switches, *args):
        super(IllegalActionException, self).__init__(text, illegal_lines_reconnections,
                                                     illegal_unavailable_lines_switches,
                                                     illegal_oncoolown_substations_switches, *args)


# Wrappers for the exceptions of the game module
class DivergingLoadflowException(pypownet.game.DivergingLoadflowException):
    def __init__(self, last_observation, *args):
        super(DivergingLoadflowException, self).__init__(last_observation, *args)


class TooManyProductionsCut(pypownet.game.TooManyProductionsCut):
    def __init__(self, *args):
        super(TooManyProductionsCut, self).__init__(*args)


class TooManyConsumptionsCut(pypownet.game.TooManyConsumptionsCut):
    def __init__(self, *args):
        super(TooManyConsumptionsCut, self).__init__(*args)


class ElementType(Enum):
    PRODUCTION = "production"
    CONSUMPTION = "consumption"
    ORIGIN_POWER_LINE = "origin of power line"
    EXTREMITY_POWER_LINE = "extremity of power line"


# class ActionSpace(object):
class ActionSpace(MultiBinary):
    # def __init__(self, number_generators, number_consumers, number_power_lines, substations_ids, prods_subs_ids,
    #              loads_subs_ids, lines_or_subs_id, lines_ex_subs_id):
    def __init__(self, number_generators, number_consumers, number_power_lines, number_substations, substations_ids,
                 prods_subs_ids, loads_subs_ids, lines_or_subs_id, lines_ex_subs_id):
        self.prods_switches_subaction_length = number_generators
        self.loads_switches_subaction_length = number_consumers
        self.lines_or_switches_subaction_length = number_power_lines
        self.lines_ex_switches_subaction_length = number_power_lines
        self.lines_status_subaction_length = number_power_lines
        self.action_length = self.prods_switches_subaction_length + self.loads_switches_subaction_length + \
                             self.lines_or_switches_subaction_length + self.lines_ex_switches_subaction_length + \
                             self.lines_status_subaction_length
        super().__init__(self.action_length)
        self.substations_ids = substations_ids
        self.prods_subs_ids = prods_subs_ids
        self.loads_subs_ids = loads_subs_ids
        self.lines_or_subs_id = lines_or_subs_id
        self.lines_ex_subs_id = lines_ex_subs_id
        self._substations_n_elements = [len(
            self.get_substation_switches_in_action(self.get_do_nothing_action(as_class_Action=True), sub_id)[1]) for
                                        sub_id in self.substations_ids]

    def get_do_nothing_action(self, as_class_Action=False):
        """ Creates and returns an action equivalent to a do-nothing: all of the activable switches are 0 i.e.
        not activated.

        :return: an instance of pypownet.game.Action that is equivalent to an action doing nothing
        """
        action = pypownet.game.Action(np.zeros(self.prods_switches_subaction_length),
                                      np.zeros(self.loads_switches_subaction_length),
                                      np.zeros(self.lines_or_switches_subaction_length),
                                      np.zeros(self.lines_ex_switches_subaction_length),
                                      np.zeros(self.lines_status_subaction_length), self.substations_ids,
                                      self.prods_subs_ids, self.loads_subs_ids, self.lines_or_subs_id,
                                      self.lines_ex_subs_id, ElementType)
        return action if as_class_Action else action.as_array()

    def array_to_action(self, array):
        """ Converts and returns an pypownet.game.Action from a array-object (e.g. list, numpy arrays).

        :param array: array-style object
        :return: an instance of pypownet.game.Action equivalent to input action
        :raise ValueError: the input array is not of the same length than the expected action (self.action_length)
        """
        if isinstance(array, pypownet.game.Action):
            return array

        if len(array) != self.action_length:
            raise ValueError('Expected action as a binary array of length %d, '
                             'got %d' % (self.action_length, len(array)))

        offset = 0
        prods_switches_subaction = array[:self.prods_switches_subaction_length]
        offset += self.prods_switches_subaction_length
        loads_switches_subaction = array[offset:offset + self.loads_switches_subaction_length]
        offset += self.loads_switches_subaction_length
        lines_or_switches_subaction = array[offset:offset + self.lines_or_switches_subaction_length]
        offset += self.lines_or_switches_subaction_length
        lines_ex_switches_subaction = array[offset:offset + self.lines_ex_switches_subaction_length]
        lines_status_subaction = array[-self.lines_status_subaction_length:]

        return pypownet.game.Action(prods_switches_subaction, loads_switches_subaction,
                                    lines_or_switches_subaction, lines_ex_switches_subaction,
                                    lines_status_subaction, self.substations_ids,
                                    self.prods_subs_ids, self.loads_subs_ids, self.lines_or_subs_id,
                                    self.lines_ex_subs_id, ElementType)

    def _verify_action_shape(self, action):
        if action is None:
            raise ValueError('Expected binary array of length %d, got None' % self.action_length)

        # If the input action is not of class pypownet.game.Action, try to format it into pypownet.game.Action
        # (action must be array-like)
        if not isinstance(action, pypownet.game.Action):
            try:
                formatted_action = self.array_to_action(action)
            except ValueError as e:
                raise e
        else:
            formatted_action = deepcopy(action)

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
        # Sanity check
        assert(substation_id in self.substations_ids)
        return self._substations_n_elements[np.where(self.substations_ids == substation_id)[0][0]]

    def get_substation_switches_in_action(self, action, substation_id, concatenated_output=True):
        """
        From the input action, retrieves the list of value of the switch (0 or 1) of the switches on which each
        element of the substation with input id. This function also computes the type of element associated to each
        switch value of the returned switches-value list.

        :param action: input action whether a numpy array or an element of class pypownet.game.Action.
        :param substation_id: an integer of the id of the substation to retrieve the switches of its elements in the
            input action.
        :param concatenated_output: False to return an array per elementype, True to return a single concatenated array
        :return: a switch-values list (binary list) in the order: production (<=1), loads (<=1), lines origins, lines
            extremities; also returns a ElementType list of same size, where each value indicates the type of element
            associated to each first-returned list values.
        """
        if not isinstance(action, pypownet.game.Action):
            try:
                action = self.array_to_action(action)
            except ValueError as e:
                raise e

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
                               lines_extremities_switches)) if concatenated_output else \
                   (prod_switches, load_switches, lines_origins_switches, lines_extremities_switches), \
               np.asarray(elements_type)

    def set_substation_switches_in_action(self, action, substation_id, new_values):
        """

        Replaces the switches (binary) values of the input substation in the input action with the new specified
        values. Note that the  mapping between the new values and the elements of the considered substation are the
        same as the one retrieved by the opposite function self.get_substation_switches. Consequently, the length of the
        array new_values is len(self.get_substation_switches(action, substation_id)[1]).

        :param action: input action whether a numpy array or an element of class pypownet.game.Action.
        :param substation_id: an integer of the id of the substation to retrieve the switches of its elements in the
            input action
        :return: the modified action; WARNING: the input action is not modified in place if of array type: ensure that
            you catch the returned action as the modified action.
        """
        if not isinstance(action, pypownet.game.Action):
            try:
                action = self.array_to_action(action)
            except ValueError as e:
                raise e

        new_values = np.asarray(new_values)

        _, elements_type = self.get_substation_switches_in_action(action, substation_id, concatenated_output=False)
        expected_configuration_size = len(elements_type)
        assert expected_configuration_size == len(new_values), 'Expected new_values of size %d for' \
                                                               ' substation %d, got size %d' % (
                                                                   expected_configuration_size, substation_id,
                                                                   len(new_values))

        action.prods_switches_subaction[self.prods_subs_ids == substation_id] = new_values[
            elements_type == ElementType.PRODUCTION]
        action.loads_switches_subaction[self.loads_subs_ids == substation_id] = new_values[
            elements_type == ElementType.CONSUMPTION]
        action.lines_or_switches_subaction[self.lines_or_subs_id == substation_id] = new_values[
            elements_type == ElementType.ORIGIN_POWER_LINE]
        action.lines_ex_switches_subaction[self.lines_ex_subs_id == substation_id] = new_values[
            elements_type == ElementType.EXTREMITY_POWER_LINE]

        return action

    def get_lines_status_switches_of_substation(self, action, substation_id):
        assert substation_id in self.substations_ids, 'Substation with id %d does not exist' % substation_id

        lines_status_switches = action.lines_status_subaction[
            np.where(np.logical_or((self.lines_or_subs_id == substation_id, self.lines_ex_subs_id == substation_id)))]

        assert len(lines_status_switches) == sum(self.lines_ex_subs_id == substation_id) + sum(
            self.lines_or_subs_id == substation_id)

        return lines_status_switches

    def set_lines_status_switches_of_substation(self, action, substation_id, new_configuration):
        new_configuration = np.asarray(new_configuration)

        lines_status_switches = self.get_substation_switches_in_action(action, substation_id)
        expected_configuration_size = len(lines_status_switches)
        assert expected_configuration_size == len(new_configuration), 'Expected configuration of size %d for' \
                                                                      ' substation %d, got %d' % (
                                                                          expected_configuration_size, substation_id,
                                                                          len(new_configuration))

        action.lines_status_subaction[np.where(np.logical_or(
            (self.lines_or_subs_id == substation_id, self.lines_ex_subs_id == substation_id)))] = new_configuration

        assert np.all(self.get_lines_status_switches_of_substation(action, substation_id) == new_configuration), \
            "Should not happen"

    @staticmethod
    def get_lines_status_switch_from_id(action, line_id):
        return action.lines_status_subaction[line_id]

    @staticmethod
    def set_lines_status_switch_from_id(action, line_id, new_switch_value):
        action.lines_status_subaction[line_id] = new_switch_value


class ObservationSpace(Dict):
    def __init__(self, number_generators, number_consumers, number_power_lines, number_substations,
                 n_timesteps_horizon_maintenance):
        self.number_productions = number_generators
        self.number_loads = number_consumers
        self.number_power_lines = number_power_lines
        self.number_substations = number_substations
        self.n_timesteps_horizon_maintenance = n_timesteps_horizon_maintenance
        self.grid_number_of_elements = self.number_productions + self.number_loads + 2 * self.number_power_lines

        dict_spaces = OrderedDict([
            ('MinimalistACObservation', Dict(OrderedDict([
                ('MinimalistObservation', Dict(OrderedDict([
                    ('active_loads', Box(low=-np.inf, high=np.inf, shape=(number_consumers,), dtype=np.float32)),
                    ('are_loads_cut', MultiBinary(n=number_consumers)),
                    ('planned_active_loads', Box(low=-np.inf, high=np.inf, shape=(number_consumers,),
                                                 dtype=np.float32)),
                    ('loads_nodes', Box(-np.inf, np.inf, (number_consumers,), np.int32)),

                    ('active_productions', Box(low=-np.inf, high=np.inf, shape=(number_generators,), dtype=np.float32)),
                    ('are_productions_cut', MultiBinary(n=number_generators)),
                    ('planned_active_productions', Box(low=-np.inf, high=np.inf, shape=(number_generators,),
                                                       dtype=np.float32)),
                    ('productions_nodes', Box(-np.inf, np.inf, (number_generators,), np.int32)),

                    ('lines_or_nodes', Box(-np.inf, np.inf, (number_power_lines,), np.int32)),
                    ('lines_ex_nodes', Box(-np.inf, np.inf, (number_power_lines,), np.int32)),

                    ('ampere_flows', Box(0, np.inf, (number_power_lines,), np.float32)),
                    ('lines_status', MultiBinary(n=number_power_lines)),
                    ('timesteps_before_lines_reconnectable', Box(0, np.inf, (number_power_lines,), np.int32)),
                    ('timesteps_before_lines_reactionable', Box(0, np.inf, (number_power_lines,), np.int32)),
                    ('timesteps_before_nodes_reactionable', Box(0, np.inf, (self.number_substations,), np.int32)),
                    ('timesteps_before_planned_maintenance', Box(0, np.inf, (number_power_lines,), np.int32)),

                    ('date_year', Discrete(3000)),
                    ('date_month', Discrete(12)),
                    ('date_day', Discrete(32)),
                    ('date_hour', Discrete(24)),
                    ('date_minute', Discrete(60)),
                    ('date_second', Discrete(60)),
                ]))),

                ('reactive_loads', Box(low=-np.inf, high=np.inf, shape=(number_consumers,), dtype=np.float32)),
                ('voltage_loads', Box(low=-np.inf, high=np.inf, shape=(number_consumers,), dtype=np.float32)),

                ('reactive_productions', Box(low=-np.inf, high=np.inf, shape=(number_generators,), dtype=np.float32)),
                ('voltage_productions', Box(low=-np.inf, high=np.inf, shape=(number_generators,), dtype=np.float32)),

                ('active_flows_origin', Box(low=-np.inf, high=np.inf, shape=(number_power_lines,), dtype=np.float32)),
                ('reactive_flows_origin', Box(low=-np.inf, high=np.inf, shape=(number_power_lines,), dtype=np.float32)),
                ('voltage_flows_origin', Box(low=-np.inf, high=np.inf, shape=(number_power_lines,), dtype=np.float32)),

                ('active_flows_extremity', Box(low=-np.inf, high=np.inf, shape=(number_power_lines,),
                                               dtype=np.float32)),
                ('reactive_flows_extremity', Box(low=-np.inf, high=np.inf, shape=(number_power_lines,),
                                                 dtype=np.float32)),
                ('voltage_flows_extremity', Box(low=-np.inf, high=np.inf, shape=(number_power_lines,),
                                                dtype=np.float32)),

                ('planned_reactive_loads', Box(low=-np.inf, high=np.inf, shape=(number_consumers,), dtype=np.float32)),
                ('planned_voltage_productions', Box(low=-np.inf, high=np.inf, shape=(number_generators,),
                                                    dtype=np.float32)),
            ]))),

            ('substations_ids', Box(-np.inf, np.inf, (number_substations,), np.int32)),
            ('loads_substations_ids', Box(-np.inf, np.inf, (number_consumers,), np.int32)),
            ('productions_substations_ids', Box(-np.inf, np.inf, (number_generators,), np.int32)),
            ('lines_or_substations_ids', Box(-np.inf, np.inf, (number_power_lines,), np.int32)),
            ('lines_ex_substations_ids', Box(-np.inf, np.inf, (number_power_lines,), np.int32)),
            ('thermal_limits', Box(0, np.inf, (number_power_lines,), np.int32)),
            ('initial_productions_nodes', Box(-np.inf, np.inf, (number_generators,), np.int32)),
            ('initial_loads_nodes', Box(-np.inf, np.inf, (number_consumers,), np.int32)),
            ('initial_lines_or_nodes', Box(-np.inf, np.inf, (number_power_lines,), np.int32)),
            ('initial_lines_ex_nodes', Box(-np.inf, np.inf, (number_power_lines,), np.int32)),
        ])

        super().__init__(dict_spaces)

        def seek_shapes(gym_dict, shape):
            """ Computes and returns the shape of self ie the set of all its attributes shapes as a tuple of tuples.

            :param gym_dict: an instance of gym Spaces
            :param shape: a container that is recursively filled with res
            :return: a tuple of tuples
            """
            # loop through all dicts first
            for k, v in gym_dict.spaces.items():
                if isinstance(v, Dict) or isinstance(v, OrderedDict):
                    shape = seek_shapes(v, shape)
            # then save shapes
            for k, v in gym_dict.spaces.items():
                if not (isinstance(v, Dict) or isinstance(v, OrderedDict)):
                    shape += (v.shape,) if not isinstance(v, Discrete) else ((1,),)

            return shape

        self.shape = seek_shapes(self, ())

    def array_to_observation(self, array):
        """ Converts and returns an pypownet.game.Observation from a array-object (e.g. list, numpy arrays).

        :param array: array-style object
        :return: an instance of pypownet.game.Action equivalent to input action
        :raise ValueError: the input array is not of the same length than the expected action (self.action_length)
        """
        expected_length = sum(list(map(sum, self.shape)))
        if len(array) != expected_length:
            raise ValueError('Expected observation array of length %d, got %d' % (expected_length, len(array)))

        def transform_array(gym_dict, input_array, res):
            # loop through all dicts first
            for k, v in gym_dict.spaces.items():
                if isinstance(v, Dict) or isinstance(v, OrderedDict):
                    input_array, res = transform_array(v, input_array, res)
            # then save shapes
            for k, v in gym_dict.spaces.items():
                if not (isinstance(v, Dict) or isinstance(v, OrderedDict)):
                    n_elements = np.prod(v.shape) if not isinstance(v,
                                                                    Discrete) else 1  # prod because some containers are flattened
                    res[k] = input_array[:n_elements]
                    input_array = input_array[n_elements:]  # shift arrato discard just selected values

            return input_array, res

        _, subobservations = transform_array(self, array, {})
        return Observation(**subobservations)


class MinimalistObservation(object):
    def __init__(self, active_loads, active_productions, ampere_flows, lines_status, are_loads_cut,
                 are_productions_cut, timesteps_before_lines_reconnectable, timesteps_before_lines_reactionable,
                 timesteps_before_nodes_reactionable, timesteps_before_planned_maintenance, planned_active_loads,
                 planned_active_productions, date_year, date_month, date_day, date_hour, date_minute, date_second,
                 productions_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes):
        # Loads related state values
        self.active_loads = active_loads
        self.are_loads_cut = are_loads_cut
        self.loads_nodes = loads_nodes

        # Productions related state values
        self.active_productions = active_productions
        self.are_productions_cut = are_productions_cut
        self.productions_nodes = productions_nodes

        # Origin flows related state values
        self.lines_or_nodes = lines_or_nodes
        # Extremity flows related state values
        self.lines_ex_nodes = lines_ex_nodes

        # Ampere flows and thermal limits
        self.ampere_flows = ampere_flows
        self.lines_status = lines_status

        # Per-line timesteps to wait before the line is full repaired, after being broken by cascading failure,
        # random hazards, or shut down for maintenance (e.g. painting)
        self.timesteps_before_lines_reconnectable = timesteps_before_lines_reconnectable
        self.timesteps_before_planned_maintenance = timesteps_before_planned_maintenance
        # Per-line/per-gridelement timesteps to wait before it can be actionable, ie there is a 1 for the corresponding
        # element in the action
        self.timesteps_before_lines_reactionable = timesteps_before_lines_reactionable
        self.timesteps_before_nodes_reactionable = timesteps_before_nodes_reactionable

        # Planned injections for the next timestep
        self.planned_active_loads = planned_active_loads
        self.planned_active_productions = planned_active_productions

        self.date_year = date_year
        self.date_month = date_month
        self.date_day = date_day
        self.date_hour = date_hour
        self.date_minute = date_minute
        self.date_second = date_second

    def as_array(self):
        return np.concatenate((
            self.active_loads, self.are_loads_cut, self.planned_active_loads.flatten(), self.loads_nodes,

            self.active_productions, self.are_productions_cut, self.planned_active_productions.flatten(),
            self.productions_nodes,

            self.lines_or_nodes, self.lines_ex_nodes,

            self.ampere_flows, self.lines_status, self.timesteps_before_lines_reconnectable,
            self.timesteps_before_lines_reactionable, self.timesteps_before_nodes_reactionable,
            self.timesteps_before_planned_maintenance,

            np.asarray([self.date_year, self.date_month, self.date_day, self.date_hour, self.date_minute,
                        self.date_second]).flatten(),
        ))

    @staticmethod
    def __keys__():
        return ['active_loads', 'are_loads_cut', 'loads_nodes', 'active_productions', 'are_productions_cut',
                'productions_nodes', 'lines_or_nodes', 'lines_ex_nodes', 'ampere_flows', 'lines_status',
                'timesteps_before_lines_reconnectable', 'timesteps_before_lines_reactionable',
                'timesteps_before_nodes_reactionable', 'timesteps_before_planned_maintenance', 'planned_active_loads',
                'planned_active_productions', 'datetime']

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if k in self.__keys__()}


class MinimalistACObservation(MinimalistObservation):
    def __init__(self, active_loads, reactive_loads, voltage_loads, active_productions, reactive_productions,
                 voltage_productions, active_flows_origin, reactive_flows_origin, voltage_flows_origin,
                 active_flows_extremity, reactive_flows_extremity, voltage_flows_extremity, ampere_flows, lines_status,
                 are_loads_cut, are_productions_cut, timesteps_before_lines_reconnectable,
                 timesteps_before_lines_reactionable, timesteps_before_nodes_reactionable,
                 timesteps_before_planned_maintenance, planned_active_loads, planned_reactive_loads,
                 planned_active_productions, planned_voltage_productions, date_year, date_month, date_day, date_hour,
                 date_minute, date_second, productions_nodes, loads_nodes,
                 lines_or_nodes, lines_ex_nodes):
        super().__init__(active_loads, active_productions, ampere_flows, lines_status, are_loads_cut,
                         are_productions_cut, timesteps_before_lines_reconnectable, timesteps_before_lines_reactionable,
                         timesteps_before_nodes_reactionable, timesteps_before_planned_maintenance,
                         planned_active_loads, planned_active_productions, date_year, date_month, date_day, date_hour,
                         date_minute, date_second, productions_nodes, loads_nodes, lines_or_nodes, lines_ex_nodes)
        self.reactive_loads = reactive_loads
        self.voltage_loads = voltage_loads

        self.reactive_productions = reactive_productions
        self.voltage_productions = voltage_productions

        self.active_flows_origin = active_flows_origin
        self.reactive_flows_origin = reactive_flows_origin
        self.voltage_flows_origin = voltage_flows_origin
        self.active_flows_extremity = active_flows_extremity
        self.reactive_flows_extremity = reactive_flows_extremity
        self.voltage_flows_extremity = voltage_flows_extremity

        self.planned_reactive_loads = planned_reactive_loads
        self.planned_voltage_productions = planned_voltage_productions

    def as_array(self):
        return np.concatenate((super(MinimalistACObservation, self).as_array(),
                               self.reactive_loads, self.voltage_loads,
                               self.reactive_productions, self.voltage_productions,
                               self.active_flows_origin, self.reactive_flows_origin, self.voltage_flows_origin,
                               self.active_flows_extremity, self.reactive_flows_extremity, self.voltage_flows_extremity,
                               self.planned_reactive_loads, self.planned_voltage_productions,))

    @staticmethod
    def __keys__():
        return ['reactive_loads', 'voltage_loads', 'reactive_productions', 'voltage_productions', 'active_flows_origin',
                'reactive_flows_origin', 'voltage_flows_origin', 'active_flows_extremity', 'reactive_flows_extremity',
                'voltage_flows_extremity', 'planned_reactive_loads', 'planned_voltage_productions']

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if k in self.__keys__() + super(MinimalistACObservation, self).__keys__()}

    def as_minimalist(self):
        return super(MinimalistACObservation, self)


class Observation(MinimalistACObservation):
    """ The class State is a container for all the values representing the state of a given grid at a given time. It
    contains the following values:
    * The active and reactive power values of the loads
    * The active power values and the voltage setpoints of the productions
    * The values of the power through the lines: the active and reactive values at the origin/extremity of the
    lines as well as the lines capacity usage
    * The exhaustive topology of the grid, as a stacked vector of one-hot vectors
    """

    def __init__(self, substations_ids, active_loads, reactive_loads, voltage_loads, active_productions,
                 reactive_productions, voltage_productions, active_flows_origin, reactive_flows_origin,
                 voltage_flows_origin, active_flows_extremity, reactive_flows_extremity, voltage_flows_extremity,
                 ampere_flows, thermal_limits, lines_status, are_loads_cut, are_productions_cut,
                 loads_substations_ids, productions_substations_ids, lines_or_substations_ids, lines_ex_substations_ids,
                 timesteps_before_lines_reconnectable, timesteps_before_lines_reactionable,
                 timesteps_before_nodes_reactionable, timesteps_before_planned_maintenance, planned_active_loads,
                 planned_reactive_loads, planned_active_productions, planned_voltage_productions, date_year,
                 date_month, date_day, date_hour, date_minute, date_second, productions_nodes,
                 loads_nodes, lines_or_nodes, lines_ex_nodes, initial_productions_nodes, initial_loads_nodes,
                 initial_lines_or_nodes, initial_lines_ex_nodes):
        super(Observation, self).__init__(active_loads, reactive_loads, voltage_loads, active_productions,
                                          reactive_productions, voltage_productions, active_flows_origin,
                                          reactive_flows_origin, voltage_flows_origin, active_flows_extremity,
                                          reactive_flows_extremity, voltage_flows_extremity, ampere_flows,
                                          lines_status, are_loads_cut, are_productions_cut,
                                          timesteps_before_lines_reconnectable, timesteps_before_lines_reactionable,
                                          timesteps_before_nodes_reactionable, timesteps_before_planned_maintenance,
                                          planned_active_loads, planned_reactive_loads, planned_active_productions,
                                          planned_voltage_productions, date_year, date_month, date_day, date_hour,
                                          date_minute, date_second, productions_nodes, loads_nodes,
                                          lines_or_nodes, lines_ex_nodes)
        # Fixed ids of elements: substations, loads, prods, lines or and lines ex
        self.substations_ids = substations_ids
        self.loads_substations_ids = loads_substations_ids
        self.productions_substations_ids = productions_substations_ids
        self.lines_or_substations_ids = lines_or_substations_ids
        self.lines_ex_substations_ids = lines_ex_substations_ids

        self.thermal_limits = thermal_limits

        # Initial topology
        self.initial_productions_nodes = initial_productions_nodes
        self.initial_loads_nodes = initial_loads_nodes
        self.initial_lines_or_nodes = initial_lines_or_nodes
        self.initial_lines_ex_nodes = initial_lines_ex_nodes

    def as_dict(self):
        return self.__dict__

    def as_array(self):
        return np.concatenate((super(Observation, self).as_array(),
                               self.substations_ids,
                               self.loads_substations_ids,
                               self.productions_substations_ids,
                               self.lines_or_substations_ids,
                               self.lines_ex_substations_ids,
                               self.thermal_limits,
                               self.initial_productions_nodes,
                               self.initial_loads_nodes,
                               self.initial_lines_or_nodes,
                               self.initial_lines_ex_nodes,
        ))

    def as_ac_minimalist(self):
        return super(Observation, self)

    def as_minimalist(self):
        return super(Observation, self).as_minimalist()

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
        assert substation_id in self.substations_ids, \
            'Substation with id {} does not exist; available substations: {}'.format(substation_id,
                                                                                     self.substations_ids)

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

    def get_lines_status_of_substation(self, substation_id):
        """ From the current observation, retrieves the list of lines status (binary) from lines connected to the input
        substations. This function also computes and retrieve the list of ifs of ids at the other end of each
        corresponding lines.

        :param substation_id: an integer of the id of the substation to retrieve the nodes on which its elements are
            wired
        :return: (consistently fixed-order list of binary (0 or 1) values, list of ids of other end substations)
        """
        assert substation_id in self.substations_ids, \
            'Substation with id {} does not exist; available substations: {}'.format(substation_id,
                                                                                     self.substations_ids)

        lines_status = self.lines_status
        lines_origin_substations_ids = self.lines_or_substations_ids
        lines_extremity_substations_ids = self.lines_ex_substations_ids

        # get lines with origin or extremity at input substation
        ori_subs_ids = lines_origin_substations_ids == substation_id
        ext_subs_ids = lines_extremity_substations_ids == substation_id
        are_concerned_lines = np.logical_or(ori_subs_ids, ext_subs_ids)
        concerned_lines_status = lines_status[are_concerned_lines]

        # compute output array with respected order independently of origin or extremity
        other_end_subs_ids = []
        for i, (ori_sub_id, ext_sub_id) in enumerate(zip(ori_subs_ids, ext_subs_ids)):
            if ori_sub_id:
                other_end_subs_ids.append(lines_extremity_substations_ids[i])
            elif ext_sub_id:
                other_end_subs_ids.append(lines_origin_substations_ids[i])
        assert len(other_end_subs_ids) == len(concerned_lines_status)

        return concerned_lines_status, list(map(int, other_end_subs_ids))

    def get_lines_capacity_usage(self):
        return np.divide(self.ampere_flows, self.thermal_limits)

    def __str__(self):
        date_str = 'date: %d of %d of %d at %dh%dm%ds' % (self.date_year, self.date_month, self.date_day,
                                                          self.date_hour, self.date_minute, self.date_second)

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
    def __init__(self, parameters_folder, game_level, chronic_looping_mode='natural', start_id=0,
                 game_over_mode='soft', renderer_latency=None, without_overflow_cutoff=False, seed=None):
        """ Instantiate the game Environment based on the specified parameters.
        Saves class object arguments and declares to be instantiated environment object. The function subcontracts
        the initialization of objects to self.reset. """
        # save parameters
        self.parameters_folder = parameters_folder
        self.game_level = game_level
        self.chronic_looping_mode = chronic_looping_mode
        self.start_id = start_id
        self.game_over_mode = game_over_mode
        self.renderer_latency = renderer_latency
        self.without_overflow_cutoff = without_overflow_cutoff

        self.game = None
        self.action_space = None
        self.observation_space = None
        self.reward_signal = None
        self.last_rewards = None

        if seed is not None:
            np.random.seed(seed)

        self.reset()

    def reset(self):
        """ Instantiate the game Environment based on the specified parameters. """
        # Instantiate game & action space
        self.game = pypownet.game.Game(parameters_folder=self.parameters_folder, game_level=self.game_level,
                                       chronic_looping_mode=self.chronic_looping_mode,
                                       chronic_starting_id=self.start_id, game_over_mode=self.game_over_mode,
                                       renderer_frame_latency=self.renderer_latency,
                                       without_overflow_cutoff=self.without_overflow_cutoff)

        self.action_space = ActionSpace(*self.game.get_number_elements(),
                                        substations_ids=self.game.get_substations_ids(),
                                        prods_subs_ids=self.game.get_substations_ids_prods(),
                                        loads_subs_ids=self.game.get_substations_ids_loads(),
                                        lines_or_subs_id=self.game.get_substations_ids_lines_or(),
                                        lines_ex_subs_id=self.game.get_substations_ids_lines_ex())
        n_prods, n_loads, n_lines, n_substations = self.game.get_number_elements()
        self.observation_space = ObservationSpace(n_prods, n_loads, n_lines, n_substations,
                                                  self.game.n_timesteps_horizon_maintenance)

        self.reward_signal = self.game.get_reward_signal_class()
        self.last_rewards = []

        return self.get_observation(True)  # in pypownet, the convention is to return any env objects as arrays

    def get_observation(self, as_array=True):
        observation = self.game.export_observation()
        return observation.as_array() if as_array else observation
    
    def _get_obs(self):
        return self.get_observation(False)

    def is_action_valid(self, action):
        return self.game.is_action_valid(action)

    def step(self, action, do_sum=True):
        """ Performs a game step given an action. The as list pattern is:
        load_cut_reward, prod_cut_reward, action_cost_reward, reference_grid_distance_reward, line_usage_reward
        """
        # First verify that the action is in expected condition: one array (or list) of expected size of 0 or 1
        try:
            submitted_action = self.action_space._verify_action_shape(action)
        except IllegalActionException as e:
            raise e

        observation, reward_flag, done = self.game.step(submitted_action)
        reward_flag = self.__wrap_exception(reward_flag)

        reward_aslist = self.reward_signal.compute_reward(observation=observation, action=submitted_action,
                                                          flag=reward_flag)
        self.last_rewards = reward_aslist

        return observation.as_array() if observation is not None else observation, \
               sum(reward_aslist) if do_sum else reward_aslist, done, reward_flag

    def simulate(self, action, do_sum=True):
        """ Computes the reward of the simulation of action to the current grid. """
        # First verify that the action is in expected condition: one array (or list) of expected size of 0 or 1
        try:
            to_simulate_action = self.action_space._verify_action_shape(action)
        except IllegalActionException as e:
            raise e

        observation, reward_flag, done = self.game.simulate(to_simulate_action)
        reward_flag = self.__wrap_exception(reward_flag)

        reward_aslist = self.reward_signal.compute_reward(observation=observation, action=to_simulate_action,
                                                          flag=reward_flag)
        self.last_rewards = reward_aslist

        return observation.as_array() if observation is not None else observation, \
               sum(reward_aslist) if do_sum else reward_aslist, done, reward_flag

    def process_game_over(self):
        self.game.process_game_over()
        return self.get_observation()

    def render(self, game_over=False):
        self.game.render(self.last_rewards, game_over=game_over)

    @staticmethod
    def __wrap_exception(flag):
        if isinstance(flag, pypownet.game.DivergingLoadflowException):
            return DivergingLoadflowException(flag.last_observation, flag.text)
        elif isinstance(flag, pypownet.game.TooManyConsumptionsCut):
            return TooManyConsumptionsCut(flag.text)
        elif isinstance(flag, pypownet.game.TooManyProductionsCut):
            return TooManyProductionsCut(flag.text)
        elif isinstance(flag, pypownet.game.IllegalActionException):
            return IllegalActionException(flag.text, flag.get_has_too_much_activations(),
                                          flag.get_illegal_broken_lines_reconnections(),
                                          flag.get_illegal_oncoolown_lines_switches(),
                                          flag.get_illegal_oncoolown_substations_switches())
        else:
            return flag

    ##### HELPERS FOR LOGGING
    def get_current_chronic_name(self):
        return self.game.get_current_chronic_name()

    def get_current_datetime(self):
        return self.game.get_current_datetime()


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

    'substations_ids': 'ID of all the substations of the grid.',
    'prods_substations_ids': 'ID of the substation on which the productions (generators) are wired.',
    'loads_substations_ids': 'ID of the substation on which the loads (consumers) are wired.',
    'lines_or_substations_ids': 'ID of the substation on which the lines origin are wired.',
    'lines_ex_substations_ids': 'ID of the substation on which the lines extremity are wired.',

    'lines_status': 'Mask whether the lines are switched ON (1) or switched OFF (0).',
    'timesteps_before_lines_reconnectable': 'Number of timesteps to wait before a line is switchable ON.',
    'timesteps_before_lines_reactionable': 'Number of timesteps to wait before a recently actioned line can be used '
                                           'again.',
    'timesteps_before_nodes_reactionable': 'Number of timesteps to wait before a recently actioned node can be used '
                                           'again.',
    'timesteps_before_planned_maintenance': 'Number of timesteps to wait before a line will be switched OFF for'
                                            'maintenance',

    'loads_nodes': 'The node on which each load is connected within their corresponding substations.',
    'productions_nodes': 'The node on which each production is connected within their corresponding substations.',
    'lines_or_nodes': 'The node on which each origin of line is connected within their corresponding substations.',
    'lines_ex_nodes': 'The node on which each extremity of line is connected within their corresponding substations.',

    'initial_productions_nodes': 'The initial (reference) node on which each load is connected within their '
                                 'corresponding substations.',
    'initial_loads_nodes': 'The initial (reference) node on which each production is connected within their '
                           'corresponding substations.',
    'initial_lines_or_nodes': 'The initial (reference) node on which each origin of line is connected within their '
                              'corresponding substations.',
    'initial_lines_ex_nodes': 'The initial (reference) node on which each extremity of line is connected within '
                              'their corresponding substations.',

    'planned_active_loads': 'An array-like container of the previsions of the active power of loads for future'
                            'timestep(s).',
    'planned_reactive_loads': 'An array-like container of the previsions of the reactive power of loads for future'
                              'timestep(s).',
    'planned_active_productions': 'An array-like container of the previsions of the active power of productions for '
                                  'future timestep(s).',
    'planned_voltage_productions': 'An array-like container of the previsions of the voltage of productions for future'
                                   'timestep(s).',

    'datetime': 'A Python datetime object containing the date of the observation.',
}

MINIMALISTACOBSERVATION_MEANING = {k: v for k, v in OBSERVATION_MEANING.items()
                                   if k in MinimalistACObservation.__keys__()}

MINIMALISTOBSERVATION_MEANING = {k: v for k, v in OBSERVATION_MEANING.items()
                                 if k in MinimalistObservation.__keys__()}
