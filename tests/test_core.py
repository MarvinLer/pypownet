"""This file contains all core tests."""

import sys


# sys.path.insert(0, "/home/mozgawamar/Documents/pypownet_last_version/pypownet/")
# print("sys path = ", sys.path)

from pypownet.environment import RunEnv, ElementType
from pypownet.runner import Runner
from pypownet.agent import *
from pypownet.game import TooManyProductionsCut, TooManyConsumptionsCut, IllegalActionException
from tests.test_basic import WrappedRunner, get_verbose_node_topology
import math


class Trivial_agent(Agent):
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_LineChangePersistance created...")

        self.current_step = 1
        self.line_to_cut = 9

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        # print(observation)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 1:
            pass

        self.current_step += 1

        return action


class Agent_test_LimitOfProdsLost(Agent):
    """This agent tests the restriction : max_number_prods_game_over: 1
        t = 1, we disconnect the prod on node 1.
        t = 2, same for node 8, connect Prod to busbar 1
        t = 3, check that we had indeed a Game Over and the game reset. IE, observation.productions_nodes = [0,0,0,0,0]
    """

    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_LimitOfProdsLost created...")

        self.current_step = 1
        self.line_to_cut = None
        self.node_to_change = 1

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        # =================================== ACT FUNCTION STATS HERE ===================================
        action_space = self.environment.action_space
        print("prods_substations_ids = ", observation.productions_substations_ids)
        print("prods_substationsTopo = ", observation.productions_nodes)
        # print("are prods cut         = ", observation.are_productions_cut)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)
        # get current configuration
        # current_conf, types = observation.get_nodes_of_substation(self.node_to_change)
        # print(f"Step[{self.current_step}]: current conf node [{self.node_to_change}] = {current_conf}")
        # print("types = ", types)

        if self.current_step == 1:
            # we disconnect the prod on node 1, ie, change the topology.

            self.node_to_change = 1
            print("we change the topology of node {}, curr_step = {}".format(self.node_to_change, self.current_step))

            expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
            # Choses a new switch configuration (binary array)
            target_configuration = np.zeros(expected_target_configuration_size)
            # we connect the PRODUCTION to BUSBAR 1
            target_configuration[0] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=1,
                                                           new_values=target_configuration)
            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, self.node_to_change)
            assert np.all(current_configuration == target_configuration)

        if self.current_step == 2:
            self.node_to_change = 8
            # CHANGE TOPOLOGY OF NODE_TO_CHANGE
            print("we change the topology of node {}, curr_step = {}".format(self.node_to_change, self.current_step))

            expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
            # Choses a new switch configuration (binary array)
            target_configuration = np.zeros(expected_target_configuration_size)
            # we connect the PRODUCTION to BUSBAR 1
            target_configuration[0] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=8,
                                                           new_values=target_configuration)
            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, self.node_to_change)
            assert np.all(current_configuration == target_configuration)

        if self.current_step == 3:
            # check that we had indeed a Game Over and the game reset. IE, observation.productions_nodes = [0,0,0,0,0]
            assert(list(observation.productions_nodes) == [0, 0, 0, 0, 0])

        print("the action we return is, action = ", action)

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        print("========================================================")
        self.current_step += 1

        return action



class Agent_test_LimitOfLoadsLost(Agent):
    """This agent tests the restriction : max_number_loads_game_over: 1
        t = 1, we disconnect the load on node 3.
        t = 2, same for node 12, connect Load to busbar 1
        t = 3, check that we had indeed a Game Over and the game reset. IE, observation.loads_nodes = [0,0,..,0,0]
    """

    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_LimitOfLoadsLost created...")

        self.current_step = 1
        self.line_to_cut = None

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        # =================================== ACT FUNCTION STATS HERE ===================================
        action_space = self.environment.action_space
        print("loads_substations_ids = ", observation.loads_substations_ids)
        print("loads_substationsTopo = ", observation.loads_nodes)
        print("are_loads_cut         = ", observation.are_loads_cut)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        # get current configuration
        # current_conf, types = observation.get_nodes_of_substation(self.node_to_change)
        # print(f"Step[{self.current_step}]: current conf node [{self.node_to_change}] = {current_conf}")
        # print("types = ", types)

        if self.current_step == 1:
            node_to_change = 3
            current_conf, types = observation.get_nodes_of_substation(node_to_change)
            print(f"Step[{self.current_step}]: current conf node [{node_to_change}] = {current_conf}")
            print("types = ", types)
            # CHANGE TOPOLOGY OF NODE_TO_CHANGE
            print("we change the topology of node {}, curr_step = {}".format(node_to_change, self.current_step))

            expected_target_configuration_size = action_space.get_number_elements_of_substation(node_to_change)
            # Choses a new switch configuration (binary array)
            target_configuration = np.zeros(expected_target_configuration_size)
            # we connect the PRODUCTION to BUSBAR 1
            target_configuration[1] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=node_to_change,
                                                           new_values=target_configuration)
            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, node_to_change)
            assert np.all(current_configuration == target_configuration)

        if self.current_step == 2:
            node_to_change = 12
            current_conf, types = observation.get_nodes_of_substation(node_to_change)
            print(f"Step[{self.current_step}]: current conf node [{node_to_change}] = {current_conf}")
            print("types = ", types)
            print("we change the topology of node {}, curr_step = {}".format(node_to_change, self.current_step))

            expected_target_configuration_size = action_space.get_number_elements_of_substation(node_to_change)
            # Choses a new switch configuration (binary array)
            target_configuration = np.zeros(expected_target_configuration_size)
            # we connect the PRODUCTION to BUSBAR 1
            target_configuration[0] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=node_to_change,
                                                           new_values=target_configuration)
            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, node_to_change)
            assert np.all(current_configuration == target_configuration)

        if self.current_step == 3:
            # check that we had indeed a Game Over and the game reset. IE, observation.loads_nodes = [0,0,...,0,0]
            assert(list(observation.loads_nodes) == list(np.zeros(len(observation.loads_substations_ids))))

        print("the action we return is, action = ", action)

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        print("========================================================")
        self.current_step += 1

        return action


class Agent_test_InputLoadValues(Agent):
    """This agent compares the input LOAD values found in the chronics and internal observations for 3 steps. """
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_InputLoadValues created...")

        self.current_step = 1
        self.line_to_cut = None

    def act(self, observation):
        print("[Current step] = ", self.current_step)
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        current_load_powers = observation.active_loads
        print("current_load_powers = ", current_load_powers)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 1:
            expected = [25.629642, 97.45528, 49.735317, 8.250563, 10.010641, 30.2604, 9.736532, 3.3486228, 7.0213113,
                        16.209476, 16.188494]
            for expected_elem, core_elem in zip(expected, current_load_powers):
                error_diff = core_elem - expected_elem

                print("error diff = ", error_diff)
                assert (math.fabs(error_diff) < 1e-3)

        if self.current_step == 2:
            expected = [21.07166, 87.22948, 43.29531, 6.9710474, 10.483086, 28.114975, 10.368015, 3.0358257, 5.108532,
                        12.720526, 12.9846325]
            for expected_elem, core_elem in zip(expected, current_load_powers):
                error_diff = core_elem - expected_elem

                print("error diff = ", error_diff)
                assert (math.fabs(error_diff) < 1e-3)

        if self.current_step == 3:
            expected = [18.838198, 86.235115, 44.783886, 6.563092, 9.875335, 24.161335, 6.824309, 3.2030978, 4.8327637,
                        12.320875, 13.072087]
            for expected_elem, core_elem in zip(expected, current_load_powers):
                error_diff = core_elem - expected_elem
                print("error diff = ", error_diff)
                assert (math.fabs(error_diff) < 1e-3)

        self.current_step += 1

        return action


class Agent_test_InputProdValues(Agent):
    """This agent compares the input PROD values found in the chronics and internal observations for 3 steps. """
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_InputProdValues created...")

        self.current_step = 1
        self.line_to_cut = None

    def act(self, observation):
        print("[Current step] = ", self.current_step)
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        current_production_powers = observation.active_productions
        # print("prods_substations_ids = ", observation.productions_substations_ids)
        # print("real power produced by generators = ", observation.active_productions)
        # print("reactive power produced by gens   = ", observation.reactive_productions)
        # print("are_prods_cut = ", observation.are_productions_cut)

        # for node_id, active_prod, reactive_prod in zip(observation.productions_substations_ids,
        #                                                observation.active_productions,
        #                                                observation.reactive_productions):
        #     print("production node {} total prod power = {}".format(node_id, active_prod + reactive_prod))

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 1:
            expected = [123.370285, 49.144115, 32.21891, 38.52085, 35.704945]
            for expected_elem, core_elem in zip(expected, current_production_powers):
                error_diff = core_elem - expected_elem

                print("error diff = ", error_diff)
                assert (math.fabs(error_diff) < 1e-3)

        if self.current_step == 2:
            expected = [104.072556, 43.576332, 31.90516, 32.831142, 32.747932]
            for expected_elem, core_elem in zip(expected, current_production_powers):
                error_diff = core_elem - expected_elem

                print("error diff = ", error_diff)
                assert (math.fabs(error_diff) < 1e-3)

        if self.current_step == 3:
            expected = [134.51176, 56.608887, 0.0, 0.0, 46.029488]
            for expected_elem, core_elem in zip(expected, current_production_powers):
                error_diff = core_elem - expected_elem
                print("error diff = ", error_diff)
                assert (math.fabs(error_diff) < 1e-3)

        self.current_step += 1

        return action


class Agent_test_method_obs_are_prods_cut(Agent):
    """This function tests the method: observation.are_prods_cut"""
    def __init__(self, environment, node_to_change):
        super().__init__(environment)
        print("Agent_test_LimitOfProdsLost created...")

        self.current_step = 1
        self.node_to_change = node_to_change

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        # =================================== ACT FUNCTION STATS HERE ===================================
        action_space = self.environment.action_space
        print("prods_substations_ids = ", observation.productions_substations_ids)
        print("prods_substationsTopo = ", observation.productions_nodes)
        print("are prods cut         = ", observation.are_productions_cut)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)
        # get current configuration
        # current_conf, types = observation.get_nodes_of_substation(self.node_to_change)
        # print(f"Step[{self.current_step}]: current conf node [{self.node_to_change}] = {current_conf}")
        # print("types = ", types)

        if self.current_step == 1:
            print("we change the topology of node {}, curr_step = {}".format(self.node_to_change, self.current_step))

            expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
            # Choses a new switch configuration (binary array)
            target_configuration = np.zeros(expected_target_configuration_size)
            # we connect the PRODUCTION to BUSBAR 1
            target_configuration[0] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)
            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, self.node_to_change)
            assert np.all(current_configuration == target_configuration)

        if self.current_step == 2:
            # here we check that are_prods_cut show the information at the correct index.
            index = list(observation.productions_nodes).index(1)
            print("index = ", index)
            assert(observation.are_productions_cut[index] == 1)


        print("the action we return is, action = ", action)

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        print("========================================================")
        self.current_step += 1

        return action


class Agent_test_method_obs_are_loads_cut(Agent):
    """This function tests the method: observation.are_loads_cut"""
    def __init__(self, environment, node_to_change):
        super().__init__(environment)
        print("Agent_test_LimitOfLoadsLost created...")

        self.current_step = 1
        self.node_to_change = node_to_change

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        # =================================== ACT FUNCTION STATS HERE ===================================
        load_index = None
        action_space = self.environment.action_space
        print("loads_substations_ids = ", observation.loads_substations_ids)
        print("loads_substationsTopo = ", observation.loads_nodes)
        print("are loads cut         = ", observation.are_loads_cut)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)
        # get current configuration
        current_conf, types = observation.get_nodes_of_substation(self.node_to_change)
        print(f"Step[{self.current_step}]: current conf node [{self.node_to_change}] = {current_conf}")
        print("types = ", types)
        for i, type in enumerate(types):
            print("elem [{}] is of type [{}]".format(i, type))
            if type == ElementType.CONSUMPTION:
                load_index = i
        print("load index = ", load_index)

        if self.current_step == 1:
            print("we change the topology of node {}, curr_step = {}".format(self.node_to_change, self.current_step))

            expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
            # Choses a new switch configuration (binary array)
            target_configuration = np.zeros(expected_target_configuration_size)
            # we connect the LOAD to BUSBAR 1
            target_configuration[load_index] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)
            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, self.node_to_change)
            assert np.all(current_configuration == target_configuration)

        if self.current_step == 2:
            # here we check that are_loads_cut show the information at the correct index.
            index = list(observation.loads_nodes).index(1)
            print("index = ", index)
            assert(observation.are_loads_cut[index] == 1)


        print("the action we return is, action = ", action)

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        print("========================================================")
        self.current_step += 1

        return action


class Agent_test_Loss_Error(Agent):
    """This agent compares the expected (from chronics) and real loss (from observation) for the first 3 iterations. """
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_Loss_Error created...")

        self.current_step = 1
        self.line_to_cut = None

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        current_prod_powers = observation.active_productions
        current_load_powers = observation.active_loads
        print("current_load_powers = ", current_load_powers)
        print("current_prod_powers = ", current_prod_powers)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 1:
            expected_prods = [123.370285, 49.144115, 32.21891, 38.52085, 35.704945]
            expected_loads = [25.629642, 97.45528, 49.735317, 8.250563, 10.010641, 30.2604, 9.736532, 3.3486228, 7.0213113,
                              16.209476, 16.188494]
            sum_expected_prods = np.sum(expected_prods)
            sum_expected_loads = np.sum(expected_loads)
            expected_loss = sum_expected_prods - sum_expected_loads
            print("sum_expected prods = ", sum_expected_prods)
            print("sum_expected loads = ", sum_expected_loads)
            print("sum_expected loss  = ", expected_loss)

            real_loss = np.sum(current_prod_powers) - np.sum(current_load_powers)
            print("real_loss = ", real_loss)
            diff_loss = expected_loss - real_loss
            print("diff expected - real  LOSS = ", diff_loss)
            assert (math.fabs(diff_loss) < 1e-3)

        if self.current_step == 2:
            expected_prods = [104.072556, 43.576332, 31.90516, 32.831142, 32.747932]
            expected_loads = [21.07166, 87.22948, 43.29531, 6.9710474, 10.483086, 28.114975, 10.368015, 3.0358257, 5.108532,
                              12.720526, 12.9846325]
            sum_expected_prods = np.sum(expected_prods)
            sum_expected_loads = np.sum(expected_loads)
            expected_loss = sum_expected_prods - sum_expected_loads
            print("sum_expected prods = ", sum_expected_prods)
            print("sum_expected loads = ", sum_expected_loads)
            print("sum_expected loss  = ", expected_loss)

            real_loss = np.sum(current_prod_powers) - np.sum(current_load_powers)
            print("real_loss = ", real_loss)
            diff_loss = expected_loss - real_loss
            print("diff expected - real  LOSS = ", diff_loss)
            assert (math.fabs(diff_loss) < 1e-3)

        if self.current_step == 3:
            expected_prods = [134.51176, 56.608887, 0.0, 0.0, 46.029488]
            expected_loads = [18.838198, 86.235115, 44.783886, 6.563092, 9.875335, 24.161335, 6.824309, 3.2030978, 4.8327637,
                              12.320875, 13.072087]
            sum_expected_prods = np.sum(expected_prods)
            sum_expected_loads = np.sum(expected_loads)
            expected_loss = sum_expected_prods - sum_expected_loads
            print("sum_expected prods = ", sum_expected_prods)
            print("sum_expected loads = ", sum_expected_loads)
            print("sum_expected loss  = ", expected_loss)

            real_loss = np.sum(current_prod_powers) - np.sum(current_load_powers)
            print("real_loss = ", real_loss)
            diff_loss = expected_loss - real_loss
            print("diff expected - real  LOSS = ", diff_loss)
            assert (math.fabs(diff_loss) < 1e-3)

        self.current_step += 1

        return action



class Agent_test_NodesPhysics(Agent):
    """This agent checks for each node, so that the sum of inputs == sum of outputs"""
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_NodesPhysics created...")

        self.current_step = 1
        self.node_to_change = None

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        # =================================== ACT FUNCTION STATS HERE ===================================
        load_index = None
        action_space = self.environment.action_space
        print("lines_or_substations_ids = ", observation.lines_or_substations_ids)
        print("lines_ex_substations_ids = ", observation.lines_ex_substations_ids)
        print("loads_substations_ids = ", observation.loads_substations_ids)
        print("loads_substationsTopo = ", observation.loads_nodes)
        print("active loads          = ", observation.active_loads)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting

        if self.current_step == 1:
            print(observation)
            node = 5
            total_ingoing = np.sum(self.get_ingoing_node_power(observation, node))
            total_outgoing = np.sum(self.get_outgoing_node_power(observation, node))
            print("total ingoing power for node {} = {}".format(node, total_ingoing))
            print("total outgoing power for node {} = {}".format(node, total_outgoing))
            print("total ingoing = ", total_ingoing)
            print("total outgoing = ", total_outgoing)

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        print("========================================================")
        self.current_step += 1

        return action


    def get_ingoing_node_power(self, obs, node):
        indexes = []
        res = []
        for i, edge in enumerate(obs.lines_ex_substations_ids):
            if edge == node:
                indexes.append(i)
        print("indexes = ", indexes)
        print("obs.active_flows_extremity = ", obs.active_flows_extremity)
        for i in indexes:
            res.append(math.fabs(obs.active_flows_extremity[i]))
        return res

    def get_outgoing_node_power(self, obs, node):
        indexes = []
        res = []
        for i, edge in enumerate(obs.lines_or_substations_ids):
            if edge == node:
                indexes.append(i)
        print("indexes = ", indexes)
        print("obs.active_flows_origin = ", obs.active_flows_origin)
        for i in indexes:
            res.append(math.fabs(obs.active_flows_origin[i]))
        return res


class Agent_test_SoftOverflowBreakLimit(Agent):
    """This agent tests the variable n_timesteps_consecutive_soft_overflow_breaks=2, with thermal limit = 300 for line 6
    at t = 9,  line's 6 flow in ampere > 300, 322
    at t = 10, line's 6 flow in ampere > 300, 347
    at t = 11, it is the third consecutive timestep so we should have a line that is CUT because of SOFT OVERFLOW"""
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_SoftOverflowBreakLimit created...")

        self.current_step = 1
        self.line_to_cut = 9

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        # print(observation)
        print("lines_status = ", list(observation.lines_status.astype(int)))

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 9:
            # line's 6 flow in ampere > 300, 322    consecutive_steps = 1
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])

        if self.current_step == 10:
            # line's 6 flow in ampere > 300, 322    consecutive_steps = 2
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])

        if self.current_step == 11:
            # line's 6 flow == 0  consecutive_steps = 3  > n_timesteps_consecutive_soft_overflow_breaks, so line BREAKS
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])

        if self.current_step == 12:
            # line should still be broken
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])

        self.current_step += 1

        return action


class Agent_test_SoftOverflowIsBroken(Agent):
    """This agent tests the variable n_timesteps_soft_overflow_is_broken: 2  # number of timesteps a soft overflow
    broken line is broken, it is a follow up test for SoftOverflowBreakLimit.
    at t = 9,  line's 6 flow in ampere > 300, 322
    at t = 10, line's 6 flow in ampere > 300, 347
    at t = 11, it is the third consecutive timestep so we should have a line that is CUT because of SOFT OVERFLOW
    from this point, for 2 more steps we will try to set the line back up, and we should get Illegal Actions exception
    until t = 13
    at t = 12 down for 2 consecutive steps
    at t = 13, 3 consecutive steps > n_timesteps_soft_overflow_is_broken: 2, so we should be able to reconnect
    at t = 14, we check line is BACK ONLINE"""
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_SoftOverflowIsBroken created...")

        self.current_step = 1
        self.line_to_cut = 6

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        # print(observation)
        print("lines_status = ", list(observation.lines_status.astype(int)))

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 11:
            # line's 6 flow == 0  consecutive_steps = 3  > n_timesteps_consecutive_soft_overflow_breaks, so line BREAKS
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])
            # we try to set it up, but because n_timesteps_soft_overflow_is_broken: 2, it should not work and we should
            # get an IllegalAction exception.
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        if self.current_step == 12:
            # line should still be broken
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])
            # we try to set it up, but because n_timesteps_soft_overflow_is_broken: 2, it still should not work and we
            # should get an IllegalAction exception.
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        if self.current_step == 13:
            # line should still be broken
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])
            # line broke on step 11, so consecutive_broken_step is now 3, which now is superior to
            # n_timesteps_soft_overflow_is_broken: 2, so we should be able to turn it back ONLINE.
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        if self.current_step == 14:
            # line should be back UP
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])

        self.current_step += 1

        return action




class Agent_test_NodeTopoChangePersistence(Agent):
    """This agent switches a nodes topology and checks for 9 steps that it is still the same"""
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_NodeTopoChangePersistence created...")

        self.current_step = 1
        self.node_to_change = 9

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        # print(observation)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)
        # get current configuration
        current_conf, types = observation.get_nodes_of_substation(self.node_to_change)
        print(f"Current conf node [{self.node_to_change}] = {current_conf}")
        full_nodes_topo = get_verbose_node_topology(observation, action_space)
        print("verbose node topology = ", full_nodes_topo)
        # print("types = ", types)


        if self.current_step == 1:
            # we connect the fourth element to busbar 1
            target_configuration[3] = 1
            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)
            assert(full_nodes_topo == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        else:
            assert(full_nodes_topo == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 6669])

        self.current_step += 1

        return action


class Agent_test_LineChangePersistence(Agent):
    """This agent cuts a line and checks for 9 steps that it is still cut"""
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_LineChangePersistence created...")

        self.current_step = 1
        self.line_to_cut = 18

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        # print(observation)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        print("lines_status = ", list(observation.lines_status.astype(int)))

        if self.current_step == 1:
            # SWITCH OFF LINE 18
            print("we switch off line {}".format(self.line_to_cut))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])
        else:
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  0, 1])

        self.current_step += 1

        return action


class Agent_test_HardOverflowCoefTest(Agent):
    """This agent checks the variable hard_overflow_coefficient: 1.5,
    the flow of line 6 for each 15 steps is =[244, 210, 223, 214, 214, 237, 244, 286, 322, 347, 381, 310, 303, 324, 275]
    with the thermal limits of the line 6 being: 200, * (overflow_coeff) 1.5 = 300.
    So at step 9, the flow value of line 6 being 322, the line should break and we should have 0.
    we expected the result to be            = [244, 210, 223, 214, 214, 237, 244, 286, 0, 0, 0, 0]
    by trying to switch the line back up at each step, for step >= 9, we make sur the variable 
    n_timesteps_hard_overflow_is_broken: 2  # number of timesteps a hard overflow broken line is broken, works.
    t == 9, line just broke so switching it back up doesnt work. So we should have illegal action
    t == 10, n_timesteps_hard_overflow_is_broken: 2, so still illegal action,
    t == 11, we switch, it works but flow > 300 so it breaks again
    t == 12, broken consec timestep = 1
    t == 13, broken consec timestep = 2,
    t == 14, we can switch back up,
    t == 15, since flow < 300, it didnt break, and we end up with all line that are ON.
    expected results are = 
    [None, None, None, None, None, None, None, None, IllegalActionException(), IllegalActionException(), None,
     IllegalActionException(), IllegalActionException(), None, None]"""

    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_HardOverflowCoefTest created...")

        self.current_step = 1
        self.line_to_cut = 6
        self.flow_save_line6 = []

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        print(observation)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        print("lines_status = ", list(observation.lines_status.astype(int)))
        print("ampere flows = ", list(observation.ampere_flows.astype(int)))
        print("ampere flows line 6 = ", list(observation.ampere_flows.astype(int))[6])
        self.flow_save_line6.append(list(observation.ampere_flows.astype(int))[6])

        if 9 <= self.current_step < 15:
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])

        if self.current_step == 15:
            print("all saved flows = ", self.flow_save_line6)
            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])


        self.current_step += 1

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        return action


########################################################################################################################
########################################################################################################################
########################################################################################################################

def test_core_Agent_test_limitOfProdsLost():
    """This function creates an Agent that tests the config variable: max_number_prods_game_over: 1
        t = 1, it disconnects 1 prod
        t = 2, it disconnects second prod, ==> causes a Game Over
        t = 3, it checks that obs.productions_nodes = [0, 0, 0, 0, 0], ie, that the game reset.
        This function checks that the game ended because of TooManyProductionsCut"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_LimitOfProdsLost(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, True, False])
    for i, action in enumerate(actions_recap):
        if i == 1:
            assert(isinstance(action, TooManyProductionsCut))
        else:
            assert(action == None)


def test_core_Agent_test_limitOfLoadsLost():
    """This function creates an Agent that tests the config variable: max_number_loads_game_over: 1
        t = 1, it disconnects 1 load
        t = 2, it disconnects second load, ==> causes a Game Over
        t = 3, it checks that obs.loads_nodes = [0, 0, ... , 0, 0], ie, that the game reset.
        This function checks that the game ended because of TooManyProductionsCut"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_LimitOfLoadsLost(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, True, False])
    for i, action in enumerate(actions_recap):
        if i == 1:
            assert(isinstance(action, TooManyConsumptionsCut))
        else:
            assert(action == None)




def test_core_Agent_test_InputProdValues():
    """This function creates an Agent that tests the correct loading of input Prod values"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_InputProdValues(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False])
    assert(list(actions_recap) == [None, None, None])


def test_core_Agent_test_InputLoadValues():
    """This function creates an Agent that tests the correct loading of input Load values"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_InputLoadValues(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False])
    assert(list(actions_recap) == [None, None, None])



# def test_core_Agent_test_SumInsEqualsSumOuts_Power():
#     """This function creates an Agent that tests """
#     parameters = "./tests/parameters/default14_for_tests/"
#     print("Parameters used = ", parameters)
#     game_level = "level0"
#     loop_mode = "natural"
#     start_id = 0
#     game_over_mode = "soft"
#     renderer_latency = 1
#     render = False
#     # render = False
#     niter = 3
#
#     env_class = RunEnv
#
#     # Instantiate environment and agent
#     env = env_class(parameters_folder=parameters, game_level=game_level,
#                     chronic_looping_mode=loop_mode, start_id=start_id,
#                     game_over_mode=game_over_mode, renderer_latency=renderer_latency)
#     agent = Agent_test_InputLoadValues(env)
#     # Instantiate game runner and loop
#     runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
#     final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
#     print("Obtained a final reward of {}".format(final_reward))
#     print("game_overs = ", game_overs)
#     print("actions_recap = ", actions_recap)





def test_core_Agent_test_method_obs_are_prods_cut():
    """This function tests the method: observation.are_prods_cut"""
    parameters = "./tests/parameters/default14_for_tests_alpha/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 2

    env_class = RunEnv
    nodes_to_change = [1, 2, 3, 6, 8]

    for node in nodes_to_change:
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO NODE [{}] OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO".format(node))
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        # Instantiate environment and agent
        env = env_class(parameters_folder=parameters, game_level=game_level,
                        chronic_looping_mode=loop_mode, start_id=start_id,
                        game_over_mode=game_over_mode, renderer_latency=renderer_latency)
        agent = Agent_test_method_obs_are_prods_cut(env, node)
        # Instantiate game runner and loop
        runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
        print("Obtained a final reward of {}".format(final_reward))
        print("game_overs = ", game_overs)
        print("actions_recap = ", actions_recap)
        assert(niter == len(game_overs) == len(actions_recap))
        assert(list(game_overs) == [False, False])
        assert(list(actions_recap) == [None, None])





def test_core_Agent_test_method_obs_are_loads_cut():
    """This function tests the method: observation.are_loads_cut"""
    parameters = "./tests/parameters/default14_for_tests_alpha/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 2

    env_class = RunEnv
    nodes_to_change = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]


    for node in nodes_to_change:
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO NODE [{}] OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO".format(node))
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        # Instantiate environment and agent
        env = env_class(parameters_folder=parameters, game_level=game_level,
                        chronic_looping_mode=loop_mode, start_id=start_id,
                        game_over_mode=game_over_mode, renderer_latency=renderer_latency)
        agent = Agent_test_method_obs_are_loads_cut(env, node)
        # Instantiate game runner and loop
        runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
        print("Obtained a final reward of {}".format(final_reward))
        print("game_overs = ", game_overs)
        print("actions_recap = ", actions_recap)
        assert(niter == len(game_overs) == len(actions_recap))
        assert(list(game_overs) == [False, False])
        assert(list(actions_recap) == [None, None])




def test_core_Agent_test_Loss_Error():
    """This function creates an Agent that compares the expected (from chronics) and real loss (from observation)
    for the first 3 iterations"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_Loss_Error(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False])
    assert(list(actions_recap) == [None, None, None])





def test_core_Agent_test_NodesPhysics():
    """This function creates an Agent checks for each node, so that the sum of inputs == sum of outpus"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = False
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_NodesPhysics(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    # assert(list(game_overs) == [False, False, True])
    # assert(list(actions_recap) == [None, None, None])





def test_core_Agent_test_SoftOverflowBreakLimit():
    """This function creates an Agent that checks variable: n_timesteps_consecutive_soft_overflow_breaks = 2"""
    parameters = "./tests/parameters/default14_for_tests_alpha/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 12

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_SoftOverflowBreakLimit(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False, False, False, False, False, False, False, False, False, False])
    assert(list(actions_recap) == [None, None, None, None, None, None, None, None, None, None, None, None])


def test_core_Agent_test_SoftOverflowIsBroken():
    """This function creates an Agent that checks variable:
    n_timesteps_soft_overflow_is_broken: 2  # number of timesteps a soft overflow broken line is broken"""
    parameters = "./tests/parameters/default14_for_tests_alpha/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 14

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_SoftOverflowIsBroken(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False, False, False, False, False, False, False, False, False, False,
                                False, False])
    for i, action in enumerate(actions_recap):
        if i == 10 or i == 11:
            assert(isinstance(action, IllegalActionException))
        else:
            assert(action is None)



def test_core_Agent_test_NodeTopoChangePersistence():
    """This function creates an Agent that switches a nodes topology and checks for 9 steps that it is still the same"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 10

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_NodeTopoChangePersistence(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False, False, False, False, False, False, False, False])
    assert(list(actions_recap) == [None, None, None, None, None, None, None, None, None, None])





def test_core_Agent_test_LineChangePersistence():
    """This function creates an Agent that cut a line and checks for 9 steps that it is still cut"""
    parameters = "./tests/parameters/default14_for_tests/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 10

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_LineChangePersistence(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False, False, False, False, False, False, False, False])
    assert(list(actions_recap) == [None, None, None, None, None, None, None, None, None, None])


def test_core_Agent_test_HardOverflowCoefTest():
    """This function creates an Agent that checks variable: hard_overflow_coefficient: 1.5 which is # % of line capacity
     usage above which a line will break bc of hard overflow, check Agent description for more info"""
    parameters = "./tests/parameters/default14_for_tests_hard_overflow/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # render = True
    niter = 15

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_HardOverflowCoefTest(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    assert(list(game_overs) == [False, False, False, False, False, False, False, False, False, False, False, False,
                                False, False, False])
    for i, action in enumerate(actions_recap):
        if i in [8, 9, 11, 12]:
            assert(isinstance(action, IllegalActionException))
        else:
            assert(action is None)


# test_core_Agent_test_InputProdValues()
# test_core_Agent_test_InputLoadValues()
# test_core_Agent_test_limitOfProdsLost()
# test_core_Agent_test_limitOfLoadsLost()
# test_core_Agent_test_method_obs_are_prods_cut()
# test_core_Agent_test_method_obs_are_loads_cut()
# test_core_Agent_test_Loss_Error()
# test_core_Agent_test_NodeTopoChangePersistance()
# test_core_Agent_test_LineChangePersistance()
# test_core_Agent_test_SoftOverflowBreakLimit()
# test_core_Agent_test_SoftOverflowIsBroken()
test_core_Agent_test_HardOverflowCoefTest()



# to finish
#test_core_Agent_test_NodesPhysics()
