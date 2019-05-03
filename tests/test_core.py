"""This file contains all core tests."""

import sys


# sys.path.insert(0, "/home/mozgawamar/Documents/pypownet_last_version/pypownet/")
# print("sys path = ", sys.path)

from pypownet.environment import RunEnv, ElementType
from pypownet.runner import Runner
from pypownet.agent import *
from pypownet.game import TooManyProductionsCut, TooManyConsumptionsCut
from tests.test_basic import WrappedRunner
import math





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


# test_core_Agent_test_InputProdValues()
# test_core_Agent_test_InputLoadValues()
# test_core_Agent_test_limitOfProdsLost()
# test_core_Agent_test_limitOfLoadsLost()
# test_core_Agent_test_method_obs_are_prods_cut()
# test_core_Agent_test_method_obs_are_loads_cut()
# test_core_Agent_test_Loss_Error()
