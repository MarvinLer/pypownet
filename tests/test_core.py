"""This file contains all core tests."""

from pypownet.environment import RunEnv
from pypownet.runner import Runner
from pypownet.agent import *


class Agent_test_LimitOfProdsLost(Agent):
    """This agent tests the restriction : max_number_prods_game_over: 2
        t = 1, the agent switches off line 0 and 1 to disconnect node [1]
        t = 2, it observes that the line X has been switched off
        t = 2, it tries to switch the line back on, but should be dropped because of the
        restriction n_timesteps_actionned_line_reactionable: 3
        t = 3, he observes that it indeed didnt do anything, because of the restriction, we did not managed to switch it back on
        t = 3, he tries to switch it on again
        t = 4, he observes that it indeed didnt do anything, because of the restriction, we did not managed to switch it back on
        t = 4, he tries to switch it on again
        t = 5, THE "SWITCH BACK ON" WORKED
        t = 5, he tries to cut it again
        t = 6, must be restricted again. Should still be back on.
    """

    def __init__(self, environment, line_to_cut):
        super().__init__(environment)
        print("TestAgent_LineLimitSwitching created...")

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
        print("prods_substations_ids = ", observation.productions_substations_ids)
        print("real power produced by generators = ", observation.active_productions)
        print("are_prods_cut = ", observation.are_productions_cut)
        print("lines_status = ", observation.lines_status)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 1:
            print("we switch off line {}, curr_step = {}".format([0, 1], self.current_step))
            action_space.set_lines_status_switch_from_id(action=action, line_id=0, new_switch_value=1)
            action_space.set_lines_status_switch_from_id(action=action, line_id=1, new_switch_value=1)

        elif self.current_step == 2:
            # Here we just OBSERVE that lines 0 and 1 have been cut.
            assert (observation.lines_status[0] == 0)
            assert (observation.lines_status[1] == 0)

            # print("we switch off line {}, curr_step = {}".format([2, 3, 4], self.current_step))
            # action_space.set_lines_status_switch_from_id(action=action, line_id=2, new_switch_value=1)
            # action_space.set_lines_status_switch_from_id(action=action, line_id=3, new_switch_value=1)

        print("the action we return is, action = ", action)

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        print("========================================================")
        self.current_step += 1

        return action


# def test_core_Agent_test_limitOfProdsLost():
#     """This function creates an Agent that tests all the Topological changes of all the Substations"""
#     parameters = "./tests/parameters/default14_for_tests/"
#     print("Parameters used = ", parameters)
#     game_level = "level0"
#     loop_mode = "natural"
#     start_id = 0
#     game_over_mode = "soft"
#     renderer_latency = 1
#     render = True
#     # render = False
#     niter = 6
#
#     env_class = RunEnv
#
#     # Instantiate environment and agent
#     env = env_class(parameters_folder=parameters, game_level=game_level,
#                     chronic_looping_mode=loop_mode, start_id=start_id,
#                     game_over_mode=game_over_mode, renderer_latency=renderer_latency)
#     agent = Agent_test_LimitOfProdsLost(env, 1)
#     # Instantiate game runner and loop
#     runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
#     final_reward = runner.loop(iterations=niter)
#     print("Obtained a final reward of {}".format(final_reward))


class Agent_test_InputProdValues(Agent):
    """This agent compares the values found in the chronics and internal observation for 3 steps. """
    def __init__(self, environment, line_to_cut):
        super().__init__(environment)
        print("TestAgent_LineLimitSwitching created...")

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
                assert (error_diff < 1e-5)

        if self.current_step == 2:
            expected = [104.072556, 43.576332, 31.90516, 32.831142, 32.747932]
            for expected_elem, core_elem in zip(expected, current_production_powers):
                error_diff = core_elem - expected_elem

                print("error diff = ", error_diff)
                assert (error_diff < 1e-5)

        if self.current_step == 3:
            expected = [134.51176, 56.608887, 0.0, 0.0, 46.029488]
            for expected_elem, core_elem in zip(expected, current_production_powers):
                error_diff = core_elem - expected_elem
                print(core_elem, expected_elem)
                print("error diff = ", error_diff)
                assert (error_diff < 1e-3)

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
    agent = Agent_test_InputProdValues(env, 1)
    # Instantiate game runner and loop
    runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))


# def test_check_load_prods_values():
#     pass


test_core_Agent_test_InputProdValues()

# test_core_Agent_test_limitOfProdsLost()
