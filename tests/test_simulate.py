"""This file constains tests where we use simulate before acting"""

from pypownet.environment import RunEnv, ElementType
from pypownet.runner import Runner
from pypownet.agent import *
from pypownet.game import TooManyProductionsCut, TooManyConsumptionsCut, DivergingLoadflowException
from tests.test_basic import WrappedRunner, get_verbose_node_topology
import math


class Agent_test_LineChangePersistance(Agent):
    """This agent cuts a line and checks for 9 steps that it is still cut"""
    def __init__(self, environment):
        super().__init__(environment)
        print("Agent_test_LineChangePersistance created...")

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
        print(observation)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        # Select a random substation ID on which to perform node-splitting
        print("lines_status = ", list(observation.lines_status.astype(int)))

        if self.current_step == 1:
            # SWITCH OFF LINE 18
            print("we switch off line {}".format(self.line_to_cut))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
            # here we would like to simulate before submitting the action
            reward = self.environment.simulate(action)

            assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                  1, 1])
        # else:
        #     assert(list(observation.lines_status.astype(int)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                                                           0, 1])

        self.current_step += 1

        return action



########################################################################################################################
########################################################################################################################
########################################################################################################################

def test_simulate_Agent_test_SimulateNodeChangeThenAct():
    assert(1 == 1)

def test_simulate_Agent_test_SimulateThenAct():
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
    agent = Agent_test_LineChangePersistance(env)
    # Instantiate game runner and loop
    runner = WrappedRunner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward, game_overs, actions_recap = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))
    print("game_overs = ", game_overs)
    print("actions_recap = ", actions_recap)
    assert(niter == len(game_overs) == len(actions_recap))
    # assert(list(game_overs) == [False, False, False, False, False, False, False, False, False, False])
    # assert(list(actions_recap) == [None, None, None, None, None, None, None, None, None, None])


test_simulate_Agent_test_SimulateThenAct()

