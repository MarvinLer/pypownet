"""This file constains basic tests for pypownet."""

import sys
import os

#add current path to sys.path
# print("current path = ", current_pwd)
# sys.path.append(os.path.abspath("../"))
#
print("dir = ", dir())
print("syspath = ", sys.path)

from pypownet.environment import RunEnv
from pypownet.runner import Runner
import pypownet.agent


def test_first():
    a = 1
    b = 1
    assert(a == b)


def test_second():
    a = "A"
    b = "A"
    assert(a == b)


def test_limit_line_switching():
    """
    This function creates an agent that switches a line, then tries to switch it again. (But should be nullified because of input param
    "n_timesteps_actionned_line_reactionable: 3", then after 3 steps, we switch it back up.
    """
    parameters = "./tests/parameters/default14/"
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    agent = "TestAgent"
    agent_class = eval('pypownet.agent.{}'.format(agent))
    print(f"Function: {__name__} is run with Agent : {agent}")
    # agent = "RandomLineSwitch"
    niter = 6
    #####################################################
    lines_to_cut = [17, 18, 0, 1, 2]
    # lines_to_cut = [i for i in range(19)]
    ####################################################

    for line_to_cut in lines_to_cut:
        print("########################## Tests for line_to_cut = [{}] ##########################".format(line_to_cut))
        env_class = RunEnv

        # Instantiate environment and agent
        env = env_class(parameters_folder=parameters, game_level=game_level,
                        chronic_looping_mode=loop_mode, start_id=start_id,
                        game_over_mode=game_over_mode, renderer_latency=renderer_latency)
        agent = agent_class(env, line_to_cut)
        # Instantiate game runner and loop
        runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward = runner.loop(iterations=niter)
        print("Obtained a final reward of {}".format(final_reward))


def test_basic_node_switching():
    """This function creates an agent that switches a node configurations, then tries to switch it again.
    (But should be nullified because of input param "n_timesteps_actionned_line_reactionable: 3", then after 3 steps,
     we switch it back up.
    """
    pass




test_limit_line_switching()
