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
from pypownet.agent import *


class Agent_test_LineLimitSwitching(Agent):
    """This agent tests the restriction : n_timesteps_actionned_line_reactionable: 3
        t = 1, the agent switches off line X,
        t = 2, he observes that the line X has been switched off
        t = 2, he tries to switch the line back on, but should be dropped because of the
        restriction n_timesteps_actionned_line_reactionable: 3
        t = 3, he observes that it indeed didnt do anything, because of the restriction, we did not managed to switch it back on
        t = 3, he tries to switch it on again
        t = 4, he observes that it indeed didnt do anything, because of the restriction, we did not managed to switch it back on
        t = 4, he tries to switch it on again
        t = 5, THE "SWITCH BACK ON" WORKED
        t = 5, he tries to cut it again
        t = 6, must be restricted again. Should still be back on.
    """
    def __init__(self, environment,  line_to_cut):
        super().__init__(environment)
        print("TestAgent_LineLimitSwitching created...")

        self.current_step = 1
        self.ioman = ActIOnManager(destination_path='testAgent.csv')
        self.line_to_cut = line_to_cut

    def act(self, observation):
        print("current step = ", self.current_step)
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
        # print("action_space.lines_status_subaction_length = ", action_space.lines_status_subaction_length)
        print("lines_status = ", observation.lines_status)

        # test before action the basic case has 14 nodes and 20 lines
        assert action_space.lines_status_subaction_length == 20

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        # t = 1, the agent switches off line X,
        # t = 2, he observes that the line X has been switched off
        # t = 2, he tries to switch the line back on, but should be dropped because of the
        # restriction n_timesteps_actionned_line_reactionable: 3
        # t = 3, he observes that it indeed didnt do anything, because of the restriction, we did not managed to cut it
        # t = 3, he tries to put it back on again
        # t = 4, he observes that it indeed didnt do anything, because of the restriction, we did not managed to cut it
        # t = 4, he tries to put it back on again
        # t = 5, THE SWITCH OF WORKED
        # t = 5, he tries to cut it again
        # t = 6, he observes that it is still on.

        if self.current_step == 1:
            # SWITCH OFF LINE X
            print("we switch off line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        elif self.current_step == 2:
            # Here we just OBSERVE that the first line 0 has been switched off
            assert (observation.lines_status[self.line_to_cut] == 0)

            # we try to switch the line back on, but should be dropped because of the
            # restriction n_timesteps_actionned_line_reactionable: 3
            print("we try switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        elif self.current_step == 3:
            # Here, because of the restriction, we did not managed to switch it ON.
            assert (observation.lines_status[self.line_to_cut] == 0)

            # we try again, it still should not work.
            print("we try switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        elif self.current_step == 4:
            # Here, because of the restriction, we did not managed to switch it ON
            assert (observation.lines_status[self.line_to_cut] == 0)

            # SWITCH BACK ON LINE X # NOW IT SHOULD WORK, because it is the last (third) step, AND IT SHOULD BE VISIBLE
            # IN STEP 5
            print("we try switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        elif self.current_step == 5:
            # Here, because of the restriction has ended, we should see the line ON
            assert (observation.lines_status[self.line_to_cut] == 1)

            # LAST CHECK:  SWITCH BACK LINE X to OFF. It should not work.
            print("we switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
            action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        elif self.current_step == 6:
            # Here, because of the restriction, we should still see the line, ON
            assert (observation.lines_status[self.line_to_cut] == 1)

        # Dump best action into stored actions file
        self.ioman.dump(action)
        self.current_step += 1
        print("the action we return is, action = ", action)

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action)) ).all())
        print("========================================================")

        return action


class Agent_test_NodeLimitSwitching(Agent):
    """This agent is used for testing purposes"""
    def __init__(self, environment,  node_to_change):
        super().__init__(environment)
        print("TestAgent_NodeLimitSwitching created...")

        self.current_step = 1
        self.ioman = ActIOnManager(destination_path='testAgent.csv')
        self.node_to_change = node_to_change

    def act(self, observation):
        print("current step = ", self.current_step)
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
        nodes_ids = self.get_verbose_node_topology(observation)
        print("nodes_ids = ", nodes_ids)

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if self.current_step == 1:
            # CHANGE TOPOLOGY OF NODE_TO_CHANGE
            print("we change the topology of node {}, curr_step = {}".format(self.node_to_change, self.current_step))

            # Select a random substation ID on which to perform node-splitting
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

        elif self.current_step == 2:
            new_supposed_name_str = "666" + str(self.node_to_change)
            print(new_supposed_name_str)
            # Here we just OBSERVE that the NODE_TO_CHANGE has the production now connected to BUSBAR1
            index_res = list(observation.productions_substations_ids).index(self.node_to_change)
            assert(observation.productions_nodes[index_res] == 1)


            # assert (observation.lines_status[self.line_to_cut] == 0)


            # we try to switch the line back on, but should be dropped because of the
            # restriction n_timesteps_actionned_line_reactionable: 3
            # print("we try switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
            # action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)

        # elif self.current_step == 3:
        #     # Here, because of the restriction, we did not managed to cut it.
        #     assert (observation.lines_status[self.line_to_cut] == 0)
        #
        #     # we try again, it still should not work.
        #     print("we try switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
        #     action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
        #
        # elif self.current_step == 4:
        #     # Here, because of the restriction, we did not managed to cut it.
        #     assert (observation.lines_status[self.line_to_cut] == 0)
        #
        #     # SWITCH BACK ON LINE X # NOW IT SHOULD WORK, because it is the last (third) step, AND IT SHOULD BE VISIBLE
        #     # IN STEP 5
        #     print("we try switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
        #     action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
        #
        # elif self.current_step == 5:
        #     # Here, because of the restriction, we did not managed to cut it.
        #     assert (observation.lines_status[self.line_to_cut] == 1)
        #
        #     # LAST CHECK:  SWITCH BACK ON LINE X # NOW IT SHOULD WORK AND IT SHOULD BE VISIBLE IN STEP 6
        #     print("we switch back line {}, curr_step = {}".format(self.line_to_cut, self.current_step))
        #     action_space.set_lines_status_switch_from_id(action=action, line_id=self.line_to_cut, new_switch_value=1)
        #
        # elif self.current_step == 6:
        #     # Here, because of the restriction, we should still see the line, ON
        #     assert (observation.lines_status[self.line_to_cut] == 1)

        # Dump best action into stored actions file
        self.ioman.dump(action)
        self.current_step += 1
        print("the action we return is, action = ", action)

        return action

    def get_verbose_node_topology(self, obs):
        """This function returns the <real> topology, ie, split nodes are displayed"""
        action_space = self.environment.action_space
        n_bars = len(action_space.substations_ids)
        # The following code allows to get just the nodes ids
        # where there are elements connected. It also considerer
        # the split node action.
        all_sub_conf = []
        for sub_id in obs.substations_ids:
            sub_conf, _ = obs.get_nodes_of_substation(sub_id)
            all_sub_conf.append(sub_conf)

        # print("all sub conf = ", all_sub_conf)

        nodes_ids = np.arange(1, n_bars + 1)
        for i in range(len(all_sub_conf)):
            # Check if all elements in sub (i)
            # are connected to busbar B1.
            # print(np.equal(all_sub_conf[i], np.ones(len(all_sub_conf[i]))))
            # print("np.ones(len(all_sub_conf[i] = ", np.ones(len(all_sub_conf[i])))
            # print("type = ", type(np.equal(all_sub_conf[i], np.ones(len(all_sub_conf[i])))))
            if (np.equal(all_sub_conf[i], np.ones(len(all_sub_conf[i])))).all():
                # Remove the existing node.
                nodes_ids = np.delete(nodes_ids, i)
                # And create a new node.
                nodes_ids = np.append(nodes_ids, int(str(666) + str(i + 1)))
            # Check if one or more elements
            # are connected to busbar B1.
            elif np.sum(all_sub_conf[i]) > 0:
                nodes_ids = np.append(nodes_ids, int(str(666) + str(i + 1)))

        nodes_ids = list(nodes_ids)
        # print("CUSTOM nodes ids = ", nodes_ids)

        for node in obs.substations_ids:
            conf = obs.get_nodes_of_substation(node)
            print(f"node [{node}] config = {conf}")
            ii = 0
            for elem, type in zip(conf[0], conf[1]):
                print(f"element n°[{ii}] connected to BusBar n°[{elem}] is a [{type}]")
                ii += 1
                return nodes_ids


def test_first():
    a = 1
    b = 1
    assert(a == b)


def test_second():
    a = "A"
    b = "A"
    assert(a == b)


def test_limit_same_line_switching():
    """
    This function creates an agent that switches a line, then tries to switch it again. (But should be nullified because
     of input param "n_timesteps_actionned_line_reactionable: 3", then after 3 steps, we switch it back up.
    """
    parameters = "./tests/parameters/default14/"
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    agent = "TestAgent"
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
        agent = Agent_test_LineLimitSwitching(env, line_to_cut)
        # Instantiate game runner and loop
        runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward = runner.loop(iterations=niter)
        print("Obtained a final reward of {}".format(final_reward))


def test_limit_same_node_switching():
    """This function creates an agent that switches a node topology, then tries to switch it again (same).
    (But should be nullified because of input param "n_timesteps_actionned_line_reactionable: 3", then after 3 steps,
     we switch it back up.
    """
    parameters = "./tests/parameters/default14/"
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # agent = "RandomLineSwitch"
    niter = 2
    #####################################################
    # nodes_to_change = [17, 18, 0, 1, 2]
    # nodes_to_change = [1, 2, 3, 4]
    # nodes_to_change = [i for i in range(1, 15)]

    # in DEFAULT 14, the nodes with Production are: 1, 2, 3, 6, 8
    nodes_to_change = [1, 2, 3, 6, 8]

    ####################################################

    for node_to_change in nodes_to_change:
        print("######################## Tests for node_to_cut = [{}] ########################".format(node_to_change))
        env_class = RunEnv

        # Instantiate environment and agent
        env = env_class(parameters_folder=parameters, game_level=game_level,
                        chronic_looping_mode=loop_mode, start_id=start_id,
                        game_over_mode=game_over_mode, renderer_latency=renderer_latency)
        agent = Agent_test_NodeLimitSwitching(env, node_to_change)
        # Instantiate game runner and loop
        runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward = runner.loop(iterations=niter)
        print("Obtained a final reward of {}".format(final_reward))




# test_limit_same_node_switching()
