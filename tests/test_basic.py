"""This file constains basic tests for pypownet."""

import sys
import os

# add current path to sys.path
# print("current path = ", current_pwd)
# sys.path.append(os.path.abspath("../"))
#
# print("dir = ", dir())
# print("syspath = ", sys.path)

from pypownet.environment import RunEnv
from pypownet.runner import Runner
from pypownet.agent import *


class Agent_test_BasicSubstationTopologyChange(Agent):
    """This agent changes all the possible connections on a given node, and checks from Observations that it did occur.
    We check this way all changes, PRODS, LOADS, OR, EX
    This test has a specific folder with n_timesteps_actionned_node_reactionable = 0, in order to be able to change
    nodes easily"""
    def __init__(self, environment, node_to_change=None):
        super().__init__(environment)
        print("Agent_test_BasicSubstationTopologyChange created...")
        self.node_to_change = node_to_change
        self.current_step = 1

        action_space = self.environment.action_space
        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)
        # give a list [0, 0, 0] ==> it generates [1, 0, 0], [0, 1, 0], [0, 0, 1] lists..
        self.node_topology_generator = create_node_topology_generator(target_configuration)
        self.all_topologies = []
        for topo in self.node_topology_generator:
            self.all_topologies.append(topo)
        print("All topologies = ", self.all_topologies)

    def act(self, observation):
        # Because we receive an observation = numpy.ndarray, we have to change it into class Observation
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        ################################################################################################################
        action_space = self.environment.action_space
        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        current_conf, types = observation.get_nodes_of_substation(self.node_to_change)
        print(f"Step[{self.current_step}]: current conf node [{self.node_to_change}] = {current_conf}")
        print("types = ", types)

        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(self.node_to_change)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)

        if self.current_step == 1:
            new_conf = self.all_topologies[self.current_step - 1]
            print(f"Step [{self.current_step}], we test configuration: {new_conf}")
            action_space.set_substation_switches_in_action(action, self.node_to_change, new_conf)

        elif self.current_step == 2:
            # t = 2, check if substation configurations are identical to t == 1
            assert(list(current_conf) == self.all_topologies[self.current_step - 2])

            new_conf = self.all_topologies[self.current_step - 1]
            print(f"Step [{self.current_step}], we test configuration: {new_conf}")

            final_conf = get_differencial_topology(new_conf, current_conf)
            print(f"Step [{self.current_step}], so final constructed conf = : {final_conf}")
            action_space.set_substation_switches_in_action(action, self.node_to_change, final_conf)

        elif self.current_step == 3:
            # t = 3, check if substation configurations are identical to t == t - 1
            assert(list(current_conf) == self.all_topologies[self.current_step - 2])

            new_conf = self.all_topologies[self.current_step - 1]
            print(f"Step [{self.current_step}], we test configuration: {new_conf}")

            final_conf = get_differencial_topology(new_conf, current_conf)
            print(f"Step [{self.current_step}], so final constructed conf = : {final_conf}")
            action_space.set_substation_switches_in_action(action, self.node_to_change, final_conf)

        elif self.current_step == 4:
            # t = 4, check if substation configurations are identical to t == t - 1
            assert(list(current_conf) == self.all_topologies[self.current_step - 2])

            if len(current_conf) > 3:
                new_conf = self.all_topologies[self.current_step - 1]
                print(f"Step [{self.current_step}], we test configuration: {new_conf}")
                action_space.set_substation_switches_in_action(action, self.node_to_change, new_conf)

            elif len(current_conf) == 3: # now reset to 000
                new_conf = [0, 0, 0]
                final_conf = get_differencial_topology(new_conf, current_conf)
                print(f"Step [{self.current_step}], we test configuration: {new_conf}")
                action_space.set_substation_switches_in_action(action, self.node_to_change, new_conf)



        elif self.current_step == 5:
            # t = 4, check if substation configurations are identical to t == t - 1
            if len(current_conf) < 4:
                return action
            else:
                assert(list(current_conf) == self.all_topologies[self.current_step - 2])


        print("END OF Step [{}], we do nothing : {}".format(self.current_step, np.equal(action.as_array(),
                                                                                        np.zeros(len(action))).all()))
        print("========================================================")
        self.current_step += 1
        return action


class Agent_test_MaxNumberActionnedSubstations(Agent):
    """This agent tests the restriction: max_number_actionned_substations = 2
        t = 1, Agent changes the topology of 3 nodes, ==> should be rejected because of the restriction.
        t = 2, check if substation configurations are identical to t == 1
        t = 2, Agent changes the topology of 2 nodes, ==> should work.
        t = 3, check if substation configurations of node [X, X] changed and the rest is identical to t == 1
        t = 3, Agent changes the topology of 4 nodes, ==> should be rejected because of the restriction.
        t = 4, check if substation configurations of node [X, X] changed and the rest is identical to t == 1
    """
    def __init__(self, environment, line_to_cut=None):
        super().__init__(environment)
        print("Agent_test_MaxNumberActionnedNodes created...")
        self.current_step = 1

    def act(self, observation):
        # Because we receive an observation = numpy.ndarray, we have to change it into class Observation
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        ################################################################################################################
        action_space = self.environment.action_space
        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        substation_topo = get_verbose_node_topology(observation, action_space)
        print("prods_substations_ids = ", observation.productions_substations_ids)
        print("prods_substationsTopo = ", observation.productions_nodes)
        print("loads_substations_ids = ", list(observation.loads_substations_ids.astype(int)))
        print("loads_substationsTopo = ", observation.loads_nodes)
        print("substation_topo = ", substation_topo)
        conf, types = observation.get_nodes_of_substation(5)
        print("conf node11 = ", conf)

        if self.current_step == 1:
            # changes the topology of 3 nodes
            # self.change_node_topology(4, action_space, action)
            # action_space.set_substation_switches_in_action(action, 4, [1, 0, 0, 0, 0, 0])
            # action_space.set_substation_switches_in_action(action, 4, [0, 1, 0, 0, 0, 0])
            # action_space.set_substation_switches_in_action(action, 4, [0, 0, 1, 0, 0, 0])
            # action_space.set_substation_switches_in_action(action, 4, [0, 0, 0, 1, 0, 0])
            # action_space.set_substation_switches_in_action(action, 4, [0, 0, 0, 0, 1, 0])
            # action_space.set_substation_switches_in_action(action, 4, [0, 0, 0, 0, 0, 1])
            action_space.set_substation_switches_in_action(action, 5, [1, 0, 0, 0, 0])
            # action_space.set_substation_switches_in_action(action, 11, [0, 1, 0])
            # action_space.set_substation_switches_in_action(action, 11, [0, 0, 1])


        if self.current_step == 2:
            # check if substation configurations are identical to t == 1
            pass

        print("the action we return is, action = ", action)

        print("action_space_get_switches_conf[5] = ", action_space.get_substation_switches_in_action(action, 5))
        print("END OF Step [{}], we do nothing : {}".format(self.current_step, np.equal(action.as_array(),
                                                                                 np.zeros(len(action))).all()))
        print("========================================================")

        self.current_step += 1
        return action

    def change_node_topology(self, substation_id, action_space, action):
        # Select a random substation ID on which to perform node-splitting
        expected_target_configuration_size = action_space.get_number_elements_of_substation(substation_id)
        # Choses a new switch configuration (binary array)
        target_configuration = np.zeros(expected_target_configuration_size)
        # we connect the CONSUMPTION to BUSBAR 1
        target_configuration[3] = 1
        print("target_configuration =", target_configuration)

        action_space.set_substation_switches_in_action(action=action, substation_id=substation_id,
                                                       new_values=target_configuration)
        # Ensure changes have been done on action
        current_configuration, _ = action_space.get_substation_switches_in_action(action, substation_id)
        assert np.all(current_configuration == target_configuration)

class Agent_test_MaxNumberActionnedLines(Agent):
    """This agent tests the restriction: max_number_actionned_lines = 2
        t = 1, Agent switches off 3 lines, ==> should be rejected because of the restriction.
        t = 2, check if all lines are ON.
        t = 2, Agent switches off 2 lines, ==> should work.
        t = 3, check if 2 lines are OFF and rest of lines are still ON
        t = 3, Agent switches of 4 lines, ==> should be rejected because of the restriction.
        t = 4, check if 2 lines are still OFF and rest of lines are still ON
    """

    def __init__(self, environment, line_to_cut=None):
        super().__init__(environment)
        print("Agent_test_MaxNumberActionnedLines created...")
        self.current_step = 1
        self.lines_to_cut = [1, 2]

    def act(self, observation):
        # Because we receive an observation = numpy.ndarray, we have to change it into class Observation
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space
        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        lines_status = observation.lines_status
        print("lines_status = ", lines_status)

        if self.current_step == 1:
            # cut 3 lines, it should end up as a do nothing action, nothing gets done.
            action_space.set_lines_status_switch_from_id(action=action, line_id=1, new_switch_value=1)
            action_space.set_lines_status_switch_from_id(action=action, line_id=0, new_switch_value=1)
            action_space.set_lines_status_switch_from_id(action=action, line_id=5, new_switch_value=1)

        if self.current_step == 2:
            # check if all lines are ON
            assert (np.equal(lines_status, np.ones(len(lines_status))).all())

            # cut line 0 and line 1
            action_space.set_lines_status_switch_from_id(action=action, line_id=0, new_switch_value=1)
            action_space.set_lines_status_switch_from_id(action=action, line_id=1, new_switch_value=1)

        if self.current_step == 3:
            # check if lines 0, 1 are OFF and the rest is ON
            assert (lines_status[0] == 0)
            assert (lines_status[1] == 0)
            for i in range(2, 15):
                assert (lines_status[i] == 1)

            # cut 4 lines, it should end up as a do nothing action, nothing gets done.
            action_space.set_lines_status_switch_from_id(action=action, line_id=2, new_switch_value=1)
            action_space.set_lines_status_switch_from_id(action=action, line_id=3, new_switch_value=1)
            action_space.set_lines_status_switch_from_id(action=action, line_id=4, new_switch_value=1)
            action_space.set_lines_status_switch_from_id(action=action, line_id=5, new_switch_value=1)

        if self.current_step == 4:
            # check AGAIN if lines 0, 1 are OFF and the rest is ON
            assert (lines_status[0] == 0)
            assert (lines_status[1] == 0)
            for i in range(2, 15):
                assert (lines_status[i] == 1)

        print("Step [{}], we do nothing : {}".format(self.current_step, np.equal(action.as_array(),
                                                                                 np.zeros(len(action))).all()))
        print("========================================================")

        self.current_step += 1
        return action


class Agent_test_LineTimeLimitSwitching(Agent):
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

    def __init__(self, environment, line_to_cut):
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

        print("We do nothing : ", np.equal(action.as_array(), np.zeros(len(action))).all())
        print("========================================================")

        return action


class Agent_test_NodeTimeLimitSwitching(Agent):
    """This agent tests the restriction : n_timesteps_actionned_node_reactionable: 3
        t == 1, [0, 0, 0] change one element. SHOULD WORK.     (restriction_step = 1)
        t == 2, [1, 0, 0] try to change again should NOT work  (restriction_step = 2)
        t == 3, [1, 0, 0] try to change again should NOT work  (restriction_step = 3)
        t == 4, [1, 0, 0] try to change node. SHOULD WORK.
        t == 5, [1, 1, 0] verify change.
    """

    def __init__(self, environment, node_to_change):
        super().__init__(environment)
        print("TestAgent_NodeLimitSwitching created...")

        self.current_step = 1
        self.node_to_change = node_to_change

    def act(self, observation):
        print("----------------------------------- current step = {} -----------------------------------".format(
            self.current_step))
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
        nodes_ids = get_verbose_node_topology(observation, action_space)
        print("nodes_ids = ", nodes_ids)

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

        if self.current_step == 1:
            # CHANGE TOPOLOGY OF NODE_TO_CHANGE
            print("we change the topology of node {}, curr_step = {}".format(self.node_to_change, self.current_step))

            # we connect the PRODUCTION to BUSBAR 1
            target_configuration[0] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)
            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, self.node_to_change)
            assert np.all(current_configuration == target_configuration)

        elif self.current_step == 2:
            new_supposed_name_str = "666" + str(self.node_to_change)
            assert(int(new_supposed_name_str) in nodes_ids)
            # Here we just OBSERVE that the NODE_TO_CHANGE has the production now connected to BUSBAR1
            index_res = list(observation.productions_substations_ids).index(self.node_to_change)
            assert (observation.productions_nodes[index_res] == 1)

            #  t == 2, try to change again should NOT work
            # we connect the second element to BUSBAR 1
            target_configuration[1] = 1

            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)

        elif self.current_step == 3:
            # we check that the change at t==2 did not occur, and that we still have current_conf = [1, 0, ... , 0, 0, ]
            for i, elem in enumerate(current_conf):
                if i == 0:
                    assert(elem == 1)
                else:
                    assert(elem == 0)
            #  t == 3, try to change again should NOT work
            # we connect the second element to BUSBAR 1
            target_configuration[1] = 1
            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)

        elif self.current_step == 4:
            # we check that the change at t==3 did not occur, and that we still have current_conf = [1, 0, ... , 0, 0, ]
            for i, elem in enumerate(current_conf):
                if i == 0:
                    assert(elem == 1)
                else:
                    assert(elem == 0)
            #  t == 4, try to change again NOW SHOULD WORK
            # we connect the second element to BUSBAR 1
            target_configuration[1] = 1
            action_space.set_substation_switches_in_action(action=action, substation_id=self.node_to_change,
                                                           new_values=target_configuration)

        elif self.current_step == 5:
            # we check that the change at t==3 did occur, and that we still have current_conf = [1, 1, ... , 0, 0, ]
            for i, elem in enumerate(current_conf):
                if i == 0 or i == 1:
                    assert(elem == 1)
                else:
                    assert(elem == 0)

        self.current_step += 1
        print("the action we return is, action = ", action)

        return action


def get_verbose_node_topology(obs, action_space):
    """This function returns the <real> topology, ie, split nodes are displayed"""
    n_bars = len(action_space.substations_ids)
    # The following code allows to get just the nodes ids
    # where there are elements connected. It also considerer
    # the split node action.
    all_sub_conf = []
    for sub_id in obs.substations_ids:
        sub_conf, _ = obs.get_nodes_of_substation(sub_id)
        all_sub_conf.append(sub_conf)

    nodes_ids = np.arange(1, n_bars + 1)
    for i in range(len(all_sub_conf)):
        # Check if all elements in sub (i) are connected to busbar B1.
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

    for node in obs.substations_ids:
        conf = obs.get_nodes_of_substation(node)
        # print(f"node [{node}] config = {conf}")
        # print(f"func: get_verbose_topo: node [{node}]")
        ii = 0
        for elem, type in zip(conf[0], conf[1]):
            # print(f"element n°[{ii}] connected to BusBar n°[{elem}] is a [{type}]")
            ii += 1
    return list(nodes_ids)


def test_first():
    a = 1
    b = 1
    assert (a == b)


def test_second():
    a = "A"
    b = "A"
    assert (a == b)


def test_differencial_topology():
    res = get_differencial_topology([0, 0, 0], [1, 1, 1])
    assert(res == [1, 1, 1])

    res = get_differencial_topology([0, 1, 0], [1, 1, 1])
    assert(res == [1, 0, 1])

    res = get_differencial_topology([0, 0, 1], [1, 1, 0])
    assert(res == [1, 1, 1])


def test_Agent_test_LineTimeLimitSwitching():
    """
    This function creates an agent that switches a line, then tries to switch it again. (But should be nullified because
     of input param "n_timesteps_actionned_line_reactionable: 3", then after 3 steps, we switch it back up.
    """
    parameters = "./tests/parameters/default14_for_tests/"
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
        agent = Agent_test_LineTimeLimitSwitching(env, line_to_cut)
        # Instantiate game runner and loop
        runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward = runner.loop(iterations=niter)
        print("Obtained a final reward of {}".format(final_reward))


def test_Agent_test_NodeTimeLimitSwitching():
    """This function creates an agent that switches a node topology, then tries to switch it again (same).
    (But should be nullified because of input param "n_timesteps_actionned_node_reactionable: 3", then after 3 steps,
     we switch it back up.
    """
    parameters = "./tests/parameters/default14_for_tests/"
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # agent = "RandomLineSwitch"
    niter = 5
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
        agent = Agent_test_NodeTimeLimitSwitching(env, node_to_change)
        # Instantiate game runner and loop
        runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
        final_reward = runner.loop(iterations=niter)
        print("Obtained a final reward of {}".format(final_reward))


def test_Agent_test_MaxNumberActionnedLines():
    """This function creates an Agent that tests the restriction param: max_number_actionned_lines """

    parameters = "./tests/parameters/default14_for_tests/"
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # agent = "RandomLineSwitch"
    niter = 5

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_MaxNumberActionnedLines(env)
    # Instantiate game runner and loop
    runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))


def test_Agent_test_MaxNumberActionnedNodes():
    """This function creates an Agent that tests the restriction param: max_number_actionned_lines """

    parameters = "./tests/parameters/default14_for_tests/"
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # agent = "RandomLineSwitch"
    niter = 3

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_MaxNumberActionnedSubstations(env)
    # Instantiate game runner and loop
    runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))


def test_Agent_test_BasicSubstationTopologyChange():
    """This function creates an Agent that tests all the Topological changes of all the Substations"""
    parameters = "./tests/parameters/default14_for_tests_alpha/"
    print("Parameters used = ", parameters)
    game_level = "level0"
    loop_mode = "natural"
    start_id = 0
    game_over_mode = "soft"
    renderer_latency = 1
    render = False
    # agent = "RandomLineSwitch"
    niter = 6

    env_class = RunEnv

    # Instantiate environment and agent
    env = env_class(parameters_folder=parameters, game_level=game_level,
                    chronic_looping_mode=loop_mode, start_id=start_id,
                    game_over_mode=game_over_mode, renderer_latency=renderer_latency)
    agent = Agent_test_BasicSubstationTopologyChange(env, 1)
    # Instantiate game runner and loop
    runner = Runner(env, agent, render, False, False, parameters, game_level, niter)
    final_reward = runner.loop(iterations=niter)
    print("Obtained a final reward of {}".format(final_reward))


def create_node_topology_generator(l):
    for i in range(len(l)):
        res = l.copy()
        res[i] = 1
        yield list(res.astype(int))


def get_differencial_topology(new_conf, old_conf):
    """new - old, for elem in result, if elem -1, then put one"""
    assert(len(new_conf) == len(old_conf))
    res = []

    for elemNew, elemOld in zip(new_conf, old_conf):
        r = elemNew - elemOld
        if r < 0:
            res.append(1)
        else:
            res.append(int(r))
    return res




test_Agent_test_NodeTimeLimitSwitching()
# test_Agent_test_MaxNumberActionnedLines()
# test_Agent_test_MaxNumberActionnedNodes()
# test_Agent_test_BasicSubstationTopologyChange()


