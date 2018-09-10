__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import pypownet.environment


class Agent(object):
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    def __init__(self, environment):
        """Initialize a new agent."""
        assert isinstance(environment, pypownet.environment.RunEnv)
        self.environment = environment
        # Do not forget to call super() for the daughter classes

    def act(self, observation):
        """Produces an action given an observation of the environment.

        Takes as argument an observation of the current state, and returns the chosen action."""
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        # Implement your policy here.
        action = self.environment.action_space.get_do_nothing_action()

        return action

    def feed_reward(self, observation, action, reward):
        """ Here, the Agent upgrades his policy given the previous observation, its subsequently produced action, and
        the associated reward. This is where the Agent learns. """
        pass


# Examples of baselines agents
import numpy as np


class RandomLineSwitch(Agent):
    """
    An example of a baseline controler that randomly switches the status of one random power line per timestep (if the
    random line is previously online, switch it off, otherwise switch it on).
    """
    def __init__(self, environment, destination_path='saved_actions_RandomLineSwitch.csv'):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path=destination_path)

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        action = action_space.get_do_nothing_action()
        action.get_lines_status_subaction()[np.random.randint(action_space.lines_status_subaction_length)] = 1

        # Dump best action into stored actions file
        self.ioman.dump(action)

        return action

        # No learning (i.e. self.feed_reward does pass)


class RandomNodeSplitting(Agent):
    def __init__(self, environment, destination_path='saved_actions_RandomNodeSplitting.csv'):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path=destination_path)

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        action = action_space.get_do_nothing_action()
        action.get_topological_subaction()[np.random.randint(action_space.topological_subaction_length)] = 1

        # Dump best action into stored actions file
        self.ioman.dump(action)

        return action

        # No learning (i.e. self.feed_reward does pass)


class TreeSearchLineServiceStatus(Agent):
    """
    Exhaustive tree search of depth 1 limited to no action + 1 line switch activation
    """
    def __init__(self, environment, destination_path='saved_actions_TreeSearchLineServiceStatus.csv', verbose=True):
        super().__init__(environment)
        self.verbose = verbose

        self.ioman = ActIOnManager(destination_path=destination_path)

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        n_lines = self.environment.action_space.lines_status_subaction_length
        # Simulate the line status switch of every line, independently, and save rewards for each simulation (also store
        # the actions for best-picking strat)
        simulated_rewards = []
        simulated_actions = []
        for l in range(n_lines):
            if self.verbose:
                print('    Simulating switch activation line %d' % l, end='')
            action = self.environment.action_space.get_do_nothing_action()
            action.lines_status_subaction[l] = 1
            simulated_reward = self.environment.simulate(action=action)

            # Store ROI values
            simulated_rewards.append(simulated_reward)
            simulated_actions.append(action)
            if self.verbose:
                print('; expected reward %.5f' % simulated_reward)

        # Also simulate the do nothing action
        donothing_action = self.environment.action_space.get_do_nothing_action()
        donothing_simulated_reward = self.environment.simulate(action=donothing_action)
        simulated_rewards.append(donothing_simulated_reward)

        # Seek for the action that maximizes the reward
        best_simulated_reward = np.max(simulated_rewards)
        best_action = simulated_actions[simulated_rewards.index(best_simulated_reward)]

        # Dump best action into stored actions file
        self.ioman.dump(best_action)

        if self.verbose:
            print('  Best simulated action: disconnect line %d; expected reward: %.5f' % (
                simulated_rewards.index(best_simulated_reward), best_simulated_reward))

        return best_action


class GreedySearch(Agent):
    """ Agent that tries every possible action and retrieves the one that gives the best reward.
    """
    def __init__(self, environment, verbose=True, destination_path='saved_actions.csv'):
        super().__init__(environment)
        self.verbose = verbose

        self.ioman = ActIOnManager(destination_path=destination_path)

    def act(self, observation):
        import itertools

        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space
        number_lines = action_space.n_lines
        length_action = action_space.n

        # Will store reward, actions, and action name, then eventually pick the maximum reward and retrieve the
        # associated values
        rewards, actions, names = [], [], []

        # Test doing nothing
        if self.verbose:
            print(' Simulation with no action', end='')
        action = action_space.get_do_nothing_action()
        reward_aslist = self.environment.simulate(action, do_sum=False)
        reward = sum(reward_aslist)
        if self.verbose:
            print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
        rewards.append(reward)
        actions.append(action)
        names.append('no action')

        # Test every line opening
        for l in range(number_lines):
            if self.verbose:
                print(' Simulation with switching status of line %d' % l, end='')
            action = action_space.get_do_nothing_action()
            action.lines_status_subaction[l] = 1
            reward_aslist = self.environment.simulate(action, do_sum=False)
            reward = sum(reward_aslist)
            if self.verbose:
                print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
            rewards.append(reward)
            actions.append(action)
            names.append('switching status of line %d' % l)

        # Then test every node configuration for the considered nodes
        # We are only interested in nodes that are connected to at least from 4 to 6 elements
        # Those elements can be transmission lines or injections
        #n_elements_nodes = [5, 8, 4, 7, 3, 8, 4, 1, 5, 3, 1, 3, 3, 1]
        n_elements_nodes = [3, 6, 4, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3]
        #element_per_node = {1: 5, 2: 8, 3: 4, 4: 7, 5: 3, 6: 8, 7: 4, 8: 1, 9: 5, 10: 3, 11: 1, 12: 3, 13: 3, 14: 1}
        action_offset = 0
        for node, n_elements in enumerate(n_elements_nodes):
            if n_elements in [4, 5]:
                # Loopking trhough all configurations of n_elements binary vector with first value fixed to 0
                for configuration in list(itertools.product([0, 1], repeat=n_elements - 1)):
                    conf = [0]+list(configuration)
                    if self.verbose:
                        print(' Simulation with change in topo of node %d with switches %s' % (node, repr(conf)), end='')
                    # Construct action
                    action = action_space.get_do_nothing_action()
                    action_space.topological_subaction_length[action_offset:action_offset + n_elements] = conf
                    reward_aslist = self.environment.simulate(action, do_sum=False)
                    reward = sum(reward_aslist)
                    if self.verbose:
                        print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
                    rewards.append(reward)
                    actions.append(action)
                    names.append('change in topo of node %d with switches %s' % (node, repr(conf)))
            # Add offset for next action position
            action_offset += n_elements

        # Take the best reward, and retrieve the corresponding action
        best_reward = max(rewards)
        best_index = rewards.index(best_reward)
        best_action = actions[best_index]
        best_action_name = names[best_index]

        # Dump best action into stored actions file
        self.ioman.dump(best_action)

        if self.verbose:
            print('Action chosen: ', best_action_name, '; expected reward %.4f' % best_reward)
        if self.verbose:
            print(best_action)

        return best_action


class ActionsFileReaderControler(Agent):
    def __init__(self, environment, storedactions_file='saved_actions.csv'):
        super().__init__(environment)

        # Loads manager + actions
        ioman = ActIOnManager(delete=False)
        self.actions = ioman.load(storedactions_file)
        self.action_ctr = 0

        self.number_actions = len(self.actions)
        number_do_nothing = np.sum([np.sum(action) == 0 for action in self.actions])
        print('% of do-nothing:', float(number_do_nothing) / float(self.number_actions))

    def act(self, observation):
        action = self.actions[self.action_ctr]  # Correspondance first action to be played = first of list
        self.action_ctr += 1
        return action

import os


class ActIOnManager(object):
    def __init__(self, destination_path='saved_actions.csv', delete=True):
        self.actions = []
        self.destination_path = destination_path
        print('Storing actions at', destination_path)

        # Delete last path with same name by default!!!
        if delete and os.path.exists(destination_path):
            os.remove(destination_path)

    def dump(self, action):
        with open(self.destination_path, 'a') as f:
            f.write(','.join([str(int(switch)) for switch in action.as_array()])+'\n')

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        actions = [[int(l) for l in line.split(',')] for line in lines]
        assert 0 in np.unique(actions) and 1 in np.unique(actions) and len(np.unique(actions)) == 2
        return actions
