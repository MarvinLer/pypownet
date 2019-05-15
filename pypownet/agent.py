__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import pypownet.environment
from abc import ABC, abstractmethod


class Agent(ABC):
    """ The template to be used to create an agent: any controller of the power grid is expected to be a daughter of this
    class.
    """

    def __init__(self, environment):
        """Initialize a new agent."""
        assert isinstance(environment, pypownet.environment.RunEnv)
        self.environment = environment

    @abstractmethod
    def act(self, observation):
        """Produces an action given an observation of the environment.

        Takes as argument an observation of the current state, and returns the chosen action of class Action or np
        array."""
        pass

    def feed_reward(self, action, consequent_observation, rewards_aslist):
        pass


class DoNothing(Agent):
    def act(self, observation):
        action_length = self.environment.action_space.action_length
        return np.zeros(action_length)


# Examples of baselines agents
import numpy as np


class RandomAction(Agent):
    """
    An example of a baseline controller that produce random actions (ie random line switches and random node switches.
    """

    def __init__(self, environment):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path='saved_actions_RandomLineSwitch.csv')

    def act(self, observation):
        action = self.environment.action_space.sample()
        # # or
        # action_length = self.environment.action_space.n
        # action = np.random.choice([0, 1], action_length)
        return action


class RandomPointAction(Agent):
    """
    An example of a baseline controller that produce 1 random activation (ie an array with all 0 but one 1).
    """

    def __init__(self, environment):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path='saved_actions_RandomLineSwitch.csv')

    def act(self, observation):
        action = self.environment.action_space.get_do_nothing_action()
        # # or
        # action_length = self.environment.action_space.n
        # action = np.zeros(action_length)
        action[np.random.randint(action.shape[0])] = 1
        return action


class RandomLineSwitch(Agent):
    """
    An example of a baseline controller that randomly switches the status of one random power line per timestep (if the
    random line is previously online, switch it off, otherwise switch it on).
    """

    def __init__(self, environment):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path='saved_actions_RandomLineSwitch.csv')

    def act(self, observation):
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

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)
        action_space.set_lines_status_switch_from_id(action=action,
                                                     line_id=np.random.randint(
                                                         action_space.lines_status_subaction_length),
                                                     new_switch_value=1)

        # Dump best action into stored actions file
        self.ioman.dump(action)

        return action

        # No learning (i.e. self.feed_reward does pass)


class RandomNodeSplitting(Agent):
    """ Implements a "random node-splitting" agent: at each timestep, this controller will select a random substation
    (id), then select a random switch configuration such that switched elements of the selected substations change the
    node within the substation on which they are directly wired.
    """

    def __init__(self, environment):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path='saved_actions_RandomNodeSplitting.csv')

    def act(self, observation):
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

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        # Select a random substation ID on which to perform node-splitting
        target_substation_id = np.random.choice(action_space.substations_ids)
        expected_target_configuration_size = action_space.get_number_elements_of_substation(target_substation_id)
        # Choses a new switch configuration (binary array)
        target_configuration = np.random.choice([0, 1], size=(expected_target_configuration_size,))

        action_space.set_substation_switches_in_action(action=action, substation_id=target_substation_id,
                                                       new_values=target_configuration)

        # Ensure changes have been done on action
        current_configuration, _ = action_space.get_substation_switches_in_action(action, target_substation_id)
        assert np.all(current_configuration == target_configuration)

        # Dump best action into stored actions file
        self.ioman.dump(action)

        return action


class TreeSearchLineServiceStatus(Agent):
    """ Exhaustive tree search of depth 1 limited to no action + 1 line switch activation
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.verbose = True

        self.ioman = ActIOnManager(destination_path='saved_actions_TreeSearchLineServiceStatus.csv')

    def act(self, observation):
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

        number_of_lines = self.environment.action_space.lines_status_subaction_length
        # Simulate the line status switch of every line, independently, and save rewards for each simulation (also store
        # the actions for best-picking strat)
        simulated_rewards = []
        simulated_actions = []
        for l in range(number_of_lines):
            if self.verbose:
                print('    Simulating switch activation line %d' % l, end='')
            # Construct the action where only line status of line l is switched
            action = action_space.get_do_nothing_action(as_class_Action=True)
            action_space.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)
            simulated_reward = self.environment.simulate(action=action)

            # Store ROI values
            simulated_rewards.append(simulated_reward)
            simulated_actions.append(action)
            if self.verbose:
                print('; expected reward %.5f' % simulated_reward)

        # Also simulate the do nothing action
        if self.verbose:
            print('    Simulating do-nothing action', end='')
        donothing_action = self.environment.action_space.get_do_nothing_action()
        donothing_simulated_reward = self.environment.simulate(action=donothing_action)
        simulated_rewards.append(donothing_simulated_reward)
        simulated_actions.append(donothing_action)

        # Seek for the action that maximizes the reward
        best_simulated_reward = np.max(simulated_rewards)
        best_action = simulated_actions[simulated_rewards.index(best_simulated_reward)]

        # Dump best action into stored actions file
        self.ioman.dump(best_action)

        if self.verbose:
            if simulated_rewards.index(best_simulated_reward) == len(simulated_rewards)-1:
                print('  Best simulated action: do-nothing')
            else:
                print('  Best simulated action: disconnect line %d; expected reward: %.5f' % (
                    simulated_rewards.index(best_simulated_reward), best_simulated_reward))

        return best_action


class GreedySearch(Agent):
    """ This agent is a tree-search model of depth 1, that is constrained to modifiying at most 1 substation
    configuration or at most 1 line status. This controller used the simulate method of the environment, by testing
    every 1-line status switch action, every new configuration for substations with at least 4 elements, as well as
    the do-nothing action. Then, it will seek for the best reward and return the associated action, expecting
    the maximum reward for the action pool it can reach.
    Note that the simulate method is only an approximation of the step method of the environment, and in three ways:
    * simulate uses the DC mode, while step is in AC
    * simulate uses only the predictions given to the player to simulate the next timestep injections
    * simulate can not compute the hazards that are supposed to come at the next timestep
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.verbose = True

        self.ioman = ActIOnManager(destination_path='saved_actions.csv')

    def act(self, observation):
        import itertools

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

        number_lines = action_space.lines_status_subaction_length
        # Will store reward, actions, and action name, then eventually pick the maximum reward and retrieve the
        # associated values
        rewards, actions, names = [], [], []

        # Test doing nothing
        if self.verbose:
            print(' Simulation with no action', end='')
        action = action_space.get_do_nothing_action()
        _, reward_aslist, _, _ = self.environment.simulate(action, do_sum=False)
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
            action = action_space.get_do_nothing_action(as_class_Action=True)
            action_space.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)
            _, reward_aslist, _, _ = self.environment.simulate(action, do_sum=False)
            reward = sum(reward_aslist)
            if self.verbose:
                print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
            rewards.append(reward)
            actions.append(action)
            names.append('switching status of line %d' % l)

        # For every substation with at least 4 elements, try every possible configuration for the switches
        for substation_id in action_space.substations_ids:
            substation_n_elements = action_space.get_number_elements_of_substation(substation_id)
            if 6 > substation_n_elements > 3:
                # Look through all configurations of n_elements binary vector with first value fixed to 0
                for configuration in list(itertools.product([0, 1], repeat=substation_n_elements - 1)):
                    new_configuration = [0] + list(configuration)
                    if self.verbose:
                        print(' Simulation with change in topo of sub. %d with switches %s' % (
                            substation_id, repr(new_configuration)), end='')
                    # Construct action
                    action = action_space.get_do_nothing_action(as_class_Action=True)
                    action_space.set_substation_switches_in_action(action=action, substation_id=substation_id,
                                                                   new_values=new_configuration)
                    _, reward_aslist, _, _ = self.environment.simulate(action, do_sum=False)
                    reward = sum(reward_aslist)
                    if self.verbose:
                        print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
                    rewards.append(reward)
                    actions.append(action)
                    names.append('change in topo of sub. %d with switches %s' % (substation_id,
                                                                                 repr(new_configuration)))

        # Take the best reward, and retrieve the corresponding action
        best_reward = max(rewards)
        best_index = rewards.index(best_reward)
        best_action = actions[best_index]
        best_action_name = names[best_index]

        # Dump best action into stored actions file
        self.ioman.dump(best_action)

        if self.verbose:
            print('Action chosen: ', best_action_name, '; expected reward %.4f' % best_reward)

        return best_action


class ActionsFileReaderControler(Agent):
    def __init__(self, environment):
        super().__init__(environment)

        # Loads manager + actions
        ioman = ActIOnManager(delete=False)
        self.actions = ioman.load('saved_actions.csv')
        self.action_ctr = 0

        self.number_actions = len(self.actions)
        number_do_nothing = np.sum([np.sum(action) == 0 for action in self.actions])
        print('% of do-nothing:', float(number_do_nothing) / float(self.number_actions))

    def act(self, observation):
        action = self.actions[self.action_ctr]  # Correspondance first action to be played = first of list
        self.action_ctr += 1
        return action

###############
# Helper agents
###############
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
            f.write('{}\n'.format(action))

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        actions = [[int(l) for l in line.split(',')] for line in lines]
        assert 0 in np.unique(actions) and 1 in np.unique(actions) and len(np.unique(actions)) == 2
        return actions


class FlowsSaver(Agent):
    def __init__(self, environment):
        """Initialize a new agent."""
        super().__init__(environment)
        assert isinstance(environment, pypownet.environment.RunEnv)
        self.environment = environment
        self.destination_path = 'saved_flows.csv'

    def act(self, observation):
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        open(self.destination_path, 'a').write(','.join(list(map(str, observation.ampere_flows))) + '\n')
        return self.environment.action_space.get_do_nothing_action()

