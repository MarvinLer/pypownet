__author__ = 'marvinler'
from pypownet.env import RunEnv


class Agent(object):
    """
    The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    def __init__(self, environment):
        """Initialize a new agent."""
        self.environment = environment
        # Do not forget to call super() for the daughter classes

    def act(self, observation):
        """Produces an action given an observation of the environment.

        Takes as argument an observation of the current state, and returns the chosen action."""
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, RunEnv.Observation)

        # Implement your policy here.
        return None

    def feed_reward(self, observation, action, reward):
        """ Here, the Agent upgrades his policy given the previous observation, its subsequently produced action, and
        the associated reward. This is where the Agent learns. """
        pass


# Examples of baselines agents
import numpy as np


class RandomSwitch(Agent):
    """
    An example of a baseline controler that randomly switches one element (either node-splitting or line service status
    switch).
    """

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, self.environment.Observation)
        action_space = self.environment.action_space
        length_action = action_space.n

        action = np.zeros((length_action,))
        action[np.random.randint(length_action)] = 1

        return action

        # No learning (i.e. self.feed_reward does pass)


class RandomLineSwitch(Agent):
    """
    An example of a baseline controler that randomly switches the status of one random power line per timestep (if the
    random line is previously online, switch it off, otherwise switch it on).
    """

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, self.environment.Observation)
        action_space = self.environment.action_space
        number_lines = action_space.n_lines
        length_action = action_space.n

        topological_switches_subaction = np.zeros((length_action - number_lines,))
        line_switches_subaction = np.zeros((number_lines,))
        line_switches_subaction[np.random.randint(number_lines)] = 1

        return np.concatenate((topological_switches_subaction, line_switches_subaction))

        # No learning (i.e. self.feed_reward does pass)


class RandomNodeSplitting(Agent):
    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, self.environment.Observation)
        action_space = self.environment.action_space
        number_lines = action_space.n_lines
        length_action = action_space.n

        line_switches_subaction = np.zeros((number_lines,))
        topological_switches_subaction = np.zeros((length_action - number_lines,))
        topological_switches_subaction[np.random.randint(length_action - number_lines)] = 1

        return np.concatenate((topological_switches_subaction, line_switches_subaction))

        # No learning (i.e. self.feed_reward does pass)


class TreeSearchLineServiceStatus(Agent):
    """
    Exhaustive tree search of depth 1 limited to no action + 1 line switch activation
    """
    def __init__(self, environment, verbose=True):
        super().__init__(environment)
        self.verbose = verbose

    def act(self, observation):
        length_action = self.environment.action_space.n
        n_lines = self.environment.action_space.n_lines

        topoligical_subaction = np.zeros((length_action - n_lines,))

        # Simulate the line status switch of every line, independently, and save rewards for each simulation (also store
        # the actions for best-picking strat)
        simulated_rewards = []
        simulated_actions = []
        for l in range(n_lines):
            if self.verbose:
                print('    Simulating switch activation line %d' % l, end='')
            linestatus_subaction = np.zeros((n_lines,))
            linestatus_subaction[l] = 1  # Activate switch of service status of line l
            action = np.concatenate((topoligical_subaction, linestatus_subaction))  # Build an action
            simulated_reward = self.environment.simulate(action=action)

            # Store ROI values
            simulated_rewards.append(simulated_reward)
            simulated_actions.append(action)
            if self.verbose:
                print('; expected reward %.5f' % simulated_reward)
        # Also simulate the do nothing action
        donothing_action = np.zeros(length_action)
        donothing_simulated_reward = self.environment.simulate(action=donothing_action)
        simulated_rewards.append(donothing_simulated_reward)

        # Seek for the action that maximizes the reward
        best_simulated_reward = np.max(simulated_rewards)
        best_action = simulated_actions[simulated_rewards.index(best_simulated_reward)]

        if self.verbose:
            print('  Best simulated action: disconnect line %d; expected reward: %.5f' % (
                simulated_rewards.index(best_simulated_reward), best_simulated_reward))

        return best_action
