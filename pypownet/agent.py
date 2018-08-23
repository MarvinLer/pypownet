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