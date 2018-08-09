__author__ = 'marvinler'
from src.env import RunEnv


class Agent(object):
    def __init__(self):
        """Initialize a new agent."""

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
