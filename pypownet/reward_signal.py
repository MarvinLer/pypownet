__author__ = 'marvinler'


class RewardSignal(object):
    """ This is the basic template for the reward signal class that should at least implement a compute_reward
    method with an observation of pypownet.environment.Observation, an action of pypownet.environment.Action and a flag
    which is an exception of the environment package.
    """
    def __init__(self):
        pass

    def compute_reward(self, observation, action, flag):
        """ Effectively computes a reward given the current observation, the action taken (some actions are penalized)
        as well as the flag reward, which contains information regarding the latter game step, including the game
        over exceptions raised or illegal line reconnections.

        :param observation: an instance of pypownet.environment.Observation
        :param action: an instance of pypownet.game.Action
        :param flag: an exception either of pypownet.environment.DivergingLoadflowException,
        pypownet.environment.IllegalActionException, pypownet.environment.TooManyProductionsCut or
        pypownet.environment.TooManyConsumptionsCut
        :return: a list of subrewards as floats or int (potentially a list with only one value)
        """
        return [0.]
