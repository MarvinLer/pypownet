__author__ = 'marvinler'
from pypownet.reward_signal import RewardSignal


class CustomRewardSignal(RewardSignal):
    def __init__(self):
        super().__init__()

        self.successful_step_reward = 1.
        self.unsuccessful_step_reward = -1.

    def compute_reward(self, observation, action, flag):
        return [self.unsuccessful_step_reward] if flag is not None else [self.successful_step_reward]

