__author__ = 'marvinler'
from pypownet.reward_signal import RewardSignal


class CustomRewardSignal(RewardSignal):
    def __init__(self):
        super().__init__()

    def compute_reward(self, observation, action, flag):
        return [-1] if flag is not None else [1]
