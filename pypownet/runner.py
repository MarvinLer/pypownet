__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
""" This is the machinnery that runs your agent in an environment. Note that this is not the machinnery of the update of the
environment; it is purely related to perform policy inference at each timestep given the last observation, and feeding
the reward signal to the appropriate function (feed_reward) of the Agent.

This is not intented to be modified during the practical.
"""
from pypownet.environment import RunEnv
from pypownet.agent import Agent
import logging
import logging.handlers
import threading
import queue

LOG_FILENAME = 'runner.log'


class TimestepTimeout(Exception):
    pass


class Runner(object):
    def __init__(self, environment, agent, render=False, verbose=False, vverbose=False, log_filepath='runner.log'):
        # Sanity checks: both environment and agent should inherit resp. RunEnv and Agent
        assert isinstance(environment, RunEnv)
        assert isinstance(agent, Agent)

        # Loggger part
        self.logger = logging.getLogger('pypownet')

        # Always create a log file for runners
        fh = logging.FileHandler(filename=log_filepath, mode='w+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        if verbose or vverbose:
            # create console handler, set level to debug, create formatter
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if vverbose and verbose else logging.INFO)
            ch.setFormatter(logging.Formatter('%(levelname)s        %(message)s'))
            self.ch = ch
            # add ch to logger
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.DEBUG if vverbose else logging.INFO)

        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.render = render

        # First observation given by the environment
        self.last_observation = self.environment._get_obs()

        self.max_seconds_per_timestep = self.environment.game.get_max_seconds_per_timestep()

        if self.render:
            self.environment.render()

    def step(self):
        # Policy inference
        # def agent_inference(obs, q):
        #     action = self.agent.act(obs)
        #     q.put(action)
        # q = queue.Queue()
        # t = threading.Thread(target=agent_inference, name='AgentActThread', args=(self.last_observation, q))
        # t.start()
        # t.join(self.max_seconds_per_timestep)
        # if t.is_alive():
        #     self.logger.warn('\b\b\bTook too much time to compute action for current timestep: allowed at most %s '
        #                      'seconds; emulating do-nothing action' % str(self.max_seconds_per_timestep))
        #     action = self.environment.action_space.get_do_nothing_action()
        # else:
        #     action = q.get()

        action = self.agent.act(self.last_observation)

        # Update the environment with the chosen action
        observation, reward_aslist, done, info = self.environment.step(action, do_sum=False)
        if done:
            self.logger.warn('\b\b\bGAME OVER! Resetting grid... (hint: %s)' % info.text)
            observation = self.environment.reset()
        elif info:
            self.logger.warn(info.text)
        # self.logger.error(observation.as_array().shape)
        # self.logger.error(observation.as_ac_minimalist().as_array().shape)
        # self.logger.error(observation.as_minimalist().as_array().shape)
        # self.logger.error('\n'.join(observation.as_dict().keys()))
        # exit()

        reward = sum(reward_aslist)

        if self.render:
            self.environment.render()

        self.last_observation = observation

        self.agent.feed_reward(action, observation, reward_aslist)

        self.logger.debug('action: %s' % ('[%s]' % ' '.join(list(map(lambda x: str(int(x)), action.as_array())))))
        self.logger.debug('reward: {}'.format('[' + ','.join(list(map(str, reward_aslist))) + ']'))
        self.logger.debug('done: {}'.format(done))
        self.logger.debug('info: {}'.format(info if not info else info.text))
        self.logger.debug('observation: \n%s' % observation.__str__())

        return observation, action, reward

    def loop(self, iterations):
        cumul_rew = 0.0
        for i in range(1, iterations + 1):
            (obs, act, rew) = self.step()
            cumul_rew += rew
            self.logger.info("step %d/%d - reward: %.2f; cumulative reward: %.2f" % (i, iterations, rew, cumul_rew))

        # Close pygame if renderer has been used
        if self.render:
            self.environment.render()

        return cumul_rew
