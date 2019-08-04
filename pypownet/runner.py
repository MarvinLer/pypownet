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
import csv
import datetime

LOG_FILENAME = 'runner.log'


class TimestepTimeout(Exception):
    pass


class Runner(object):
    def __init__(self, environment, agent, render=False, verbose=False, vverbose=False, parameters=None, level=None,
                 max_iter=None, log_filepath='runner.log', machinelog_filepath='machine_logs.csv'):
        # Sanity checks: both environment and agent should inherit resp. RunEnv and Agent
        assert isinstance(environment, RunEnv)
        assert isinstance(agent, Agent)

        # Logger part
        self.logger = logging.getLogger('pypownet')
        if machinelog_filepath is not None:
            self.csv_writer = csv.writer(open(machinelog_filepath, 'w'), delimiter=';')
            self.csv_writer.writerow(['param_env_name', 'level', 'chronic_name', 'max_iter',
                                      'timestep', 'time', 'game_over', 'timestep_reward_aslist', 'timestep_reward',
                                      'cumulated_reward'])
            self.parameters = parameters
            self.level = level
            self.max_iter = max_iter
        else:
            self.csv_writer = None
            self.parameters, self.level, self.max_iter = None, None, None

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

        self.max_seconds_per_timestep = self.environment.game.get_max_seconds_per_timestep()

        if self.render:
            self.environment.render()

    def step(self, observation):
        """
        Performs a full RL step: the agent acts given an observation, receives and process the reward, and the env is
        resetted if done was returned as True; this also logs the variables of the system including actions,
        observations.
        :param observation: input observation to be given to the agent
        :return: (new observation, action taken, reward received)
        """
        self.logger.debug('observation: ' + str(self.environment.observation_space.array_to_observation(observation)))
        action = self.agent.act(observation)

        # Update the environment with the chosen action
        observation, reward_aslist, done, info = self.environment.step(action, do_sum=False)
        if done:
            self.logger.warning('\b\b\bGAME OVER! Resetting grid... (hint: %s)' % info.text)
            observation = self.environment.process_game_over()
        elif info:
            self.logger.warning(info.text)

        reward = sum(reward_aslist)

        if self.render:
            self.environment.render()

        self.agent.feed_reward(action, observation, reward_aslist)

        self.logger.debug('action: {}'.format(action))
        self.logger.debug('reward: {}'.format('[' + ','.join(list(map(str, reward_aslist))) + ']'))
        self.logger.debug('done: {}'.format(done))
        self.logger.debug('info: {}'.format(info if not info else info.text))

        return observation, action, reward, reward_aslist, done

    def loop(self, iterations, epochs=1):
        """
        Runs the simulator for the given number of iterations time the number of episodes.
        :param iterations: int of number of iterations per episode
        :param epochs: int of number of episodes, each resetting the environment at the beginning
        :return:
        """
        cumul_rew = 0.0
        for i_episode in range(epochs):
            # clean restart of environment at the beginning of each epoch
            self.logger.warning('Resetting environment...')
            observation = self.environment.reset()
            for i_iter in range(1, iterations + 1):
                (observation, action, reward, reward_aslist, done) = self.step(observation)
                cumul_rew += reward

                # save some info in txt and csv loggers
                self.logger.info("step %d/%d - reward: %.2f; cumulative reward: %.2f" %
                                 (i_iter, iterations, reward, cumul_rew))
                self.dump_machinelogs(i_iter, done, reward, reward_aslist, cumul_rew,
                                      self.environment.get_current_datetime())

        return cumul_rew

    def dump_machinelogs(self, timestep_id, done, reward, reward_aslist, cumul_rew, datetime):
        if self.csv_writer is None:
            return

        param_env_name = self.parameters
        level = self.level
        chronic_name = self.environment.get_current_chronic_name()
        max_iter = self.max_iter
        timestep = timestep_id
        time = datetime.strftime("%Y-%m-%d %H:%M")
        game_over = done
        timestep_reward_aslist = reward_aslist
        timestep_reward = reward
        cumulated_reward = cumul_rew

        self.csv_writer.writerow([param_env_name, level, chronic_name, max_iter, timestep, time, game_over,
                                  timestep_reward_aslist, timestep_reward, cumulated_reward])
