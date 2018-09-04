__author__ = 'marvinler'
"""
This is the machinnery that runs your agent in an environment. Note that this is not the machinnery of the update of the
environment; it is purely related to perform policy inference at each timestep given the last observation, and feeding
the reward signal to the appropriate function (feed_reward) of the Agent.

This is not intented to be modified during the practical.
"""
from pypownet.environment import RunEnv
from pypownet.agent import Agent
import logging
import logging.handlers

LOG_FILENAME = 'runner.log'


class Runner(object):
    def __init__(self, environment, agent, render=False, verbose=False, vverbose=False, log_filepath='runner.log'):
        # Sanity checks: both environment and agent should inherit resp. RunEnv and Agent
        assert isinstance(environment, RunEnv)
        assert isinstance(agent, Agent)

        # Loggger part
        logging.basicConfig(filename=log_filepath, level=logging.WARNING)
        self.logger = logging.getLogger(__file__)

        # Always create a log file for runners
        sh = logging.FileHandler(filename='runner.log', mode='w')
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(sh)

        if verbose or vverbose:
            # create console handler, set level to debug, create formatter
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO if not vverbose else logging.DEBUG)
            ch.setFormatter(logging.Formatter('%(levelname)s     %(message)s'))
            # add ch to logger
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO if not vverbose else logging.DEBUG)
            print(self.logger.getEffectiveLevel())

        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.render = render

        # First observation given by the environment
        self.last_observation = self.environment._get_obs()

    def step(self):
        # Policy inference
        action = self.agent.act(self.last_observation)

        # Update the environment with the chosen action
        observation, reward_aslist, done, info = self.environment.step(action, do_sum=False)
        reward = sum(reward_aslist)

        if self.render:
            self.environment.render(game_over=done)  # Propagate game over signal to plot 'Game over' on screen

        # Feed the reward signal to the Agent along with last observation and its resulting action
        self.agent.feed_reward(self.last_observation, action, reward)

        if done:
            self.logger.info('Game over! hint: %s. Resetting grid...' % info.text)
            observation = self.environment.reset(restart=False)

        self.last_observation = observation

        self.logger.debug('observation: \n%s', observation.__str__())
        self.logger.debug('reward: {}'.format('['+','.join(list(map(str, reward_aslist)))+']'))
        self.logger.debug('done: {}'.format(done))
        self.logger.debug('info: {}'.format(info if not info else info.text))

        return observation, action, reward

    def loop(self, iterations):
        cumul_rew = 0.0
        for i in range(1, iterations + 1):
            (obs, act, rew) = self.step()  # Close if last iteration
            cumul_rew += rew
            self.logger.info("Step %d/%d - reward: %.2f; cumulative reward: %.2f" % (i, iterations, rew, cumul_rew))

        # Close pygame if renderer has been used
        if self.render:
            self.environment.render(close=True)

        return cumul_rew


def iter_or_loopcall(o, count):
    if callable(o):
        return [o() for _ in range(count)]
    else:
        # must be iterable
        return list(iter(o))


class BatchRunner(object):
    """ Runs several instances of the game simultaneously and aggregates the results. """
    def __init__(self, env_maker, agent_maker, count, verbose=False, render=False):
        environments = iter_or_loopcall(env_maker, count)
        agents = iter_or_loopcall(agent_maker, count)
        self.runners = [Runner(env, agent, render=False, verbose=False) for (env, agent) in zip(environments, agents)]

        self.verbose = verbose
        self.render = render

    def step(self):
        batch_reward = 0.
        for runner in self.runners:
            _, _, rew = runner.step()
            batch_reward += rew

        return batch_reward / len(self.runners)

    def loop(self, iterations):
        cum_avg_reward = 0.0
        for i in range(iterations):
            avg_reward = self.step()
            cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation step {}:".format(i+1))
                print(" ->       average step reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward


if __name__ == '__main__':
    # Here is a code-snippet for running one single experiment
    # Instantiate an environment, an agent and an associated experiment runner
    single_environment = RunEnv(grid_case=118)
    agent = Agent()  # The basic Agent class is equivalent to a do-nothing policy
    experiment_runner = Runner(single_environment, agent, verbose=True)
    # Run the Agent on the environment for 100 iterations
    iterations = 100
    experiment_runner.loop(iterations)

    # Here is a code-snippet for running 4 experiments simultaneously
    experiments_batch_runner = BatchRunner(env_maker=RunEnv,
                                           agent_maker=Agent,
                                           count=4,
                                           verbose=True)
    # Run the Agent on the environment for 100 iterations for each environment
    iterations = 100
    experiments_batch_runner.loop(iterations)
