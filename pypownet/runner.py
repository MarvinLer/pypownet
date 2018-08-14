__author__ = 'marvinler'
"""
This is the machinnery that runs your agent in an environment. Note that this is not the machinnery of the update of the
environment; it is purely related to perform policy inference at each timestep given the last observation, and feeding
the reward signal to the appropriate function (feed_reward) of the Agent.

This is not intented to be modified during the practical.
"""
from pypownet.env import RunEnv
from pypownet.agent import Agent


class Runner(object):
    def __init__(self, environment, agent, verbose=False, render=False):
        # Sanity checks: both environment and agent should inherit resp. RunEnv and Agent
        assert isinstance(environment, RunEnv)
        assert isinstance(agent, Agent)

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
        observation, reward, done, info = self.environment.step(action)

        if self.render:
            self.environment.render()

        # Feed the reward signal to the Agent along with last observation and its resulting action
        self.agent.feed_reward(self.last_observation, action, reward)

        if done:
            if self.verbose:
                print(info)  # Print the reason raised by the environment for the game over
                print("Game over! Restarting game")
            observation = self.environment.reset(restart=False)

        self.last_observation = observation

        return observation, action, reward

    def loop(self, iterations):
        cumul_reward = 0.0
        for i in range(1, iterations + 1):
            (obs, act, rew) = self.step()
            cumul_reward += rew
            if self.verbose:
                print("Simulation step {}:".format(i))
                #print(" ->       observation: {}".format(obs))
                #print(" ->            action: {}".format(act))
                print(" ->            reward: {}".format(rew))
                print(" -> cumulative reward: {}".format(cumul_reward))
        return cumul_reward


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
        self.runners = [Runner(env, agent, verbose=False, render=False) for (env, agent) in zip(environments, agents)]

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
