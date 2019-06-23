import numpy as np
from pypownet.runner import Runner


class WrappedRunner(Runner):
    def __init__(self, environment, agent, render=False, verbose=False, vverbose=False, parameters=None, level=None,
                 max_iter=None, log_filepath='runner.log', machinelog_filepath='machine_logs.csv'):
        super().__init__(environment, agent, render, verbose, vverbose, parameters, level, max_iter, log_filepath,
                         machinelog_filepath)

    def step(self, observation):
        """
        Performs a full RL step: the agent acts given an observation, receives and process the reward, and the env is
        resetted if done was returned as True; this also logs the variables of the system including actions,
        observations.
        :param observation: input observation to be given to the agent
        :return: (new observation, action taken, reward received)
        """
        action = self.agent.act(observation)

        # Update the environment with the chosen action
        observation, reward_aslist, done, info = self.environment.step(action, do_sum=False)
        if done:
            observation = self.environment.reset()
        elif info:
            self.logger.warning(info.text)

        reward = sum(reward_aslist)

        if self.render:
            self.environment.render()

        self.agent.feed_reward(action, observation, reward_aslist)

        return observation, action, reward, reward_aslist, done, info

    def loop(self, iterations, episodes=1):
        """
        Runs the simulator for the given number of iterations time the number of episodes.
        :param iterations: int of number of iterations per episode
        :param episodes: int of number of episodes, each resetting the environment at the beginning
        :return:
        """
        cumul_rew = 0.0
        dones = []
        infos = []
        for i_episode in range(episodes):
            observation = self.environment.reset()
            for i in range(1, iterations + 1):
                (observation, action, reward, reward_aslist, done, info) = self.step(observation)
                cumul_rew += reward
                dones.append(done)
                infos.append(info)

        return cumul_rew, dones, infos


def get_verbose_node_topology(obs, action_space):
    """This function returns the <real> topology, ie, split nodes are displayed"""
    n_bars = len(action_space.substations_ids)
    # The following code allows to get just the nodes ids
    # where there are elements connected. It also considerer
    # the split node action.
    all_sub_conf = []
    for sub_id in obs.substations_ids:
        sub_conf, _ = obs.get_nodes_of_substation(sub_id)
        all_sub_conf.append(sub_conf)

    nodes_ids = np.arange(1, n_bars + 1)
    for i in range(len(all_sub_conf)):
        # Check if all elements in sub (i) are connected to busbar B1.
        # print(np.equal(all_sub_conf[i], np.ones(len(all_sub_conf[i]))))
        # print("np.ones(len(all_sub_conf[i] = ", np.ones(len(all_sub_conf[i])))
        # print("type = ", type(np.equal(all_sub_conf[i], np.ones(len(all_sub_conf[i])))))
        if (np.equal(all_sub_conf[i], np.ones(len(all_sub_conf[i])))).all():
            # Remove the existing node.
            nodes_ids = np.delete(nodes_ids, i)
            # And create a new node.
            nodes_ids = np.append(nodes_ids, int(str(666) + str(i + 1)))
        # Check if one or more elements
        # are connected to busbar B1.
        elif np.sum(all_sub_conf[i]) > 0:
            nodes_ids = np.append(nodes_ids, int(str(666) + str(i + 1)))

    nodes_ids = list(nodes_ids)

    for node in obs.substations_ids:
        conf = obs.get_nodes_of_substation(node)
        # print(f"node [{node}] config = {conf}")
        # print(f"func: get_verbose_topo: node [{node}]")
        ii = 0
        for elem, type in zip(conf[0], conf[1]):
            # print(f"element n°[{ii}] connected to BusBar n°[{elem}] is a [{type}]")
            ii += 1
    return list(nodes_ids)