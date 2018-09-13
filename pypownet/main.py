__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import argparse
from pypownet.environment import RunEnv
from pypownet.runner import Runner, BatchRunner
import pypownet.agent

parser = argparse.ArgumentParser(description='CLI tool to run experiments using PyPowNet.')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='class to use for the agent (must be within the \'pypownet/agent.py\' file); '
                         'default class Agent')
parser.add_argument('--parameters', '-p', metavar='params', default='./parameters/default14/', type=str,
                    help='parent folder containing the parameters of the simulator to be used (folder should contain '
                         'configuration.json and reference_grid.m)')
parser.add_argument('--niter', type=int, metavar='n', default='100',
                    help='number of iterations to simulate; default 100')
# parser.add_argument('--batch', type=int, metavar='n_agent', default=None,
#                     help='number of game instances to run simultaneously; default 1')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='display live info of the current experiment including reward, cumulative reward')
parser.add_argument('-vv', '--vverbose', action='store_true',
                    help='display live info + observations and actions played')
parser.add_argument('-r', '--render', action='store_true',
                    help='render the power network observation at each timestep (not available if --batch is not 1)')
parser.add_argument('--start', type=int, default=None,
                    help='id of the timestep to start the game at (>= 0)')
parser.add_argument('-l', '--latency', type=float, default=None,
                    help='time to sleep after each frame plot of the renderer (in seconds); note: there are multiple'
                         ' frame plots per timestep (at least 2, varies)')


def main():
    args = parser.parse_args()
    env_class = RunEnv
    agent_class = eval('pypownet.agent.{}'.format(args.agent))

    start_id = args.start

    # Instantiate environment and agent
    env = env_class(parameters_folder=args.parameters, start_id=start_id, latency=args.latency)
    agent = agent_class(env)
    # Instantiate game runner and loop
    runner = Runner(env, agent, args.render, args.verbose, args.vverbose)
    final_reward = runner.loop(iterations=args.niter)
    print("Obtained a final reward of {}".format(final_reward))


if __name__ == "__main__":
    main()