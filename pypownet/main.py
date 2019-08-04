__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
import argparse
from pypownet.environment import RunEnv
from pypownet.runner import Runner
import pypownet.agent

parser = argparse.ArgumentParser(description='CLI tool to run experiments using PyPowNet.')
parser.add_argument('-a', '--agent', metavar='AGENT_CLASS', default='DoNothing', type=str,
                    help='class to use for the agent (must be within the \'pypownet/agent.py\' file); '
                         'default class Agent')
parser.add_argument('-e', '--epochs', type=int, metavar='NUMBER_EPOCHS', default=10,
                    help='number of epochs to simulate (default 10); each epoch completely restarts the environment')
parser.add_argument('-n', '--niter', type=int, metavar='NUMBER_ITERATIONS_PER_EPOCH', default=1000,
                    help='number of iterations per epoch to simulate (default 1000)')
parser.add_argument('-p', '--parameters', metavar='PARAMETERS_FOLDER', default='./parameters/default14/', type=str,
                    help='parent folder containing the parameters of the simulator to be used (folder should contain '
                         'configuration.json and reference_grid.m)')
parser.add_argument('-lv', '--level', metavar='GAME_LEVEL', type=str, default='level0',
                    help='game level of the timestep entries to be played (default \'level0\')')
parser.add_argument('-s', '--start-id', metavar='CHRONIC_START_ID', type=int, default=0,
                    help='id of the first chronic to be played (default 0)')
parser.add_argument('-lm', '--loop-mode', metavar='CHRONIC_LOOP_MODE', type=str, default='natural',
                    help='the way the game will loop through chronics of the specified game level: "natural" will'
                         ' play chronic in alphabetical order, "random" will load random chronics ids and "fixed"'
                         ' will always play the same chronic folder (default "natural")')
parser.add_argument('-m', '--game-over-mode', metavar='GAME_OVER_MODE', type=str, default='soft',
                    help='game over mode to be played among "easy", "soft", "hard". With "easy", overflowed lines do '
                         'not break and game over do not end scenarios; '
                         'with "soft" overflowed lines are destroyed but game over do not end the scenarios; '
                         'with "hard" game over end the chronic upon game over signals and start the next ones if any.')
parser.add_argument('-r', '--render', action='store_true',
                    help='render the power network observation at each timestep (not available if --batch is not 1)')
parser.add_argument('-la', '--latency', type=float, default=None,
                    help='time to sleep after each frame plot of the renderer (in seconds); note: there are multiple'
                         ' frame plots per timestep (at least 2, varies)')
parser.add_argument('--seed', type=int, metavar='SEED', default=None,
                    help='seed used to initiate a random state for "random" loop mode only once at the beginning of '
                         'the script')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='display live info of the current experiment including reward, cumulative reward')
parser.add_argument('-vv', '--vverbose', action='store_true',
                    help='display live info + observations and actions played')


def main():
    args = parser.parse_args()
    env_class = RunEnv
    agent_class = eval('pypownet.agent.{}'.format(args.agent))

    # Instantiate environment and agent
    if args.game_over_mode.lower() not in ['easy', 'soft', 'hard']:
        raise ValueError('Unknown value {} for argument --game-over-mode; choices {}'.format(args.game_over_mode,
                                                                                             ['easy', 'soft', 'hard']))
    game_over_mode = 'hard' if args.game_over_mode.lower() == 'hard' else 'soft'
    without_overflow_cutoff = args.game_over_mode.lower() == 'easy'
    env = env_class(parameters_folder=args.parameters, game_level=args.level,
                    chronic_looping_mode=args.loop_mode, start_id=args.start_id,
                    game_over_mode=game_over_mode, renderer_latency=args.latency,
                    without_overflow_cutoff=without_overflow_cutoff, seed=args.seed)
    agent = agent_class(env)
    # Instantiate game runner and loop
    runner = Runner(env, agent, args.render, args.verbose, args.vverbose, args.parameters, args.level, args.niter)
    final_reward = runner.loop(iterations=args.niter, epochs=args.epochs)
    print("Obtained a final reward of {}".format(final_reward))


if __name__ == "__main__":
    main()
