__author__ = 'marvinler'
import argparse
from pypownet.env import RunEnv
from pypownet.runner import Runner, BatchRunner
import pypownet.agent

parser = argparse.ArgumentParser(description='CLI tool to run experiments using PyPowNet.')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='class to use for the agent (must be within the \'pypownet/agent.py\' file); '
                         'default class Agent')
parser.add_argument('--niter', type=int, metavar='n', default='100',
                    help='number of iterations to simulate; default 100')
parser.add_argument('--batch', type=int, metavar='n_agent', default=None,
                    help='number of game instances to run simultaneously; default 1')
parser.add_argument('--case', type=int, metavar='n_agent', default=118,
                    help='grid case to use; this is equal to the number of substations of the grid to be used; '
                         'default 118; among 14 or 118')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='display live info of the current experiment including reward, cumulative reward')
parser.add_argument('-vv', '--vverbose', action='store_true',
                    help='display live info + observations and actions played')
parser.add_argument('-r', '--render', action='store_true',
                    help='render the power network observation at each timestep (not available if --batch is not 1)')
parser.add_argument('--start', type=int, default=None,
                    help='id of the timestep to start the game at (>= 0)')


def main():
    args = parser.parse_args()
    env_class = RunEnv
    agent_class = eval('pypownet.agent.{}'.format(args.agent))

    start_id = args.start

    if not args.batch:
        print("Running a single instance simulation on case", args.case, "for", args.niter, "iterations...")
        # Instantiate environment and agent
        env = env_class(grid_case=args.case, start_id=start_id)
        agent = agent_class(env)
        # Instantiate game runner and loop
        runner = Runner(env, agent, args.render, args.verbose, args.vverbose)
        final_reward = runner.loop(iterations=args.niter)
        print("Obtained a final reward of {}".format(final_reward))
    else:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        runner = BatchRunner(env_class, agent_class, args.batch, args.verbose, args.render)
        final_reward = runner.loop(iterations=args.niter)
        print("Obtained a final average reward of {}".format(final_reward))


if __name__ == "__main__":
    main()