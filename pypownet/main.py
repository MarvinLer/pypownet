__author__ = 'marvinler'
import argparse
from pypownet.env import RunEnv
from pypownet.runner import Runner, BatchRunner
import pypownet.agent

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str,
                    help='Class to use for the agent. Must be within the \'agent.py\' file.')
parser.add_argument('--niter', type=int, metavar='n', default='100',
                    help='Number of iterations to simulate.')
parser.add_argument('--batch', type=int, metavar='nagent', default=None,
                    help='Batch run several agent at the same time.')
parser.add_argument('--verbose', action='store_true',
                    help='Display cumulative reward results at each step.')


def main():
    args = parser.parse_args()
    agent_class = eval('pypownet.agent.{}'.format(args.agent))
    env_class = RunEnv

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        runner = BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward = runner.loop(iterations=args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
    else:
        print("Running a single instance simulation...")
        runner = Runner(env_class(), agent_class(), args.verbose)
        final_reward = runner.loop(iterations=args.niter)
        print("Obtained a final reward of {}".format(final_reward))


if __name__ == "__main__":
    main()