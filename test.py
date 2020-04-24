import os
import argparse

from utils.utils import filter_tf_warnings
from utils.rl_agent import RLAgent
from utils.test_utils import AgentTestFramework, TEST_EPISODES


def main():
    parser = argparse.ArgumentParser(description='Test a model inside a directory or a set of models.')
    parser.add_argument("logdir",
                        help="log directory")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Run for all subdirs of 'logdir' (default: run for 'logdir')")
    parser.add_argument("-l", "--latest", action="store_true",
                        help="Use latest dir inside 'logdir' (default: run for 'logdir')")
    parser.add_argument('-e', '--episodes', type=int, default=TEST_EPISODES,
                        help="Number of test episodes (default: 2500)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print game boards (default: Don't print)")
    parser.add_argument('-s', '--model_suffix', type=str, default="",
                        help="Use a suffix for the model file e.g _best (default: )")
    args = parser.parse_args()

    log_dir = args.logdir

    all_subdirs = [log_dir + '/' + d for d in sorted(os.listdir(log_dir)) if os.path.isdir(log_dir + '/' + d)]

    if args.latest:
        log_dir = sorted(all_subdirs)[-1]

    if not args.all:
        all_subdirs = [log_dir]

    filter_tf_warnings()

    for sub_dir in all_subdirs:

        print("\nTesting model in dir:", sub_dir)

        agent = RLAgent(sub_dir, model_suffix=args.model_suffix)

        model_name = agent.name + args.model_suffix

        print("Model name:", model_name)

        out_file = "test_{}.csv".format(model_name)

        test_framework = AgentTestFramework(agent, args.episodes, sub_dir, out_file=out_file, verbose=args.verbose)
        test_framework.test()


if __name__ == "__main__":
    main()
