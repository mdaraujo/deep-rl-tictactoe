import os
import argparse
import warnings
import logging

from utils.rl_agent import RLAgent
from utils.test_utils import AgentTestFramework

# Filter tensorflow version warnings
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir",
                        help="log directory")
    parser.add_argument("-l", "--latest", action="store_true",
                        help="use latest dir inside 'logdir'")
    parser.add_argument("-a", "--all", action="store_true",
                        help="run for all subdirs of 'logdir'")
    parser.add_argument('-e', '--episodes', type=int, default=5000)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument('-s', '--model_suffix', type=str, default="")
    args = parser.parse_args()

    log_dir = args.logdir

    all_subdirs = [log_dir + '/' + d for d in sorted(os.listdir(log_dir)) if os.path.isdir(log_dir + '/' + d)]

    if args.latest:
        log_dir = sorted(all_subdirs)[-1]

    if not args.all:
        all_subdirs = [log_dir]

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
