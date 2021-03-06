import os
import argparse
import json
from datetime import datetime
from collections import OrderedDict

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.random_agent import RandomAgent
from gym_tictactoe.agents.min_max_agent import MinMaxAgent

from utils.test_utils import AgentTestFramework


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, default=100000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    agents = [RandomAgent(), MinMaxAgent(side=TicTacToeEnv.CROSS)]

    for agent in agents:

        now = datetime.now()

        log_dir = "logs_minmax_random/{}_{}_{}".format(now.strftime('%Y%m%d-%H%M%S'),
                                                       agent.name, args.episodes)

        os.makedirs(log_dir, exist_ok=True)

        print("\nlog_dir:", log_dir)

        params = OrderedDict()
        params['agent'] = agent.name
        params['num_episodes'] = args.episodes
        params['datetime'] = now.replace(microsecond=0).isoformat()

        with open(log_dir + "/params.json", "w") as f:
            json.dump(params, f, indent=4)

        test_framework = AgentTestFramework(agent, args.episodes, log_dir, verbose=args.verbose)
        test_framework.test()
