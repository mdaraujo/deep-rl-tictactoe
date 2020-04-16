import os
import json
import argparse
from datetime import datetime
from collections import OrderedDict

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.random_agent import RandomAgent
from gym_tictactoe.agents.min_max_agent import MinMaxAgent

from utils.test_utils import test_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, default=100000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    num_episodes = args.episodes

    agents = [RandomAgent(), MinMaxAgent(side=TicTacToeEnv.CROSS)]

    for agent in agents:

        now = datetime.now()

        params = {"datetime": now.isoformat(),
                  "agent": agent.name,
                  "obs_format": agent.obs_format,
                  "num_episodes": num_episodes}

        log_dir = "logs_minmax_random/{}_{}_{}".format(now.strftime('%Y%m%d-%H%M%S'),
                                                     agent.name, num_episodes,)

        os.makedirs(log_dir, exist_ok=True)

        print("\nlog_dir:", log_dir)

        with open(log_dir + "/params.json", "w") as f:
            json.dump(OrderedDict(params), f, indent=4)

        test_agent(agent, log_dir, num_episodes, verbose=args.verbose)
