import os
import time
import json
import csv
import numpy as np
from collections import OrderedDict
from pathlib import Path

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.random_agent import RandomAgent
from gym_tictactoe.agents.min_max_agent import MinMaxAgent

from utils.utils import get_env, get_elapsed_time
from utils.rl_agent import RLAgent

TEST_HEADER = ["Episodes", "First Player", "Second Player", "Wins", "Draws",
               "Losses", "Invalids", "Mean Reward"]


class AgentTestFramework:

    def __init__(self, test_agent, num_episodes, log_dir, out_file="test_outcomes.csv", verbose=False):
        self.test_agent = test_agent
        self.num_episodes = num_episodes
        self.log_dir = log_dir
        self.out_file = out_file
        self.verbose = verbose
        self.random_agent = RandomAgent()
        self.minmax_agent = MinMaxAgent()
        self.res_random_first = []
        self.res_random_second = []
        self.res_minmax_first = []
        self.res_minmax_second = []
        self.train_params = None
        self.current_score = 0

    def test(self, train_episode=None, train_time=None):

        start_time = time.time()

        if self.verbose:
            print("\nEvaluating going first vs random in {} episodes.".format(self.num_episodes))

        self.res_random_first.append(self.evaluate(self.random_agent, 'X'))

        if self.verbose:
            print("\nEvaluating going second vs random in {} episodes.".format(self.num_episodes))

        self.res_random_second.append(self.evaluate(self.random_agent, 'O'))

        if self.verbose:
            print("\nEvaluating going first vs minmax in {} episodes.".format(self.num_episodes))

        self.res_minmax_first.append(self.evaluate(self.minmax_agent, 'X'))

        if self.verbose:
            print("\nEvaluating going second vs minmax in {} episodes.".format(self.num_episodes))

        self.res_minmax_second.append(self.evaluate(self.minmax_agent, 'O'))

        if not train_episode or not train_time:

            train_logs_file = Path(os.path.join(self.log_dir, "train_logs.json"))
            if train_logs_file.is_file():

                with train_logs_file.open() as f:
                    train_logs = json.load(f)

                    if 'elapsed_time_h' in train_logs:
                        train_time = train_logs['elapsed_time_h']

                    if 'end_episode' in train_logs:
                        train_episode = train_logs['end_episode']

        if not self.train_params:
            with open(os.path.join(self.log_dir, 'params.json'), 'r') as f:
                self.train_params = json.load(f, object_pairs_hook=OrderedDict)

        self.train_params['train_time'] = train_time

        # Calculate Score
        # Average of wins versus random and draws vs minmax
        self.current_score = self.res_random_first[-1][0] + self.res_random_second[-1][0] + \
            self.res_minmax_first[-1][1] + self.res_minmax_second[-1][1]

        self.current_score /= 4

        # Append results
        rows = []
        rows.append([self.num_episodes, self.test_agent.name, self.random_agent.name] + self.res_random_first[-1])
        rows.append([self.num_episodes, self.random_agent.name, self.test_agent.name] + self.res_random_second[-1])
        rows.append([self.num_episodes, self.test_agent.name, self.minmax_agent.name] + self.res_minmax_first[-1])
        rows.append([self.num_episodes, self.minmax_agent.name, self.test_agent.name] + self.res_minmax_second[-1])

        _, test_time_h = get_elapsed_time(time.time(), start_time)

        self.train_params['test_time'] = test_time_h

        with open(os.path.join(self.log_dir, self.out_file), 'a') as f:
            writer = csv.writer(f)

            header = TEST_HEADER + ['Score', 'train_episode'] + list(self.train_params.keys())

            writer.writerow(header)

            for row in rows:
                new_row = row + [self.current_score, train_episode]
                new_row = [x if type(x) is not float else format(x, '.2f') for x in new_row]
                new_row.extend(self.train_params.values())
                writer.writerow(new_row)

            writer.writerow([])

    def evaluate(self, env_agent, player_one_char):

        win_count = 0
        loss_count = 0
        draw_count = 0
        invalid_count = 0

        # This function will only work for a single Environment
        env = get_env(self.test_agent.obs_format, env_agent, player_one_char, self.test_agent.rewards)

        if isinstance(env_agent, RLAgent):
            # print("env:", env.env)
            # print("env id:", hex(id(env.env)))
            # print("env.self_play = True")
            env.env.self_play = True

        all_episode_rewards = []
        for episode in range(1, self.num_episodes + 1):
            episode_rewards = []
            done = False
            obs = env.reset()

            if self.verbose:
                print("\n\nGame number {}".format(episode))

            while not done:

                if self.verbose:
                    env.render()

                action = self.test_agent.play(obs)

                obs, reward, done, info = env.step(action)

                episode_rewards.append(reward)

                # if done and info['outcome'] == TicTacToeEnv.INVALID:
                if self.verbose:

                    naught_action = -1
                    if 'naught_action' in info:
                        naught_action = info['naught_action']

                    info_str = "CROSS: {:1d} | NAUGHT: {:2d} | Reward: {:2.0f}".format(
                        action, naught_action, reward)

                    if done:
                        info_str = "{} | Outcome: {} | First Player: {}".format(
                            info_str, info['outcome'], info['player_one'])

                        print(info_str)
                        env.render()
                        print()
                    else:
                        print(info_str)

            all_episode_rewards.append(sum(episode_rewards))

            if self.verbose and episode == int(self.num_episodes/2):
                mean_reward = np.mean(all_episode_rewards)
                print("Episode: {:6d} | Mean reward: {:5.2f}".format(episode, mean_reward))

            outcome = info['outcome']

            if outcome == TicTacToeEnv.CROSS:
                win_count += 1
            elif outcome == TicTacToeEnv.DRAW:
                draw_count += 1
            elif outcome == TicTacToeEnv.NAUGHT:
                loss_count += 1
            elif outcome == TicTacToeEnv.INVALID:
                invalid_count += 1

        mean_reward = np.mean(all_episode_rewards)
        if self.verbose:
            print("Episode: {:6d} | Mean reward: {:5.2f}".format(self.num_episodes, mean_reward))

        return [win_count * 100.0 / self.num_episodes,
                draw_count * 100.0 / self.num_episodes,
                loss_count * 100.0 / self.num_episodes,
                invalid_count * 100.0 / self.num_episodes,
                float(mean_reward)]
