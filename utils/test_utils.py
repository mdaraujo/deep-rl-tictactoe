import os
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from pathlib import Path

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.random_agent import RandomAgent
from gym_tictactoe.agents.min_max_agent import MinMaxAgent

from utils.utils import get_env, get_elapsed_time, FIG_SIZE
from utils.rl_agent import RLAgent

TEST_HEADER = ["Episodes", "FirstPlayer", "SecondPlayer", "Wins", "Draws",
               "Losses", "Invalids", "MeanReward", "TestTime",
               "NewBoards", "TotalBoards", "NewEnvBoards", "TotalEnvBoards"]


class AgentTestFramework:

    def __init__(self, test_agent, num_episodes, log_dir, out_file="test_outcomes.csv", verbose=False):
        self.test_agent = test_agent
        self.num_episodes = num_episodes
        self.log_dir = log_dir
        self.out_file = out_file
        self.verbose = verbose
        self.random_agent_first = RandomAgent()
        self.random_agent_second = RandomAgent()
        self.minmax_agent_first = MinMaxAgent()
        self.minmax_agent_second = MinMaxAgent()
        self.res_random_first = []
        self.res_random_second = []
        self.res_minmax_first = []
        self.res_minmax_second = []
        self.res_self = []
        self.current_score = 0
        self.best_score = 0
        self.current_idx = 0
        self.best_score_idx = 0
        self.best_train_episode = 0
        self.plot = None
        self.x_values = []
        self.scores = []
        self.other_params = OrderedDict()
        self.all_board_states = []
        self.n_self_episodes = num_episodes

        with open(os.path.join(self.log_dir, 'params.json'), 'r') as f:
            self.train_params = json.load(f, object_pairs_hook=OrderedDict)

    def test(self, train_episode=None, train_time=None):

        start_time = time.time()

        self.all_board_states.clear()

        self.res_random_first.append(self.evaluate(self.random_agent_first, 'X', self.num_episodes))
        self.res_random_second.append(self.evaluate(self.random_agent_second, 'O', self.num_episodes))
        self.res_minmax_first.append(self.evaluate(self.minmax_agent_first, 'X', self.num_episodes))
        self.res_minmax_second.append(self.evaluate(self.minmax_agent_second, 'O', self.num_episodes))

        # Limit number of episodes for self play test
        if self.n_self_episodes > 500:
            self.n_self_episodes = 500
        self.res_self.append(self.evaluate(self.test_agent, '-', self.n_self_episodes))

        # Calculate Score
        # Average of wins vs random and draws vs minmax
        self.current_score = self.res_random_first[-1][0] + self.res_random_second[-1][0] + \
            self.res_minmax_first[-1][1] + self.res_minmax_second[-1][1]

        self.current_score /= 4

        if self.current_score >= self.best_score:
            self.best_score = self.current_score
            self.best_score_idx = self.current_idx
            self.best_train_episode = train_episode

        self.x_values.append(train_episode)
        self.scores.append(self.current_score)

        if len(self.res_random_first) > 1:
            self.plot_test_outcomes()

        self.other_params['SumEnvBoards'] = len(self.random_agent_first.board_states) \
            + len(self.random_agent_second.board_states) \
            + len(self.minmax_agent_first.board_states) \
            + len(self.minmax_agent_second.board_states)

        self.other_params['Score'] = self.current_score

        if train_episode:
            self.other_params['TrainEpisode'] = train_episode

        if train_time:
            self.other_params['TrainTime'] = train_time

        self.all_board_states.append({"test_agent": self.test_agent.name,
                                      "total_boards": len(self.test_agent.board_states),
                                      "boards": self.test_agent.board_states})

        with open(self.log_dir + "/test_agents_states.json", "w") as f:
            json.dump(self.all_board_states, f, indent=4)

        _, test_time_h = get_elapsed_time(time.time(), start_time)

        self.other_params['TotalTestTime'] = test_time_h

        self.write_test_outcomes()

        self.current_idx += 1

    def write_test_outcomes(self):

        rows = []

        rows.append([self.num_episodes, self.test_agent.name, self.random_agent_first.name]
                    + self.res_random_first[-1])

        rows.append([self.num_episodes, self.random_agent_second.name, self.test_agent.name]
                    + self.res_random_second[-1])

        rows.append([self.num_episodes, self.test_agent.name, self.minmax_agent_first.name]
                    + self.res_minmax_first[-1])

        rows.append([self.num_episodes, self.minmax_agent_second.name, self.test_agent.name]
                    + self.res_minmax_second[-1])

        rows.append([self.n_self_episodes, self.test_agent.name, self.test_agent.name]
                    + self.res_self[-1])

        with open(os.path.join(self.log_dir, self.out_file), 'a') as f:
            writer = csv.writer(f)

            header = TEST_HEADER + list(self.other_params.keys()) + list(self.train_params.keys())

            writer.writerow(header)

            for row in rows:
                new_row = row + list(self.other_params.values())
                new_row = [x if type(x) is not float else format(x, '.2f') for x in new_row]
                new_row.extend(self.train_params.values())
                writer.writerow(new_row)

            writer.writerow([])

    def plot_test_outcomes(self):

        wins_random_first = [x[0] for x in self.res_random_first]
        wins_random_second = [x[0] for x in self.res_random_second]
        draws_minmax_first = [x[1] for x in self.res_minmax_first]
        draws_minmax_second = [x[1] for x in self.res_minmax_second]

        if self.plot is None:
            fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
            ax1.set_xlabel('Train Episode')
            ax1.set_ylabel('Outcomes %')

            line1, = ax1.plot(self.x_values, wins_random_first, 'blue')
            line2, = ax1.plot(self.x_values, wins_random_second, 'cyan')
            line3, = ax1.plot(self.x_values, draws_minmax_first, 'darkgreen')
            line4, = ax1.plot(self.x_values, draws_minmax_second, 'mediumseagreen')
            line5, = ax1.plot(self.x_values, self.scores, 'orangered')

            self.plot = (ax1, fig1, line1, line2, line3, line4, line5)

        self.plot[0].set_title("{} Test Outcomes | Best Score {:5.2f} at Ep {}".format(self.test_agent.name,
                                                                                       self.best_score,
                                                                                       self.best_train_episode))

        self.plot[2].set_label("WinsVsRandom1º | {:5.2f}".format(self.res_random_first[self.best_score_idx][0]))
        self.plot[3].set_label("WinsVsRandom2º | {:5.2f}".format(self.res_random_second[self.best_score_idx][0]))
        self.plot[4].set_label("DrawsVsMinMax1º| {:5.2f}".format(self.res_minmax_first[self.best_score_idx][1]))
        self.plot[5].set_label("DrawsVsMinMax2º| {:5.2f}".format(self.res_minmax_second[self.best_score_idx][1]))
        self.plot[6].set_label("Score (Average)| {:5.2f}".format(self.best_score))

        self.plot[2].set_data(self.x_values, wins_random_first)
        self.plot[3].set_data(self.x_values, wins_random_second)
        self.plot[4].set_data(self.x_values, draws_minmax_first)
        self.plot[5].set_data(self.x_values, draws_minmax_second)
        self.plot[6].set_data(self.x_values, self.scores)

        self.plot[0].relim()
        self.plot[0].autoscale_view(True, True, True)
        self.plot[0].legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
        self.plot[1].tight_layout()
        self.plot[1].canvas.draw()

        self.plot[1].savefig(self.log_dir + "/test_outcomes.png")

    def evaluate(self, env_agent, player_one_char, n_episodes):

        start_time = time.time()

        boards = len(self.test_agent.board_states)
        env_boards = len(env_agent.board_states)

        if self.verbose:
            print("\n\n --- Evaluating vs {}. First Player: {}. Episodes: {}.".format(
                env_agent.name, player_one_char, n_episodes))

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
        for episode in range(1, n_episodes + 1):
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

            if self.verbose and episode == int(n_episodes/2):
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
            print("Episode: {:6d} | Mean reward: {:5.2f}".format(n_episodes, mean_reward))

        total_boards = len(self.test_agent.board_states)
        new_boards = total_boards - boards

        total_env_boards = len(env_agent.board_states)
        new_env_boards = total_env_boards - env_boards

        if not isinstance(env_agent, RLAgent):
            board_states = {"env_agent": env_agent.name,
                            "player_one": player_one_char,
                            "current_idx": self.current_idx,
                            "total_env_boards": total_env_boards,
                            "env_boards": env_agent.board_states.copy()}

            self.all_board_states.append(board_states)

        _, test_time_h = get_elapsed_time(time.time(), start_time)

        return [win_count * 100.0 / n_episodes,
                draw_count * 100.0 / n_episodes,
                loss_count * 100.0 / n_episodes,
                invalid_count * 100.0 / n_episodes,
                float(mean_reward),
                test_time_h,
                new_boards,
                total_boards,
                new_env_boards,
                total_env_boards]
