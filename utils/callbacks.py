import os
import time
import datetime
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from tqdm.auto import tqdm
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv

from utils.rl_agent import RLAgent
from utils.test_utils import AgentTestFramework


class PlotTestSaveCallback(object):
    """
    Callback for plotting training outcomes, test the agent, and save the model.
    """

    def __init__(self, train_episodes, eval_freq, log_dir, alg_name, self_play: bool, env):
        super(PlotTestSaveCallback, self).__init__()
        self.train_episodes = train_episodes
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.self_play = self_play
        self.pbar = None
        self.best_mean_reward = -np.inf
        self.current_episode = 0
        self.save_logs = []
        self.start_time = None
        self.plot_mr = None
        self.x_values = []
        self.mean_rewards = []
        self.plot_outcomes = None
        self.wins_perc = []
        self.draws_perc = []
        self.losses_perc = []
        self.invalids_perc = []
        self.cross_first_perc = []
        self.figsize = (12, 5)
        self.results = pd.DataFrame()
        self.model = None
        self.env_agent = None
        self.env = env
        self.agent = None
        self.test_framework = None

        with open(self.log_dir + "/params.json", "r") as f:
            params = json.load(f)
            self.alg_name = params["alg"]
            self.plot_mr_fig = "{}/train_rewards_{}.png".format(self.log_dir, self.alg_name)

            self.env_agent_name = params["env_agent"]

            if self_play:
                self.env_agent_name = "Self"

            self.env_exploration_rate = params["env_exploration_rate"]

            self.plot_mr_title = "{} Training Mean Rewards Vs {}".format(
                self.alg_name, self.env_agent_name)

            self.plot_outcomes_title = "{} Training Outcomes Vs {}".format(
                self.alg_name, self.env_agent_name)

            if self.env_exploration_rate > 0:
                self.plot_mr_title = "{} eps={}".format(
                    self.plot_mr_title, self.env_exploration_rate)

                self.plot_outcomes_title = "{} eps={}".format(
                    self.plot_outcomes_title, self.env_exploration_rate)

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.train_episodes)
        self.start_time = time.time()

        return self.__call__

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback

        # Save model at the end
        self.model.save(self.log_dir + "/" + self.alg_name)

        # Save train logs
        elapsed_time_seconds = time.time() - self.start_time
        elapsed_time_h = datetime.timedelta(seconds=elapsed_time_seconds)
        elapsed_time_h = str(datetime.timedelta(days=elapsed_time_h.days, seconds=elapsed_time_h.seconds))

        train_logs = {"end_episode": self.current_episode,
                      "end_mean_reward": self.mean_rewards[-1],
                      "best_mean_reward": self.best_mean_reward,
                      "elapsed_time": elapsed_time_seconds,
                      "elapsed_time_h": elapsed_time_h,
                      "save_logs": self.save_logs}

        with open(self.log_dir + "/train_logs.json", "w") as f:
            json.dump(OrderedDict(train_logs), f, indent=4)

        # Final test
        self.test_framework.test(train_episode=self.current_episode, elapsed_time=elapsed_time_h)

        self.pbar.n = self.train_episodes
        self.pbar.update(0)
        self.pbar.close()

    def __call__(self, locals_, globals_):
        """
        This method will be called by the model.

        :param locals_: (dict)
        :param globals_: (dict)
        :return: (bool)
        """

        # print("\n\n\nlocals_.keys:", locals_.keys())
        # print("\nlocals_:", locals_)
        # print("\n\n\nglobals_.keys:", globals_.keys())
        # print("\nglobals_:", globals_)
        # exit(0)

        if 'DQN' in globals_:
            if 'info' in locals_:
                # print("\ninfo: {}".format(locals_['info']))
                if 'episode' in locals_['info']:
                    ep_info = locals_['info']['episode']
                    # print("\nepisode: {}".format(ep_info))
                    if self.results.empty:
                        self.results = pd.DataFrame([ep_info])
                    else:
                        self.results = self.results.append(ep_info, ignore_index=True)
                else:
                    return
            else:
                return

        elif 'PPO2' in globals_:
            # print("ep_infos:", locals_['ep_infos'])
            df = pd.DataFrame(locals_['ep_infos'])
            # print("current:", len(self.results))
            # print("new:", len(df))
            self.results = pd.concat([self.results, df], ignore_index=True)

        if len(self.results.index) < self.eval_freq:
            return

        # Get the self object of the model
        self.model = locals_['self']

        # self.results = load_results(self.log_dir)

        # print(self.results)

        while len(self.results.index) >= self.eval_freq:

            self.current_episode += self.eval_freq

            # Plot Mean Reward
            rewards = self.results['r']
            mean_reward = float(np.mean(rewards[:self.eval_freq]))

            self.x_values.append(self.current_episode)
            self.mean_rewards.append(mean_reward)

            if self.plot_mr is None:
                fig1, ax1 = plt.subplots(figsize=self.figsize)
                ax1.set_title(self.plot_mr_title)
                ax1.set_xlabel('Number of Episodes')
                ax1.set_ylabel('Mean {} Episode Reward'.format(self.eval_freq))

                line, = ax1.plot(self.x_values, self.mean_rewards)
                self.plot_mr = (line, ax1, fig1)
            else:
                self.plot_mr[0].set_data(self.x_values, self.mean_rewards)
                self.plot_mr[1].relim()
                self.plot_mr[1].autoscale_view(True, True, True)
                self.plot_mr[2].tight_layout()
                self.plot_mr[2].canvas.draw()

            self.plot_mr[1].set_title("{} | Best: {:5.2f}".format(self.plot_mr_title, self.best_mean_reward))
            self.plot_mr[2].savefig(self.plot_mr_fig)

            # print(self.x_values)
            # print(self.mean_rewards)

            # Plot Outcomes Percentage
            outcomes = self.results['outcome']
            player_one = self.results['player_one']

            # print(outcomes.tail())
            # print(player_one.tail())
            # print()

            win_count = 0
            draw_count = 0
            loss_count = 0
            invalid_count = 0
            cross_first_player_count = 0

            for i in range(self.eval_freq):

                if outcomes[i] == TicTacToeEnv.CROSS:
                    win_count += 1
                elif outcomes[i] == TicTacToeEnv.DRAW:
                    draw_count += 1
                elif outcomes[i] == TicTacToeEnv.NAUGHT:
                    loss_count += 1
                elif outcomes[i] == TicTacToeEnv.INVALID:
                    invalid_count += 1

                if player_one[i] == 'X':
                    cross_first_player_count += 1

            self.wins_perc.append(win_count * 100.0 / self.eval_freq)
            self.draws_perc.append(draw_count * 100.0 / self.eval_freq)
            self.losses_perc.append(loss_count * 100.0 / self.eval_freq)
            self.invalids_perc.append(invalid_count * 100.0 / self.eval_freq)
            self.cross_first_perc.append(cross_first_player_count * 100.0 / self.eval_freq)

            if self.plot_outcomes is None:
                fig2, ax2 = plt.subplots(figsize=self.figsize)
                ax2.set_title(self.plot_outcomes_title)
                ax2.set_xlabel('Number of Episodes')
                ax2.set_ylabel('Episode Outcomes %')

                line1, = ax2.plot(self.x_values, self.wins_perc, 'g-', label="Wins")
                line2, = ax2.plot(self.x_values, self.draws_perc, 'b-', label="Draws")
                line3, = ax2.plot(self.x_values, self.losses_perc, 'r-', label="Losses")
                line4, = ax2.plot(self.x_values, self.invalids_perc, 'y-', label="Invalids")
                self.plot_outcomes = (line1, line2, line3, line4, ax2, fig2)
            else:
                self.plot_outcomes[0].set_data(self.x_values, self.wins_perc)
                self.plot_outcomes[1].set_data(self.x_values, self.draws_perc)
                self.plot_outcomes[2].set_data(self.x_values, self.losses_perc)
                self.plot_outcomes[3].set_data(self.x_values, self.invalids_perc)
                self.plot_outcomes[4].relim()
                self.plot_outcomes[4].autoscale_view(True, True, True)
                self.plot_outcomes[4].legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
                self.plot_outcomes[5].tight_layout()
                self.plot_outcomes[5].canvas.draw()

            self.plot_outcomes[5].savefig(self.log_dir + "/train_outcomes_" + self.alg_name + ".png")

            # Write results to CSV file
            rows = zip(self.x_values, self.wins_perc, self.draws_perc, self.losses_perc,
                       self.invalids_perc, self.mean_rewards, self.cross_first_perc)

            with open(self.log_dir + "/train_results_" + self.alg_name + ".csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(("Episode", "Wins", "Draws", "Losses", "Invalids", "Mean Rewards", "X_First"))

                for row in rows:
                    new_row = [x if type(x) is not float else format(x, '.1f') for x in row]
                    writer.writerow(new_row)

                writer.writerow(['Min',
                                 '{:.1f}'.format(np.min(self.wins_perc)),
                                 '{:.1f}'.format(np.min(self.draws_perc)),
                                 '{:.1f}'.format(np.min(self.losses_perc)),
                                 '{:.1f}'.format(np.min(self.invalids_perc)),
                                 '{:.1f}'.format(np.min(self.mean_rewards)),
                                 '{:.1f}'.format(np.min(self.cross_first_perc))])

                writer.writerow(['Max',
                                 '{:.1f}'.format(np.max(self.wins_perc)),
                                 '{:.1f}'.format(np.max(self.draws_perc)),
                                 '{:.1f}'.format(np.max(self.losses_perc)),
                                 '{:.1f}'.format(np.max(self.invalids_perc)),
                                 '{:.1f}'.format(np.max(self.mean_rewards)),
                                 '{:.1f}'.format(np.max(self.cross_first_perc))])

                writer.writerow(['Mean',
                                 '{:.1f}'.format(np.mean(self.wins_perc)),
                                 '{:.1f}'.format(np.mean(self.draws_perc)),
                                 '{:.1f}'.format(np.mean(self.losses_perc)),
                                 '{:.1f}'.format(np.mean(self.invalids_perc)),
                                 '{:.1f}'.format(np.mean(self.mean_rewards)),
                                 '{:.1f}'.format(np.mean(self.cross_first_perc))])

            # New best model, save the agent
            if mean_reward >= self.best_mean_reward:
                self.best_mean_reward = mean_reward

                self.save_logs.append({"episode": self.current_episode,
                                       "best_mean_reward": self.best_mean_reward})

                # print("Saving new best model. Best Mean Reward: {:5.2f} | Episode: {:6d}".format(
                #     mean_reward, self.current_episode)

                self.model.save(self.log_dir + "/" + self.alg_name + "_best")

            # print("before drop:", self.results)
            self.results.drop(range(self.eval_freq),  inplace=True)
            self.results.reset_index(drop=True, inplace=True)
            # print("after drop:", self.results)

        if not self.agent:
            self.agent = RLAgent(self.log_dir, model=self.model)
            self.test_framework = AgentTestFramework(self.agent, 1000, self.log_dir, verbose=False)
        else:
            self.agent.model = self.model

        if self.current_episode >= self.train_episodes:
            return False

        elapsed_time_seconds = time.time() - self.start_time
        elapsed_time_h = datetime.timedelta(seconds=elapsed_time_seconds)
        elapsed_time_h = str(datetime.timedelta(days=elapsed_time_h.days, seconds=elapsed_time_h.seconds))

        self.test_framework.test(train_episode=self.current_episode, elapsed_time=elapsed_time_h)

        if self.self_play:

            # self.model.save(self.log_dir + "/" + self.alg_name)

            # print("self.current_episode:", self.current_episode)

            if not self.env_agent:

                envs = []

                if 'DQN' in globals_:

                    env = self.env.env.env
                    envs.append(env)

                elif 'PPO2' in globals_:
                    runner = locals_['runner']

                    # print("runner.env.num_envs:", runner.env.num_envs)

                    for env in runner.env.envs:
                        envs.append(env.env.env)

                self.env_agent = RLAgent(self.log_dir, model=self.model, deterministic=False)

                for env in envs:
                    env.self_play = True
                    env.naught_agent = self.env_agent

                #     print("env:", env)
                #     print("env id:", hex(id(env)))
                #     print("naught_agent id:", hex(id(env.naught_agent)))

                # print("env_agent id:", hex(id(self.env_agent)))
                # print()

            else:
                self.env_agent.model = self.model

        self.pbar.n = self.current_episode
        self.pbar.update(0)

        return True
