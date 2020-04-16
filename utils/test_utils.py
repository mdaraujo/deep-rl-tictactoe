import os
import json
import csv
import numpy as np
from collections import OrderedDict
from pathlib import Path

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.base import Agent
from gym_tictactoe.agents.random_agent import RandomAgent
from gym_tictactoe.agents.min_max_agent import MinMaxAgent

from utils.utils import get_env
from utils.rl_agent import RLAgent

TEST_HEADER = ["Episodes", "First Player", "Second Player", "Wins", "Draws",
               "Losses", "Invalids", "Mean Reward"]


def battle(agent: Agent, env_agent: Agent, player_one_char, num_episodes=1000, verbose=False):

    win_count = 0
    loss_count = 0
    draw_count = 0
    invalid_count = 0

    # This function will only work for a single Environment
    env = get_env(agent.obs_format, env_agent, player_one_char, agent.rewards)

    if isinstance(env_agent, RLAgent):
        # print("env:", env.env)
        # print("env id:", hex(id(env.env)))
        # print("env.self_play = True")
        env.env.self_play = True

    all_episode_rewards = []
    for episode in range(1, num_episodes + 1):
        episode_rewards = []
        done = False
        obs = env.reset()

        if verbose:
            print("\n\nGame number {}".format(episode))

        while not done:

            if verbose:
                env.render()

            action = agent.play(obs)

            obs, reward, done, info = env.step(action)

            episode_rewards.append(reward)

            # if done and info['outcome'] == TicTacToeEnv.INVALID:
            if verbose:

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

        if verbose and episode == int(num_episodes/2):
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
    if verbose:
        print("Episode: {:6d} | Mean reward: {:5.2f}".format(num_episodes, mean_reward))

    return [win_count * 100.0 / num_episodes,
            draw_count * 100.0 / num_episodes,
            loss_count * 100.0 / num_episodes,
            invalid_count * 100.0 / num_episodes,
            float(mean_reward)]


def battle_results(agent: Agent, env_agent: Agent, player_one_char, num_episodes=1000, verbose=False):

    if player_one_char == 'X':
        row = [num_episodes, agent.name, env_agent.name]
    elif player_one_char == 'O':
        row = [num_episodes, env_agent.name, agent.name]

    results = battle(agent, env_agent, player_one_char, num_episodes, verbose=verbose)

    row.extend(results)

    if verbose:
        print(row)

    return row


def evaluate(agent: Agent, env_agent: Agent, num_episodes=1000, verbose=False):

    rows = []

    # Agent playing as First Player
    if verbose:
        print("\nEvaluating agent as First Player in {} episodes against {}".format(num_episodes, env_agent.name))
    rows.append(battle_results(agent, env_agent, 'X', num_episodes, verbose))

    # Agent playing as Second Player
    if verbose:
        print("\nEvaluating agent as Second Player in {} episodes against {}".format(num_episodes, env_agent.name))
    rows.append(battle_results(agent, env_agent, 'O', num_episodes, verbose))

    mean_row = [rows[0][0] + rows[1][0], "Mixed", "Mixed"]

    mean_results = []

    for i in range(3, 8):
        mean_results.append((rows[0][i] + rows[1][i]) / 2)

    mean_row.extend(mean_results)

    rows.append(mean_row)

    for i in range(len(rows)):
        rows[i] = [x if type(x) is not float else format(x, '.2f') for x in rows[i]]

    return rows


def test_agent(agent: Agent, log_dir, num_episodes, out_file="test_outcomes.csv", train_episode=None, verbose=False):

    rows = []

    env_agent = RandomAgent()

    rows.extend(evaluate(agent, env_agent, num_episodes, verbose))

    env_agent = MinMaxAgent()

    rows.extend(evaluate(agent, env_agent, num_episodes, verbose))

    elapsed_time = None
    end_episode = None

    train_logs_file = Path(os.path.join(log_dir, "train_logs.json"))
    if train_logs_file.is_file():

        with train_logs_file.open() as f:
            train_logs = json.load(f)

            if 'elapsed_time_h' in train_logs:
                elapsed_time = train_logs['elapsed_time_h']

            if 'end_episode' in train_logs:
                end_episode = train_logs['end_episode']

    with open(os.path.join(log_dir, "params.json"), "r") as f:
        train_params = json.load(f, object_pairs_hook=OrderedDict)

        if elapsed_time:
            train_params['elapsed_time'] = elapsed_time

        mode = "w"
        if end_episode:
            train_params['train_episode'] = end_episode
        elif train_episode:
            mode = "a"
            train_params['train_episode'] = train_episode

        # Write results
        with open(os.path.join(log_dir, out_file), mode) as f:
            writer = csv.writer(f)

            header = TEST_HEADER + list(train_params.keys())

            writer.writerow(header)

            for row in rows:
                row.extend(train_params.values())

                writer.writerow(row)

            if train_episode:
                writer.writerow([])
