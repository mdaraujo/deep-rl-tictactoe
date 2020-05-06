import os
import argparse
import json
from datetime import datetime
from collections import OrderedDict

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.base import Agent, OBS_FORMAT_ONE_HOT, OBS_FORMAT_2D, OBS_FORMAT_2D_FLAT
from gym_tictactoe.agents.random_agent import RandomAgent
from gym_tictactoe.agents.min_max_agent import MinMaxAgent

from utils.utils import get_alg, get_env, filter_tf_warnings
from utils.callbacks import PlotTestSaveCallback
from utils.cnn_extractor import tic_tac_toe_cnn


def train(alg, obs_format, env_agent: Agent, self_play: bool,
          train_episodes=10000, eval_freq=1000, player_one_char='-',
          gamma=1.0, net_arch=[256, 256, 512], rewards=TicTacToeEnv.DEFAULT_REWARDS,
          env_exploration_rate=0.0, n_envs=1, tensorboard_log=None):

    now = datetime.now()

    env_agent_name = env_agent.name

    if self_play:
        env_agent_name = "Self"

    if alg.__name__ == "DQN":
        n_envs = 1

    params = OrderedDict()
    params['alg'] = alg.__name__
    params['env_agent'] = env_agent_name
    params['train_episodes'] = train_episodes
    params['eval_freq'] = eval_freq
    params['obs_format'] = obs_format
    params['env_exploration_rate'] = env_exploration_rate
    params['n_envs'] = n_envs
    params['gamma'] = gamma
    params['net_arch'] = net_arch
    params['r_win'] = rewards[0]
    params['r_draw'] = rewards[1]
    params['r_still_playing'] = rewards[2]
    params['r_lose'] = rewards[3]
    params['r_invalid'] = rewards[4]
    params['player_one_char'] = player_one_char
    params['datetime'] = now.replace(microsecond=0).isoformat()

    net_arch_str = '-'.join([str(elem) for elem in net_arch])

    # rewards_str = '-'.join([str(elem) for elem in rewards])

    log_dir = "logs/{}_{}/{}_{}_{}_{}_{}_{}_{}".format(alg.__name__, env_agent_name,
                                                       now.strftime('%y%m%d-%H%M%S'),
                                                       train_episodes, obs_format,
                                                       env_exploration_rate, n_envs,
                                                       gamma, net_arch_str)
    os.makedirs(log_dir, exist_ok=True)
    print("\nLog dir:", log_dir)

    with open(log_dir + "/params.json", "w") as f:
        json.dump(params, f, indent=4)

    train_env = get_env(obs_format, env_agent, player_one_char, rewards,
                        env_exploration_rate, monitor=True, n_envs=n_envs)

    policy_network = "MlpPolicy"
    policy_kwargs = None

    if obs_format == OBS_FORMAT_2D or obs_format == OBS_FORMAT_2D_FLAT:
        policy_network = "CnnPolicy"
        policy_kwargs = {'cnn_extractor': tic_tac_toe_cnn, 'cnn_arch': net_arch}

    if alg.__name__ == "PPO2":

        if not policy_kwargs:
            policy_kwargs = {'net_arch': net_arch}

        model = alg(policy_network, train_env, gamma=gamma, policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log, verbose=0)

    elif alg.__name__ == "DQN":

        if not policy_kwargs:
            policy_kwargs = {'layers': net_arch}

        model = alg(policy_network, train_env, gamma=gamma, policy_kwargs=policy_kwargs,
                    prioritized_replay=True, tensorboard_log=tensorboard_log, verbose=0)

    max_train_timesteps = train_episodes * 9

    with PlotTestSaveCallback(train_episodes, eval_freq, log_dir, alg.__name__, self_play, train_env) as callback:
        model.learn(max_train_timesteps, callback=callback, log_interval=eval_freq)


def main():
    parser = argparse.ArgumentParser(
        description='Train a model, plot training and testing results, and save the model.')
    parser.add_argument('-a', '--alg', type=str, default='DQN',
                        help='Algorithm name. PPO2 or DQN (default: DQN)')
    parser.add_argument('-e', '--episodes', type=int, default=10000,
                        help='Training Episodes (default: 10000)')
    parser.add_argument('-f', '--freq', type=int, default=1000,
                        help='Evaluation Frequency (default: 1000)')
    parser.add_argument("-tb", "--tensorboard", default=None,
                        help="Tensorboard logdir. (default: None)")
    parser.add_argument('-p', '--player_one', type=str, default='-',
                        help='X for the agent, O for the environment, or - for randomly choosing in each train episode (default: -)')
    parser.add_argument('-g', '--gamma', type=float, default=1.0,
                        help='Gamma (default: 1.0)')
    parser.add_argument("-r", "--random_agent", action="store_true",
                        help='Train vs Random agent (default: Train vs Self)')
    parser.add_argument("-m", "--min_max", action="store_true",
                        help='Train vs MinMax agent (default: Train vs Self)')
    parser.add_argument("-o", "--one_hot", action="store_true",
                        help='Use one hot encoded observations (Mlp) (default: Use 2D observations (Cnn))')
    parser.add_argument('-n', '--n_envs', type=int, default=8,
                        help='Number of parallel environments when using PPO2 (default: 8)')
    args = parser.parse_args()

    alg = get_alg(args.alg)

    if not alg:
        print("Algorithm not found.")
        exit(1)

    env_agent = RandomAgent()

    self_play = True

    if args.random_agent:
        self_play = False

    if args.min_max:
        env_agent = MinMaxAgent()
        self_play = False

    obs_format = OBS_FORMAT_2D

    if args.one_hot:
        obs_format = OBS_FORMAT_ONE_HOT

    filter_tf_warnings()

    train(alg, obs_format, env_agent, self_play, train_episodes=args.episodes,
          eval_freq=args.freq, player_one_char=args.player_one, gamma=args.gamma,
          n_envs=args.n_envs, tensorboard_log=args.tensorboard)


if __name__ == "__main__":
    main()
