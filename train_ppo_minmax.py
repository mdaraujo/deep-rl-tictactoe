from stable_baselines import PPO2

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.base import OBS_FORMAT_2D
from gym_tictactoe.agents.min_max_agent import MinMaxAgent
from gym_tictactoe.agents.random_agent import RandomAgent

from utils.hyperparams import P_CHAR, REWARDS, N_REPEATS
from utils.utils import filter_tf_warnings

from train import train


filter_tf_warnings()

ALG = PPO2

ENV_AGENT = MinMaxAgent()

OBS_FORMAT = OBS_FORMAT_2D

SELF_PLAY = False

TRAIN_EPISODES = 500000

EVAL_FREQ = 50000

# invalid_rewards = [2, 1, 0, -1, -10]

P_CHAR = '-'

GAMMA = [1.0, 0.99]

ENV_EXP = [0.2, 0.5]

N_ENVS = 8

NET_ARCH = [[256, 128, 256], [512, 128, 256]]

total_trainings = len(REWARDS) * len(GAMMA) * len(ENV_EXP) * len(NET_ARCH) * N_REPEATS
count = 0

for gamma in GAMMA:
    for rewards in REWARDS:
        for env_exp in ENV_EXP:
            for net_arch in NET_ARCH:
                for _ in range(N_REPEATS):

                    count += 1
                    print("\n\nTraining {} / {}".format(count, total_trainings))

                    train(ALG, OBS_FORMAT, ENV_AGENT, SELF_PLAY, TRAIN_EPISODES,
                          EVAL_FREQ, P_CHAR, gamma, net_arch, rewards, env_exp, N_ENVS)
