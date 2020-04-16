import os
import warnings
import logging

from stable_baselines import DQN

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.base import OBS_FORMAT_2D
from gym_tictactoe.agents.min_max_agent import MinMaxAgent
from gym_tictactoe.agents.random_agent import RandomAgent

from utils.hyperparams import P_CHAR, NET_ARCH, REWARDS

from train import train

# Filter tensorflow version warnings
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

logging.getLogger("tensorflow").setLevel(logging.ERROR)


ALG = DQN

ENV_AGENT = MinMaxAgent()

OBS_FORMAT = OBS_FORMAT_2D

SELF_PLAY = False

TRAIN_EPISODES = 20000

EVAL_FREQ = 1000

P_CHAR = '-'

GAMMA = [1.0]

ENV_EXP = [0.2, 0.5]

N_ENVS = 1

N_REPEATS = 5

total_trainings = len(REWARDS) * len(GAMMA) * len(ENV_EXP) * len(NET_ARCH) * N_REPEATS
count = 0

for rewards in REWARDS:
    for gamma in GAMMA:
        for env_exp in ENV_EXP:
            for net_arch in NET_ARCH:
                for _ in range(N_REPEATS):

                    count += 1
                    print("\n\nTraining {} / {}".format(count, total_trainings))

                    train(ALG, OBS_FORMAT, ENV_AGENT, SELF_PLAY, TRAIN_EPISODES,
                          EVAL_FREQ, P_CHAR, gamma, net_arch, rewards, env_exp, N_ENVS)
