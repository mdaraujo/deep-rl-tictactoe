from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.base import OBS_FORMAT_RAW, OBS_FORMAT_ONE_HOT, OBS_FORMAT_2D, OBS_FORMAT_2D_FLAT
from stable_baselines import PPO2, DQN

ALG = [PPO2, DQN]

SELF_PLAY = [True, False]

# base_rewards = [1, 0, 0, -1, -2]

base_rewards = [2, 1, 0, -2, -3]

REWARDS = [TicTacToeEnv.DEFAULT_REWARDS]

OBS_FORMAT = [OBS_FORMAT_2D, OBS_FORMAT_2D_FLAT, OBS_FORMAT_ONE_HOT, OBS_FORMAT_RAW]

P_CHAR = '-'

GAMMA = [0.99]

ENV_EXP = [0.2]

# ENV_EXP = [0.0, 0.2, 0.5, 1.0]

NET_ARCH = [[64, 128]]

# NET_ARCH = [[512, 512], [256, 256, 512], [512, 128, 256], [256, 256, 128, 256]]

FILTER_SIZES = [3]

PADS = ['SAME']

N_ENVS = [1, 4, 8, 16]

N_REPEATS = 5
