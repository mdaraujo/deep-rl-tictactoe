from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.base import OBS_FORMAT_2D, OBS_FORMAT_ONE_HOT, OBS_FORMAT_RAW
from stable_baselines import PPO2, DQN

ALG = [PPO2, DQN]

SELF_PLAY = [True, False]

base_rewards = [1, 0, 0, -1, -2]

REWARDS = [TicTacToeEnv.DEFAULT_REWARDS, base_rewards]

OBS_FORMAT = [OBS_FORMAT_2D, OBS_FORMAT_ONE_HOT, OBS_FORMAT_RAW]

P_CHAR = '-'

GAMMA = [0.99, 1.0]

ENV_EXP = [0.0, 0.2, 0.5]

# NET_ARCH = [[512, 512], [256, 256, 512], [512, 128, 256], [256, 256, 128, 256]]

N_ENVS = [1, 4, 8, 16]

N_REPEATS = 5
