from stable_baselines import DQN

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.base import OBS_FORMAT_2D
from gym_tictactoe.agents.min_max_agent import MinMaxAgent
from gym_tictactoe.agents.random_agent import RandomAgent

from utils.hyperparams import P_CHAR, REWARDS
from utils.utils import filter_tf_warnings

from train import train


filter_tf_warnings()

ALG = DQN

ENV_AGENT = RandomAgent()

OBS_FORMAT = OBS_FORMAT_2D

SELF_PLAY = True

TRAIN_EPISODES = 30000

EVAL_FREQ = [3000, 1500]

P_CHAR = '-'

GAMMA = [1.0]

ENV_EXP = [0.2]

N_ENVS = 1

N_REPEATS = 3

NET_ARCH = [[256, 256, 512], [512, 128, 256]]

total_trainings = len(REWARDS) * len(GAMMA) * len(ENV_EXP) * len(NET_ARCH) * len(EVAL_FREQ) * N_REPEATS
count = 0

for eval_freq in EVAL_FREQ:
    for gamma in GAMMA:
        for rewards in REWARDS:
            for env_exp in ENV_EXP:
                for net_arch in NET_ARCH:
                    for _ in range(N_REPEATS):

                        count += 1
                        print("\n\nTraining {} / {}".format(count, total_trainings))

                        train(ALG, OBS_FORMAT, ENV_AGENT, SELF_PLAY, TRAIN_EPISODES,
                              eval_freq, P_CHAR, gamma, net_arch, rewards, env_exp, N_ENVS)
