from stable_baselines import DQN

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.min_max_agent import MinMaxAgent
from gym_tictactoe.agents.random_agent import RandomAgent

from utils.hyperparams import P_CHAR, REWARDS, N_REPEATS, ENV_EXP, NET_ARCH, OBS_FORMAT
from utils.utils import filter_tf_warnings

from train import train


filter_tf_warnings()

ALG = DQN

ENV_AGENT = MinMaxAgent()

SELF_PLAY = False

TRAIN_EPISODES = 10000

EVAL_FREQ = [1000]

P_CHAR = '-'

GAMMA = [1.0]

# ENV_EXP = [0.2]

N_ENVS = 1

total_trainings = len(OBS_FORMAT) * len(REWARDS) * len(GAMMA) * \
    len(ENV_EXP) * len(NET_ARCH) * len(EVAL_FREQ) * N_REPEATS

count = 0

for eval_freq in EVAL_FREQ:
    for obs_format in OBS_FORMAT:
        for gamma in GAMMA:
            for rewards in REWARDS:
                for env_exp in ENV_EXP:
                    for net_arch in NET_ARCH:
                        for _ in range(N_REPEATS):

                            count += 1
                            print("\n\nTraining {} / {}".format(count, total_trainings))

                            train(ALG, obs_format, ENV_AGENT, SELF_PLAY, TRAIN_EPISODES,
                                  eval_freq, P_CHAR, gamma, net_arch, rewards, env_exp, N_ENVS)
