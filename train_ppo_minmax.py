from stable_baselines import PPO2

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv
from gym_tictactoe.agents.min_max_agent import MinMaxAgent
from gym_tictactoe.agents.random_agent import RandomAgent

from utils.hyperparams import P_CHAR, REWARDS, N_REPEATS, ENV_EXP, NET_ARCH, OBS_FORMAT, FILTER_SIZES, PADS
from utils.utils import filter_tf_warnings

from train import train


filter_tf_warnings()

ALG = PPO2

ENV_AGENT = MinMaxAgent()

SELF_PLAY = False

TRAIN_EPISODES = 500000

EVAL_FREQ = [100000]

# invalid_rewards = [2, 1, 0, -1, -10]

P_CHAR = '-'

GAMMA = [0.99]

# ENV_EXP = [0.2]

N_ENVS = 8

total_trainings = len(OBS_FORMAT) * len(REWARDS) * len(GAMMA) * \
    len(ENV_EXP) * len(NET_ARCH) * len(FILTER_SIZES) * len(PADS) * \
    len(EVAL_FREQ) * N_REPEATS

count = 0

for obs_format in OBS_FORMAT:
    for eval_freq in EVAL_FREQ:
        for gamma in GAMMA:
            for rewards in REWARDS:
                for env_exp in ENV_EXP:
                    for net_arch in NET_ARCH:
                        for filter_size in FILTER_SIZES:
                            for pad in PADS:
                                for _ in range(N_REPEATS):

                                    count += 1
                                    print("\n\nTraining {} / {}".format(count, total_trainings))

                                    train(ALG, obs_format, ENV_AGENT, SELF_PLAY, TRAIN_EPISODES,
                                          eval_freq, P_CHAR, gamma, net_arch, filter_size, pad,
                                          rewards, env_exp, N_ENVS)
