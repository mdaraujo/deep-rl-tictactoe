import gym
import warnings
import os
import datetime

from stable_baselines import PPO2, DQN
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv, ObsRawToOneHot, ObsRawTo2D
from gym_tictactoe.agents.base import Agent, OBS_FORMAT_ONE_HOT, OBS_FORMAT_2D

FIG_SIZE = (12, 5)

def get_alg(alg_name):

    if alg_name == "PPO2":
        return PPO2
    elif alg_name == "DQN":
        return DQN

    return None


def get_env(obs_format, env_agent: Agent, player_one_char, rewards=None, env_exploration_rate=0.0, monitor=False, n_envs=1):

    env_kwargs = dict(naught_agent=env_agent, player_one_char=player_one_char,
                      env_exploration_rate=env_exploration_rate)

    if rewards:
        env_kwargs['rewards'] = rewards

    env = TicTacToeEnv(**env_kwargs)

    if obs_format == OBS_FORMAT_ONE_HOT:
        wrapper_class = ObsRawToOneHot
        env = wrapper_class(env)
    elif obs_format == OBS_FORMAT_2D:
        wrapper_class = ObsRawTo2D
        env = wrapper_class(env)

    if monitor:
        if n_envs > 1:
            env = make_vec_env(TicTacToeEnv, n_envs=n_envs, vec_env_cls=DummyVecEnv,
                               wrapper_class=wrapper_class, env_kwargs=env_kwargs)
        else:
            env = Monitor(env, filename=None, allow_early_resets=False, info_keywords=('outcome', 'player_one'))

    return env


def make_vec_env(env_id, n_envs=1, seed=None, start_index=0,
                 monitor_dir=None, wrapper_class=None,
                 env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
    """
    Create a wrapped, monitored `VecEnv`.
    By default it uses a `DummyVecEnv` which is usually faster
    than a `SubprocVecEnv`.
    (Adapted from StableBaselines)

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id)
                if len(env_kwargs) > 0:
                    warnings.warn("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, allow_early_resets=False,
                          info_keywords=('outcome', 'player_one'))

            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def get_elapsed_time(current_time, start_time):
    elapsed_time_seconds = current_time - start_time
    elapsed_time_h = datetime.timedelta(seconds=elapsed_time_seconds)
    elapsed_time_h = str(datetime.timedelta(days=elapsed_time_h.days, seconds=elapsed_time_h.seconds))
    return elapsed_time_seconds, elapsed_time_h
