import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines.common.env_checker import check_env

from gym_tictactoe.agents.base import OBS_FORMAT_ONE_HOT, OBS_FORMAT_2D
from gym_tictactoe.agents.random_agent import RandomAgent


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    BOARD_LENGTH = 3
    SIZE = BOARD_LENGTH**2

    # Rewards
    DEFAULT_REWARDS = [2, 1, 0, -1, -2]

    # States and Outcomes
    EMPTY = 0
    CROSS = 1
    NAUGHT = 2
    DRAW = 3
    INVALID = 4
    STILL_PLAYING = 5

    STATE_CHARS = ['-', 'X', 'O']

    def __init__(self, naught_agent, rewards=DEFAULT_REWARDS, player_one_char='-', env_exploration_rate=0.0):

        # print("TicTacToeEnv:", self)
        # print("TicTacToeEnv id:", hex(id(self)))

        self.naught_agent = naught_agent

        if player_one_char not in self.STATE_CHARS:
            raise ValueError(
                "Received invalid player_one_char={} which is not part of the possible player chars.\nUse 'X' for the agent, 'O' for the environemnt, or use '-' for choosing the first player randomly in every reset.".format(player_one_char))

        self.player_one = self.STATE_CHARS.index(player_one_char)

        self.current_player_one = self.player_one

        self.board = np.zeros((self.SIZE,), dtype=np.int)

        self.action_space = spaces.Discrete(self.SIZE)

        self.observation_space = spaces.Box(low=self.EMPTY, high=self.NAUGHT,
                                            shape=(self.SIZE,), dtype=np.int)

        self.env_exploration_rate = env_exploration_rate

        # Rewards
        self.win_reward = rewards[0]
        self.draw_reward = rewards[1]
        self.still_playing_reward = rewards[2]
        self.loss_reward = rewards[3]
        self.invalid_reward = rewards[4]

        # Self play flag
        self.self_play = False

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Received invalid action={} which is not part of the action space.".format(action))

        done = False
        reward = self.still_playing_reward
        info = {'player_one': self.STATE_CHARS[self.current_player_one]}

        if self.board[action] != self.EMPTY:
            # raise ValueError("Received invalid action={}. Position already occupied by {} player.".format(
            #     action, self._position_char(action)))
            done = True
            reward = self.invalid_reward
            info['outcome'] = self.INVALID

            return self.board, reward, done, info

        # Play received CROSS action
        self.board[action] = self.CROSS

        winner = self.check_winner(self.board)

        if winner == self.CROSS:
            done = True
            reward = self.win_reward
            info['outcome'] = self.CROSS
        elif winner == self.DRAW:
            done = True
            reward = self.draw_reward
            info['outcome'] = self.DRAW

        # Play for NAUGHT player
        if not done:
            naught_action = self._naught_action()

            if self.board[naught_action] != self.EMPTY:
                print("Invalid")
                self.render()

            self.board[naught_action] = self.NAUGHT

            info['naught_action'] = naught_action

            winner = self.check_winner(self.board)

            if winner == self.NAUGHT:
                done = True
                reward = self.loss_reward
                info['outcome'] = self.NAUGHT
            elif winner == self.DRAW:
                done = True
                reward = self.draw_reward
                info['outcome'] = self.DRAW

        return self.board, reward, done, info

    def _naught_action(self):
        """
        Get NAUGHT agent action (env action)
        """
        if np.random.uniform(0.0, 1.0) < self.env_exploration_rate:
            return self.random_available_action()

        board_copy = np.copy(self.board)

        if self.self_play:
            for i in range(self.SIZE):
                if board_copy[i] == self.CROSS:
                    board_copy[i] = self.NAUGHT
                elif board_copy[i] == self.NAUGHT:
                    board_copy[i] = self.CROSS

            if self.naught_agent.obs_format == OBS_FORMAT_ONE_HOT:
                board_copy = ObsRawToOneHot.get_one_hot_obs(board_copy)
            elif self.naught_agent.obs_format == OBS_FORMAT_2D:
                board_copy = ObsRawTo2D.get_2d_obs(board_copy)

        action = self.naught_agent.play(board_copy)

        if self.board[action] != self.EMPTY:
            return self.random_available_action()

        return action

    def reset(self):
        self.board = np.zeros((self.SIZE,), dtype=int)

        self.current_player_one = self.player_one

        if self.current_player_one == self.EMPTY:

            players = [self.CROSS, self.NAUGHT]

            self.current_player_one = random.choice(players)

        if self.current_player_one == self.NAUGHT:
            # Play for NAUGHT player
            self.board[self._naught_action()] = self.NAUGHT

        # if np.random.random_sample() > 0.97:
        #     print("-- env naught_agent id:", hex(id(self.naught_agent)))

        return self.board

    def render(self, mode='human'):
        for y in range(self.BOARD_LENGTH):
            for x in range(self.BOARD_LENGTH):
                print(self._position_char(y*self.BOARD_LENGTH+x), end=' ')
            print()

    def _position_char(self, position):
        return self.STATE_CHARS[self.board[position]]

    def available_actions(self):
        return np.ravel(np.nonzero(self.board == self.EMPTY))

    def random_available_action(self):
        return np.random.choice(self.available_actions())

    @staticmethod
    def check_winner(board) -> int:

        for player in [TicTacToeEnv.CROSS, TicTacToeEnv.NAUGHT]:

            # check rows
            for j in range(0, 9, 3):
                if [player] * 3 == [board[i] for i in range(j, j+3)]:
                    return player
            # check columns
            for j in range(0, 3):
                if board[j] == player and board[j+3] == player and board[j+6] == player:
                    return player
            # diagonal left to right
            if board[0] == player and board[4] == player and board[8] == player:
                return player
            # diagonal right to left
            if board[2] == player and board[4] == player and board[6] == player:
                return player

        # still playing
        for i in range(TicTacToeEnv.SIZE):
            if board[i] == TicTacToeEnv.EMPTY:
                return TicTacToeEnv.STILL_PLAYING

        # draw
        return TicTacToeEnv.DRAW


class ObsRawToOneHot(gym.ObservationWrapper):

    N = TicTacToeEnv.SIZE * len(TicTacToeEnv.STATE_CHARS)

    def __init__(self, env):
        super().__init__(env)

        observation_space = env.observation_space

        assert isinstance(
            observation_space, gym.spaces.Box), "This wrapper only works with continuous observation space (spaces.Box)"

        self.observation_space = gym.spaces.Box(0, 1, (self.N,), dtype=np.int)

    def observation(self, obs):
        return self.get_one_hot_obs(obs)

    @staticmethod
    def get_one_hot_obs(obs):
        new_obs = np.zeros(ObsRawToOneHot.N, dtype=np.int)
        for i, state in enumerate(obs):
            new_obs[i + (state * TicTacToeEnv.SIZE)] = 1
        return new_obs


class ObsRawTo2D(gym.ObservationWrapper):

    SHAPE = (TicTacToeEnv.BOARD_LENGTH,
             TicTacToeEnv.BOARD_LENGTH,
             len(TicTacToeEnv.STATE_CHARS))

    def __init__(self, env):
        super().__init__(env)

        observation_space = env.observation_space

        assert isinstance(
            observation_space, gym.spaces.Box), "This wrapper only works with continuous observation space (spaces.Box)"

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=ObsRawTo2D.SHAPE,
                                            dtype=np.uint8)

    def observation(self, obs):
        return self.get_2d_obs(obs)

    @staticmethod
    def get_2d_obs(obs):
        new_obs = np.zeros(ObsRawTo2D.SHAPE, dtype=np.int)

        for i, state in enumerate(obs):
            y = int(i / TicTacToeEnv.BOARD_LENGTH)
            x = i - y * TicTacToeEnv.BOARD_LENGTH
            new_obs[state][y][x] = 255

        return new_obs


if __name__ == "__main__":
    env = TicTacToeEnv(RandomAgent(), player_one_char='O')
    env = ObsRawTo2D(env)
    print("Checking environment...")
    check_env(env, warn=True)

    print("Observation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    print("Observation space high:", env.observation_space.high)
    print("Observation space low:", env.observation_space.low)

    print("Action space:", env.action_space)

    obs = env.reset()
    done = False

    while not done:
        print("Available actions:", env.available_actions())

        action = env.random_available_action()
        print("Sampled action:", action)
        obs, reward, done, info = env.step(action)

        naught_action = -1
        if 'naught_action' in info:
            naught_action = info['naught_action']

        print("CROSS: {:2d} | NAUGHT: {:2d} | Reward: {:.0f} | Done: {}".format(
            action, naught_action, reward, done))

        print(obs)

        env.render()
        print()
