#
# This MinMax implementation was adapted from https://github.com/fcarsten/tic-tac-toe/ RndMinMaxAgent
#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

import random
import numpy as np

from gym_tictactoe.agents.base import Agent, OBS_FORMAT_RAW
from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv


class MinMaxAgent(Agent):

    WIN_VALUE = 1
    DRAW_VALUE = 0
    LOSS_VALUE = -1

    def __init__(self, is_deterministic=False, side=TicTacToeEnv.NAUGHT):
        self.is_deterministic = is_deterministic

        if self.is_deterministic:
            super().__init__("MinMaxDet", OBS_FORMAT_RAW)  # Not implemented
        else:
            super().__init__("MinMax", OBS_FORMAT_RAW)

        self.side = side

        if self.side == TicTacToeEnv.NAUGHT:
            self.other_side = TicTacToeEnv.CROSS
        elif self.side == TicTacToeEnv.CROSS:
            self.other_side = TicTacToeEnv.NAUGHT

        self.first_player = None

        self.cache = {}

    def _min(self, board: np.array):
        """
        Evaluate the board position `board` from the Minimizing player's point of view.

        :param board: The board position to evaluate
        :return: Tuple of (Best Result, Best Move in this situation). Returns -1 for best move if the game has already
        finished
        """

        #
        # First we check if we have seen this board position before, and if yes just return a random choice
        # from the cached values
        #
        board_hash = self.board_hash(board)

        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        #
        # If the game has already finished we return. Otherwise we look at possible continuations
        #
        winner = TicTacToeEnv.check_winner(board)
        if winner == self.side:
            best_moves = [(self.WIN_VALUE, -1)]
        elif winner == self.other_side:
            best_moves = [(self.LOSS_VALUE, -1)]
        else:
            #
            # Init the min value as well as action. Min value is set to DRAW as this value will pass through in case
            # of a draw
            #
            min_value = self.DRAW_VALUE
            action = -1
            best_moves = [(min_value, action)]
            for index in [i for i in range(len(board)) if board[i] == TicTacToeEnv.EMPTY]:
                b = np.copy(board)
                b[index] = self.other_side

                res, _ = self._max(b)
                if res < min_value or action == -1:
                    min_value = res
                    action = index
                    best_moves = [(min_value, action)]

                    # # Shortcut: Can't get better than that, so abort here and return this move
                    # if min_value == self.LOSS_VALUE:
                    #     break

                elif res == min_value:
                    action = index
                    best_moves.append((min_value, action))

        self.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def _max(self, board: np.array):
        """
        Evaluate the board position `board` from the Maximizing player's point of view.

        :param board: The board position to evaluate
        :return: Tuple of (Best Result, Best Move in this situation). Returns -1 for best move if the game has already
        finished
        """

        #
        # First we check if we have seen this board position before, and if yes just return a random choice
        # from the cached values
        #
        board_hash = self.board_hash(board)

        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        #
        # If the game has already finished we return. Otherwise we look at possible continuations
        #
        winner = TicTacToeEnv.check_winner(board)
        if winner == self.side:
            best_moves = [(self.WIN_VALUE, -1)]
        elif winner == self.other_side:
            best_moves = [(self.LOSS_VALUE, -1)]
        else:
            #
            # Init the min value as well as action. Min value is set to DRAW as this value will pass through in case
            # of a draw
            #
            max_value = self.DRAW_VALUE
            action = -1
            best_moves = [(max_value, action)]
            for index in [i for i in range(len(board)) if board[i] == TicTacToeEnv.EMPTY]:
                b = np.copy(board)
                b[index] = self.side

                res, _ = self._min(b)
                if res > max_value or action == -1:
                    max_value = res
                    action = index
                    best_moves = [(max_value, action)]

                    # # Shortcut: Can't get better than that, so abort here and return this move
                    # if max_value == self.WIN_VALUE:
                    #     print("win: best moves:", best_moves)
                    #     break

                elif res == max_value:
                    action = index
                    best_moves.append((max_value, action))

        self.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def play(self, obs):

        super().play(obs)

        if np.sum(obs) == 0:
            self.first_player = self.side
        elif np.sum(obs) == self.other_side:
            self.first_player = self.other_side

        # board_hash = self.board_hash(obs)

        # if board_hash not in self.cache:
        #     print("NEW OBS")

        # print("obs:", obs)

        score, action = self._max(obs)

        # print("score:", score)
        # print("action:", action)

        # print("board_hash:", board_hash)
        # print("best_moves:", self.cache[board_hash])

        # print()
        return action

    def board_hash(self, board):
        return hash(str(board) + str(self.first_player))
