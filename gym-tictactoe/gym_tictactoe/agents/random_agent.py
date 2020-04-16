import random

from gym_tictactoe.agents.base import Agent, OBS_FORMAT_RAW


class RandomAgent(Agent):

    def __init__(self):
        super().__init__("Random", OBS_FORMAT_RAW)

    def play(self, obs):
        available_actions = []

        for i in range(9):
            if obs[i] == 0:
                available_actions.append(i)

        return random.choice(available_actions)
