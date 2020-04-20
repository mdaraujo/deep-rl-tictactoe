from abc import ABC, abstractmethod

OBS_FORMAT_RAW = "RAW"
OBS_FORMAT_ONE_HOT = "ONE_HOT"
OBS_FORMAT_2D = "2D"


class Agent(ABC):

    def __init__(self, name, obs_format):
        self._name = name
        self._obs_format = obs_format
        self._rewards = None
        self._board_states = {}

    @property
    def name(self):
        return self._name

    @property
    def obs_format(self):
        return self._obs_format

    @abstractmethod
    def play(self, obs):
        if str(obs) in self._board_states:
            self._board_states[str(obs)] += 1
        else:
            self._board_states[str(obs)] = 1
        pass

    @property
    def rewards(self):
        return self._rewards

    @property
    def board_states(self):
        return self._board_states
