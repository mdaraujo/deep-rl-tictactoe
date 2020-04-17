import json
import numpy as np

from gym_tictactoe.agents.base import Agent

from utils.utils import get_alg


class RLAgent(Agent):

    def __init__(self, log_dir, model=None, deterministic=True, model_suffix=""):

        self.log_dir = log_dir
        self.deterministic = deterministic

        with open(self.log_dir + "/params.json", "r") as f:
            params = json.load(f)

            super().__init__(params["alg"], params["obs_format"])

            self.alg = get_alg(self.name)

            self._rewards = [params["r_win"],
                             params["r_draw"],
                             params["r_still_playing"],
                             params["r_lose"],
                             params["r_invalid"]]

            if not self.alg:
                print("Algorithm not found.")
                exit(1)

            if model:
                self.model = model
            else:
                self.model = self.alg.load(self.log_dir + "/" + self.name + model_suffix)

    def play(self, obs):

        if self.deterministic:

            action_max = np.argmax(self.model.action_probability(obs))

            action, _states = self.model.predict(obs, deterministic=True)

            if action != action_max:
                print("DIFF action_max")
                print("action_max:", action_max)
                print("action:", action)

                with np.printoptions(precision=3, suppress=True):
                    print("action_probability:", self.model.action_probability(obs))

                print()

            # if obs[action] == 0:
            #     print("INVALID")
            #     print("obs:", obs)
            #     print("action_max:", action_max)
            #     print("action:", action)

            #     with np.printoptions(precision=3, suppress=True):
            #         print("action_probability:", self.model.action_probability(obs))
            #     print()
        else:
            action, _states = self.model.predict(obs)

        return action
