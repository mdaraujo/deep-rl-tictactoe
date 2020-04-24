# Deep Reinforcement Learning Tic-Tac-Toe

Deep Reinforcement Learning algorithms from Stable Baselines learn to play Tic-Tac-Toe in a custom Gym environment.

The training agent always plays with X and the environment agent plays with O. The first player can be chosen, but by default it will be random, so the agent can learn to play in both player one and player two positions.

The environment agent can be Random, MinMax, or the Self RL agent in which case the board will be symmetrically changed for X so the environment agent can play properly.

The observation input can be the Row 9 positions array (Mlp), the One-hot encoded version (Mlp), or the 2D version (Cnn).

Any algorithm from Stable Baselines can be used, but currently the code only supports DQN and PPO2.

The agent is evaluated vs the Random agent in both player one and player two positions, as well as evaluated against MinMax agent in both positions. We also test the agent against itself. 

The goal is the RL agent can be perfect vs MinMax and still beat the Random player with the highest win rate possible.


## Prerequisites

    Python >= 3.5
    Gym
    Stabe-Baselines

## Gym TicTacToe Installation

    cd gym-tictactoe/
    pip install -e .
    

## Usage

### Train

Train a model, plot training results, plot test results versus random and minmax agents during training, and save the best model.

There are other parameters like network architecture, environment exploration rate, and training rewards that can not be passed as a program argument, but can be used calling the train function as shown in train_*.py scripts.

        usage: train.py [-h] [-a ALG] [-e EPISODES] [-f FREQ] [-p PLAYER_ONE]
                        [-g GAMMA] [-r] [-m] [-o] [-n N_ENVS]

        Train a model, plot training and testing results, and save the model.

        optional arguments:
          -h, --help            show this help message and exit
          -a ALG, --alg ALG     Algorithm name. PPO2 or DQN (default: DQN)
          -e EPISODES, --episodes EPISODES
                                Training Episodes (default: 10000)
          -f FREQ, --freq FREQ  Evaluation Frequency (default: 1000)
          -p PLAYER_ONE, --player_one PLAYER_ONE
                                X for the agent, O for the environment, or - for
                                randomly choosing in each train episode (default: -)
          -g GAMMA, --gamma GAMMA
                                Gamma (default: 1.0)
          -r, --random_agent    Train vs Random agent (default: Train vs Self)
          -m, --min_max         Train vs MinMax agent (default: Train vs Self)
          -o, --one_hot         Use one hot encoded observations (Mlp) (default: Use
                                2D observations (Cnn))
          -n N_ENVS, --n_envs N_ENVS
                                Number of parallel environments when using PPO2
                                (default: 8)
   
#### Example training PPO vs Random agent

```
python train.py -e 50000 -f 5000 -a PPO2 -r
```
                                
### Test

        usage: test.py [-h] [-a] [-l] [-e EPISODES] [-v] [-s MODEL_SUFFIX] logdir

        Test a model inside a directory or a set of models.

        positional arguments:
          logdir                log directory

        optional arguments:
          -h, --help            show this help message and exit
          -a, --all             Run for all subdirs of 'logdir' (default: run for
                                'logdir')
          -l, --latest          Use latest dir inside 'logdir' (default: run for
                                'logdir')
          -e EPISODES, --episodes EPISODES
                                Number of test episodes (default: 2500)
          -v, --verbose         Print game boards (default: Don't print)
          -s MODEL_SUFFIX, --model_suffix MODEL_SUFFIX
                                Use a suffix for the model file e.g _best (default: )

#### Example testing PPO best model in the last directory for 1000 episodes

```
python test.py logs/PPO2_Random/ -l -e 1000 -s _best
