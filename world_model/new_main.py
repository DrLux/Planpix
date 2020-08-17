from env import GymEnv
import torch

env_name = 'BipedalWalker-v3'
seed = 12
max_episode_length = 100

env = GymEnv(env_name, seed, max_episode_length)

print(env.reset())
        