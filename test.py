from env import ControlSuiteEnv,EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
import numpy as np
import torch
from torch import nn, optim
import os
from planner import MPCPlanner
import random
from torch.distributions import Normal
from math import inf
from tqdm import tqdm
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import lineplot, write_video
from torchvision.utils import make_grid, save_image
from env import postprocess_observation
from initializer import Initializer

params = Parameters()
init = Initializer(params)
env = ControlSuiteEnv(params.env_name, params.seed, params.max_episode_length, params.bit_depth)
D = ExperienceReplay(params.ex_replay_buff_size, env.observation_size, env.action_size, params.bit_depth, params.device)
init.init_exp_rep(D)

'''
### ENV ####
env_name = 'reacher-easy'
seed = 1
max_episode_length = 1000
bit_depth = 5
results_dir = os.path.join('/home/luca/Desktop/luca/agosto/experience/')
######

def run_episode(env):
    observation, done, t, total_reward = env.reset(), False, 0, 0
    frames = []
    while not done:
        action = env.sample_random_action()
        next_observation, reward, done = env.step(action)
        observation = next_observation
        t += 1
        total_reward += reward
        prova = np.clip(np.floor((np.array(observation) + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)
        frames.append(make_grid(torch.cat([observation, torch.from_numpy(prova).float()], dim=3) + 0.5, nrow=5).numpy())  # Decentre

    #save_image(torch.as_tensor(frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % t))
    print("total_reward: ", total_reward)
    print("dumping video!")
    write_video(frames, 'dump_video', results_dir)

metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': []}
env = ControlSuiteEnv(env_name, seed, max_episode_length, bit_depth)
run_episode(env)
'''