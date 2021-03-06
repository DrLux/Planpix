from parameters import Parameters
import os
import torch
from env import ControlSuiteEnv
from memory import ExperienceReplay
from trainer import Trainer
from tqdm import tqdm
import random
import numpy as np


class Initializer():
  def __init__(self):  
      self.parms = Parameters()
      self.results_dir = os.path.join(self.parms.results_path)
      self.dataset_path = os.path.join(self.parms.results_path,'dataset/')  
      os.makedirs(self.dataset_path, exist_ok=True) 
      self.metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'predicted_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'regularizer_loss': []}    
      os.makedirs(self.results_dir, exist_ok=True) 
      
      ## Setting cuda options 
      if torch.cuda.is_available() and self.parms.use_cuda:
        self.parms.device = torch.device('cuda')
        torch.cuda.set_device(self.parms.gpu_id)
        print("Using gpu: ", torch.cuda.current_device())
      else:
        self.parms.device = torch.device('cpu')
        self.use_cuda = False
        print("Work on: ", self.parms.device)
      
      # Initilize buffer experience replay
      self.env = ControlSuiteEnv(self.parms.env_name, self.parms.seed, self.parms.max_episode_length, self.parms.bit_depth)
      self.D = ExperienceReplay(self.parms.ex_replay_buff_size, self.env.observation_size, self.env.action_size, self.parms.bit_depth, self.parms.device)
      
      if self.parms.seed > 0: 
        self.set_seed()

      self.trainer = Trainer(self.parms,self.D,self.metrics,self.results_dir,self.env)
      self.init_exp_rep()
      
      
      # Start Training
      print("Total training episodes: ", self.parms.training_episodes, " Buffer sampling: ", self.parms.collect_interval)
      self.trainer.train_models()
      print("END.")
      


  def set_seed(self):
    print("Setting seed")
    os.environ['PYTHONHASHSEED']=str(self.parms.seed)
    random.seed(self.parms.seed)
    np.random.seed(self.parms.seed)
    torch.manual_seed(self.parms.seed)   
    if self.parms.use_cuda:
      torch.cuda.manual_seed(self.parms.seed)
      #torch.backends.cudnn.enabled=False      # This makes the training slower 
      #torch.backends.cudnn.deterministic=True # This makes the training slower
        

  # Init buffer experience replay
  def init_exp_rep(self):
    print("Starting initialization buffer.")
    for s in tqdm(range(1, self.parms.num_init_episodes +1)):
      observation, done, t = self.env.reset(), False, 0
      while not done:
        action = self.env.sample_random_action()
        next_observation, reward, done = self.env.step(action)
        self.D.append(observation, action, reward, done)
        observation = next_observation
        t += 1
      self.metrics['steps'].append(t * self.env.action_repeat + (0 if len(self.metrics['steps']) == 0 else self.metrics['steps'][-1]))
      self.metrics['episodes'].append(s)     

if __name__ == "__main__":
  Initializer()