from parameters import Parameters
import os
import torch
from env import ControlSuiteEnv
from memory import ExperienceReplay
from trainer import Trainer
from tqdm import tqdm


class Initializer():
  def __init__(self):  
      self.parms = Parameters()
      self.results_dir = os.path.join(self.parms.results_path)
      self.metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': []}
      

      os.makedirs(self.results_dir, exist_ok=True) 
      
      ## set cuda 
      if torch.cuda.is_available() and self.parms.use_cuda:
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"]= str(self.parms.gpu_id)
        #export CUDA_VISIBLE_DEVICES=0
        torch.cuda.set_device(self.parms.gpu_id)
        self.parms.device = torch.device('cuda')
        torch.cuda.manual_seed(self.parms.seed)
        print("Using gpu: ", torch.cuda.current_device())
      else:
        self.parms.device = torch.device('cpu')
        print("Work on: ", self.parms.device)
      
      # Initilize buffer experience replay
      self.env = ControlSuiteEnv(self.parms.env_name, self.parms.seed, self.parms.max_episode_length, self.parms.bit_depth)
      self.D = ExperienceReplay(self.parms.ex_replay_buff_size, self.env.observation_size, self.env.action_size, self.parms.bit_depth, self.parms.device)
      print("Starting initialization buffer.")
      self.init_exp_rep()
      
      self.trainer = Trainer(self.parms,self.D,self.metrics,self.results_dir,self.env)
      
      # Load checkpoints
      #self.trainer.load_checkpoints()
      print("Total training episodes: ", self.parms.training_episodes, " Buffer sampling: ", self.parms.collect_interval)
      self.trainer.train_models()
      #self.trainer.test_model()
      self.env.close()
      print("END.")


  def set_seed(self):
      os.environ['PYTHONHASHSEED']=str(self.parms.seed)
      random.seed(self.parms.seed)
      np.random.seed(self.parms.seed)
      torch.manual_seed(self.parms.seed)   

  def init_exp_rep(self):
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
            
Initializer()