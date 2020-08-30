import os

class Parameters():
    def __init__(self):

        self.env_name = 'cheetah-run'
        self.max_episode_length = 1000
        self.max_iters = 1000
        self.buffer_replay_size = 1000000

        self.seed = 2
        self.use_cuda = False
        self.device = None
        self.gpu_id = 0
        self.checkpoint_interval = 100
        self.test_interval = 100

        self.gamma = 0.99
        self.tau = 1e-3
        self.max_iters = 10000000
        self.noise_stddev = 0.3
        self.hard_swap_interval = 100
        
        self.batch_size = 64
        self.actor_lr   = 1e-4 
        self.critic_lr  = 1e-4
        self.weight_decay = 0.002

        self.results_path = '/home/luca/Desktop/luca/ddpg/'
        self.checkpoint_dir = os.path.join(self.results_path, 'checkpoint/')
        self.statistic_dir = os.path.join(self.results_path, 'statistics/')

        #if folder do not exists, create it
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.statistic_dir, exist_ok=True)