import os

class Parameters():
    def __init__(self):
        # Parametri che cambio più frequentemente
        self.gpu_id = 0
        self.seed = 256
        self.num_init_episodes = 3
        self.use_cuda = False
        #self.collect_interval = 10 


        ### ENV ####
        self.env_name = 'cheetah-run'
        self.max_episode_length = 1000
        self.bit_depth = 5
        self.frame_as_state = False
        
        ### Experience Replay
        self.ex_replay_buff_size = 1000000
        #self.num_init_episodes = 5 #primo episodio random per inizializzare il buffer
        
        # Setup
        self.results_path = '/home/luca/Desktop/luca/agosto/ddpg/'

        # Model Parameters
        self.discount_factor = 0.99
        self.update_factor =  0.001
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.weight_decay = 0.005

        # Learning
        self.adam_epsilon = 1e-4
        self.learning_rate_schedule = 0 #Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)' 
        self.learning_rate = 1e-3
        self.grad_clip_norm = 1000
        

        self.activation_function = 'relu'
        self.device = None
        #self.use_cuda = False
        #self.gpu_id = 0
        self.batch_size = 50

        
        # Interactions with the environment
        self.action_noise = 0.3
        self.test_episodes = 3
        self.training_episodes = 602
        self.collect_interval = 100 #numero di campioni che peschi dal buffer ad ogni iterazione 
        self.test_interval = 10
        self.storing_dataset_interval = 30

        # os
        self.results_dir = os.path.join(self.results_path)
        self.dataset_path = os.path.join(self.results_path,'dataset/')
