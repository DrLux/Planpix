import os

class Parameters():
    def __init__(self):
        # Parametri che cambio più frequentemente
        self.gpu_id = 0
        self.seed = 256
        self.num_init_episodes = 10
        self.use_cuda = True
        #self.collect_interval = 10 


        ### ENV ####
        self.env_name = 'cartpole-balance'
        self.max_episode_length = 1000
        self.bit_depth = 5
        
        ### Experience Replay
        self.ex_replay_buff_size = 1000000
        
        # Setup
        self.results_path = '/home/luca/Desktop/luca/agosto_world_model/con_gauss_noise'

        # Model Parameters
        self.latent_size = 1024
        self.rnn_hidden_size = 1026
        self.num_gaussians = 3
        self.gmm_output_size = ((self.latent_size * self.num_gaussians) *2) + self.num_gaussians + 2
        self.reward_model_hidden_size = 200
        self.sotchastic_state_size = 1026
        

        # Planner
        self.planning_horizon = 12
        self.optimisation_iters = 10
        self.candidates = 1000
        self.top_candidates = 100

        # Learning
        self.adam_epsilon = 1e-4
        self.learning_rate_schedule = 0 #Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)' 
        self.learning_rate = 1e-3
        self.grad_clip_norm = 1000
        self.activation_function = 'relu'
        self.device = None
        self.batch_size = 50
        self.chunk_size = 13#32

        
        # Interactions with the environment
        self.action_noise = 0.3
        self.test_episodes = 3
        self.flag_render = False
        self.training_episodes = 1000
        self.collect_interval = 100 #numero di campioni che peschi dal buffer ad ogni iterazione 
        self.test_interval = 20

        # os
        self.results_dir = os.path.join(self.results_path)
        self.dataset_path = os.path.join(self.results_path,'dataset/')
