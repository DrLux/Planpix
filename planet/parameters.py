import os

class Parameters():
    def __init__(self):     

        ### ENV ####
        self.env_name = 'cheetah-run'
        self.max_episode_length = 1000
        self.bit_depth = 5
        
        ### Experience Replay
        self.ex_replay_buff_size = 1000000
        self.num_init_episodes = 5 #random episodes, usefull to fill the buffer
        
        # Setup
        self.results_path = 'placeholder'
        self.seed = 3


        # Model Parameters
        self.belief_size = 200
        self.state_size = 30
        self.hidden_size = 200
        self.embedding_size = 1024
        

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
        self.use_cuda = True
        self.gpu_id = 0
        self.batch_size = 50
        self.chunk_size = 50

        
        # Interactions with the environment
        self.free_nats = 3 # mean of the three best value in kl-loss
        self.action_noise = 0.3
        self.test_episodes = 1
        self.flag_render = False
        self.training_episodes = 1000
        self.collect_interval = 100 #number of samples to be taken from the buffer at each iteration
        self.test_interval = 100
        self.storing_dataset_interval = 100

        # os
        self.results_dir = os.path.join(self.results_path)
        self.dataset_path = os.path.join(self.results_path,'dataset/')
