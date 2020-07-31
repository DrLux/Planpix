
class Parameters():
    def __init__(self):
        # Parametri che cambio più frequentemente
        self.gpu_id = 1
        self.seed = 1#123
        self.num_init_episodes = 10
        self.collect_interval = 100


        ### ENV ####
        self.env_name = 'cheetah-run'
        self.max_episode_length = 1000
        self.bit_depth = 5
        
        ### Experience Replay
        self.ex_replay_buff_size = 1000000
        #self.num_init_episodes = 5 #primo episodio random per inizializzare il buffer
        
        # Setup
        #self.results_dir = os.path.join('/home/luca/Desktop/luca/agosto/experience/')
        self.results_path = '/home/luca/Desktop/luca/agosto/experience/'

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

        # Regularizer
        self.reg_batch_size = 80
        self.reg_chunck_len = self.planning_horizon
        self.reg_hidden_size = 500
        self.noise_std = 0.5
        self.reg_num_hidden_layers = 5

        # Learning
        self.activation_function = 'relu'
        self.learning_rate_schedule = 0 #Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)' 
        self.learning_rate = 1e-3
        self.adam_epsilon = 1e-4
        self.device = None
        self.use_cuda = True
        #self.gpu_id = 0
        self.batch_size = 50
        
        # Interactions with the environment
        self.free_nats = 3 # nella loss di KL invece di prendere il valore maggiore di distanza prende la media dei 3 valori 
        self.action_noise = 0.3
        self.test_episodes = 3
        self.flag_render = False
        self.max_episode_length = 1000
        self.training_episodes = 802
        #self.collect_interval = 10#0 #numero di campioni che peschi dal buffer ad ogni iterazione 
        self.chunk_size = 50
        self.grad_clip_norm = 1000
        self.test_interval = 20
        self.checkpoint_interval = 20
