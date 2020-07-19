
class Parameters():
    def __init__(self):
        # Parametri che cambio più frequentemente
        self.gpu_id = 1


        ### ENV ####
        self.env_name = 'reacher-easy'
        self.seed = 1
        self.max_episode_length = 1000
        self.bit_depth = 5
        
        ### Experience Replay
        self.ex_replay_buff_size = 1000000
        self.num_init_episodes = 5 #primo episodio random per inizializzare il buffer
        
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
        self.free_nats = 3
        self.action_noise = 0.3
        self.test_episodes = 1#3
        self.flag_render = False
        self.max_episode_length = 1000
        self.training_episodes = 500
        self.collect_interval = 100 #numero di campioni che peschi dal buffer ad ogni iterazione 
        self.chunk_size = 50
        self.grad_clip_norm = 1000
        self.test_interval = 25
        self.checkpoint_interval  = 25
