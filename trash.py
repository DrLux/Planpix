######### DUMP PLAN
    '''
    def dump_plan(self): 
        # Set models to eval mode
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()

        with torch.no_grad():
            #save_image(torch.as_tensor(observation), os.path.join(self.results_dir, 'intial_obs.png'))
            done = False
            observation, video_frames, reward = self.env.reset(),[], 0
            belief, posterior_state, action = torch.zeros(1, self.parms.belief_size, device=self.parms.device), torch.zeros(1, self.parms.state_size, device=self.parms.device), torch.zeros(1, self.env.action_size, device=self.parms.device)
            tqdm.write("Dumping plan")
            observation = observation.to(device=self.parms.device)

            ##### add reward to the encoded obs
            encoded_obs = self.encoder(observation).unsqueeze(dim=0)
            rew_as_obs = torch.tensor([reward]).type(torch.float).unsqueeze(dim=0)
            rew_as_obs = rew_as_obs.unsqueeze(dim=-1).cuda()
            enc_obs_with_rew = torch.cat([encoded_obs, rew_as_obs], dim=2)
            #####

            belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, enc_obs_with_rew)  

            belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
            
            next_action, beliefs, states, plan = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)      
            
            predicted_frames = self.observation_model(beliefs, states).cpu()

            for i in range(self.parms.planning_horizon):
                plan[i].clamp_(min=self.env.action_range[0], max=self.env.action_range[1])  # Clip action range
                next_observation, reward, done = self.env.step(plan[i].cpu())  
                next_observation = next_observation.squeeze(dim=0)
                video_frames.append(make_grid(torch.cat([next_observation, predicted_frames[i]], dim=1) + 0.5, nrow=2).numpy())  # Decentre
                #save_image(torch.as_tensor(next_observation), os.path.join(self.results_dir, 'original_obs%d.png' % i))
                #save_image(torch.as_tensor(predicted_frames[i]), os.path.join(self.results_dir, 'prediction_obs%d.png' % i))


            write_video(video_frames, 'dump_plan', self.video_path)  # Lossy compression
        
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()
    '''

###################
OLD REGULARIZER

class Regularizer(jit.ScriptModule):

    #DAE with (obs, act, next obs) as the data
    def __init__(self,obs_size, act_size, hidden_size, num_hidden_layers,act_fn='relu'):
        super().__init__()
        self.linear_input_layer = nn.Linear(obs_size*2 + act_size, hidden_size) #input is the concatanation of obs,act,next_obs
        self.linear_hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_output_layer =  nn.Linear(hidden_size, obs_size*2 + act_size)   
        self.act_fn = getattr(F, act_fn)
        self.num_hidden_layers = num_hidden_layers
        
    def predict(self,input):
        hidden = self.act_fn(self.linear_input_layer(input))
        for _ in range(self.num_hidden_layers):
            hidden = self.act_fn(self.linear_hidden_layer(hidden))
        output = self.act_fn(self.linear_output_layer(hidden))
        return output


#####à 
Preprocess data

obs = obs.view(1,1,-1)# -1 stand for 3*64*64
next_obs = next_obs.view(1,1,-1)
        
input = torch.cat([obs,acts,next_obs] , dim=2) #input.shape:  torch.Size([1, 1, 24582])
