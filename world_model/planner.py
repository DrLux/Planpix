from math import inf
import torch
from torch import jit

# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, mdrnn, vae, device,min_action=-inf, max_action=inf):
        super().__init__()
        self.mdrnn = mdrnn
        self.vae = vae 
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.device = device

    #@jit.script_method
    def get_action(self, current_obs, done_flag):
        current_z = self.vae.encode(current_obs.to(device=self.device))
        latent_size = current_z.size(1)
        current_z = current_z.expand(self.candidates, latent_size)

        predicted_z = torch.tensor([self.planning_horizon, self.candidates, latent_size]) #init prediction

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(self.planning_horizon, 1, self.action_size, device=self.device), torch.ones(self.planning_horizon, 1, self.action_size, device=self.device)   
        planned_actions = torch.tensor([self.planning_horizon,self.action_size])
        #mean_next_return = torch.tensor(B)
        
        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, self.candidates, self.action_size, device=action_mean.device)) # Sample actions (time x candidates x actions)
            actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
            # Sample next states
            
            # Calculate expected returns (technically sum of rewards over planning horizon)
            #actions = [12, 1000, 1]
            returns = self.evaluate_plan(current_z,actions,done_flag)

            
            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False) # topk = 100 indexes
            #topk += self.candidates * torch.arange(0, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, self.top_candidates, self.action_size)# take the best 100 actions (the final best action is the mean of them)

            # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=1, keepdim=True), best_actions.std(dim=1, unbiased=False, keepdim=True)
            
        print("finito")
        assert 1 == 2

        # Return first action mean µ_t                    
        #return action_mean[0].squeeze(dim=1)
        
    def evaluate_plan(self, latent, actions,done_flag):
        seq_len, candidates = actions.size(0), actions.size(1)
        total_reward = torch.zeros([candidates,1])

        for t in range(seq_len):
            print("optim ", t)
            latent = latent.unsqueeze(dim=0)
            action = actions[t].unsqueeze(dim=0)
            pred_mus, pred_sigmas, log_pred_pi = self.mdrnn.forward(latent,action) # -> lui è addestrato con 12,50,1025
            latent, next_reward,next_flag = self.mdrnn.get_multiple_prediction(pred_mus, pred_sigmas, log_pred_pi)
            latent = latent.to(device=self.device)
            total_reward += next_reward

        total_reward.to(device=self.device)
        return total_reward