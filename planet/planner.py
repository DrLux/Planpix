from math import inf
import torch
from torch import jit

# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model,min_action=-inf, max_action=inf):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    @jit.script_method
    def forward(self, belief, state, explore:bool):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)  #B is the batch size, H belief size, Z state size 
        belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
        beliefs = torch.tensor([self.planning_horizon, self.candidates, H])
        states = torch.tensor([self.planning_horizon,self.candidates,Z])        
        planned_actions = torch.tensor([self.planning_horizon,self.action_size])
        mean_next_return = torch.tensor(B)
        
        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
            actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
            # Sample next states
            beliefs, states, _, _ = self.transition_model(state, actions, belief)

            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1)
            next_returns = returns[0,:]
            summed_returns = returns.sum(dim=0)

            # Re-fit belief to the K best action sequences
            _, topk = summed_returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False) # topk = 100 indexes
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)# take the best 100 actions (the final best action is the mean of them)
            top_next_returns = next_returns[topk.view(-1)].reshape(B, self.top_candidates)

            ##########################################################
            # This could go outside of the loop
            #  
            ###### prepare belief and states

            beliefs = beliefs[:, topk.view(-1)] #take 100 best beliefs
            states = states[:, topk.view(-1)] #take 100 best states

            beliefs = beliefs.mean(dim=1, keepdim=False) #usare questo

            states = states.mean(dim=1, keepdim=False) #usare questo
            planned_actions = action_mean.squeeze(dim=1)
            #######################################################

            # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
            mean_next_return = top_next_returns.mean(dim=1)
            

        # Return first action mean µ_t                    
        return action_mean[0].squeeze(dim=1),mean_next_return,beliefs, states,planned_actions
        