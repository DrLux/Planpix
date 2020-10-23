import gym
from dm_control import suite
import torch
import numpy as np

class CustomEnv():
    def __init__(self, env_name, seed, max_episode_length):
        self._env =  gym.make(env_name)
        self.max_episode_length = max_episode_length
        self._env.seed(seed)

    def reset(self):
        self.t = 0  # Reset internal timer
        return self._env.reset()

    def close(self):
        self._env.close()

    def sample_random_action(self):
        action = self._env.action_space.sample()
        return action

    def step(self,action):
        return self._env.step(action)

    def state_space(self):
        return self._env.observation_space.shape[0]
    
    def action_space(self):
        return self._env.action_space


class ControlSuite():
    def __init__(self, env_name, seed, max_episode_length):
        domain, task = env_name.split('-')
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        self.max_episode_length = max_episode_length

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset() 
        return np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0)
        

    def close(self):
        self._env.close()

    def sample_random_action(self):
        action = self._env.action_spec()
        return np.random.uniform(action.minimum, action.maximum, action.shape)

    def step(self,action):
        step = self._env.step(action)
        state = np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in step.observation.values()], axis=0)  
        reward = step.reward
        done = step.last()
        discount = step.discount
        return state,reward,done, discount        
    
    def action_range(self):
        action = self._env.action_spec()
        return action.minimum[0], action.maximum[0]

    def state_space(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()])
    
    def action_space(self):
        return self._env.action_spec().shape[0]

