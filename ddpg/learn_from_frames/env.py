import gym
from dm_control import suite
import torch
import numpy as np
from dm_control.suite.wrappers import pixels

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
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        domain, task = env_name.split('-')
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat

    def close(self):
        self._env.close()
    
    def get_obs(self):
        obs = self._env.physics.render(height=64, width=64, camera_id=0)
        obs = obs / 255 #scaled between 0,1
        return obs 

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset() 
        obs = self.get_obs()
        return obs        

    def sample_random_action(self):
        action = self._env.action_spec()
        return np.random.uniform(action.minimum, action.maximum, action.shape)

    def step(self,action):
        reward = 0
        for k in range(self.action_repeat):
            step = self._env.step(action)
            obs = step.observation['pixels']
            reward += step.reward
            self.t += 1
            done = step.last() or self.t == self.max_episode_length
            if done: break
        
        frame = self.get_obs()
        return frame,reward,done,obs        
    
    def action_range(self):
        action = self._env.action_spec()
        return action.minimum[0], action.maximum[1]

    def state_space(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()])
    
    def action_space(self):
        return self._env.action_spec().shape[0]

