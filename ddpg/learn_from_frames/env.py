import gym
from dm_control import suite
import torch
import numpy as np
from dm_control.suite.wrappers import pixels
from collections import deque  

from torchvision import transforms
import torch

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
    def __init__(self, env_name, seed, max_episode_length, action_repeat, number_of_stack):
        domain, task = env_name.split('-')
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.number_of_stack = number_of_stack
        # Initialize deque with zero-images one array for each image
        self.stacked_frames  =  deque([np.zeros((64,64), dtype=np.int) for i in range(self.number_of_stack)], maxlen=self.number_of_stack)


    def close(self):
        self._env.close()
    
    def get_obs(self):
        frame = self._env.physics.render(height=64, width=64, camera_id=0)
        frame = torch.from_numpy(frame.copy())
        frame = torch.transpose(frame, 0, 2)
        trans = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    ])
        preprocessed_frame = trans(frame)
        return preprocessed_frame.numpy()

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset() 
        obs = self.get_obs()

        # Because we're in a new episode, copy the same frame 3x
        self.stacked_frames.append(obs)
        self.stacked_frames.append(obs)
        self.stacked_frames.append(obs)

        # Stack the frames
        current_frame = np.stack(self.stacked_frames, axis=1)
        return current_frame

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
        self.stacked_frames.append(frame)
        current_frame = np.stack(self.stacked_frames, axis=1)
        
        return current_frame,reward,done,obs        
    
    def action_range(self):
        action = self._env.action_spec()
        return action.minimum[0], action.maximum[1]

    def state_space(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()])
    
    def action_space(self):
        return self._env.action_spec().shape[0]

