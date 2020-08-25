import gym

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

