from env import *
import os
import random
import numpy as np
import torch
from memory import ReplayMemory,Transition
from ddpg import *

class Initializer():
    def __init__(self): 
        self.seed = 2
        self.use_cuda = True
        self.replay_size = 1000000
        self.gamma = 0.99
        self.tau = 0.001
        self.hidden_size = [400, 300]
        self.device = 'cpu'
        self.max_iters = 10000000
        self.batch_size = 10

    
    def start(self):
        self.set_seed()
        env = CustomEnv('MountainCarContinuous-v0', 2, 1000) 

        self.agent = DDPG(self.gamma, self.tau,self.hidden_size,env.state_space(),env.action_space(),self.device)
        # Initialize replay memory
        memory = ReplayMemory(int(self.replay_size))

        done = False
        step = 0
        for iter in range(self.max_iters):
            state = torch.Tensor([env.reset()]).to(self.device)
            done = False
            while not done:
                step += 1
                action = self.agent.get_action(state)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

                mask = torch.Tensor([done]).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                next_state = torch.Tensor([next_state]).to(self.device)
                print("step: ", step, " reward: ", reward)

                memory.push(state, action, mask, next_state, reward)
                state = next_state

                if len(memory) > self.batch_size:
                    transitions = memory.sample(self.batch_size)
                    # Transpose the batch
                    # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                    batch = Transition(*zip(*transitions))

                    # Update actor and critic according to the batch
                    value_loss, policy_loss = self.agent.update_params(batch)

    


    def set_seed(self):
        print("Setting seed")
        os.environ['PYTHONHASHSEED']=str(self.seed)
        random.seed(self.seed)
        #torch.random.seed()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)   
        torch.manual_seed(self.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.enabled=False
            torch.backends.cudnn.deterministic=True


if __name__ == "__main__":
    I = Initializer()
    I.start()

    