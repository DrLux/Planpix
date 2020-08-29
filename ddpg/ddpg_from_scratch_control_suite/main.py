from env import *
import os
import random
import numpy as np
import torch
from memory import ReplayMemory,Transition
from ddpg import *
import matplotlib.pyplot as plt


class Initializer():
    def __init__(self): 
        self.seed = 2
        self.use_cuda = True
        self.replay_size = 1000000
        self.gamma = 0.99
        self.tau = 1e-3
        self.hidden_size = [400, 300]
        self.device = torch.device('cuda')
        self.max_iters = 10000000
        self.batch_size = 64

    
    def start(self):
        self.set_seed()
        #env = CustomEnv('HalfCheetah-v2', 2, 1000) 
        #control_env = ControlSuite('cheetah-run', 2, 1000)
        env = ControlSuite('cheetah-run', 2, 1000)
        
        self.agent = DDPG(self.gamma, self.tau,self.hidden_size,env.state_space(),env,self.device)
        # Initialize replay memory
        memory = ReplayMemory(int(self.replay_size))

        done = False
        step = 0
        total_reward = 0
        list_total_rewards = []
        list_iter = []
        for iter in range(self.max_iters):
            state = torch.Tensor([env.reset()]).to(self.device)
            done = False
            total_reward = 0

            while not done:
                step += 1
                action = self.agent.get_action(state,iter)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

                mask = torch.Tensor([done]).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                next_state = torch.Tensor([next_state]).to(self.device)
                total_reward += reward

                memory.push(state, action, mask, next_state, reward)
                state = next_state

                if len(memory) > self.batch_size:
                    transitions = memory.sample(self.batch_size)
                    # Transpose the batch
                    # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                    batch = Transition(*zip(*transitions))

                    # Update actor and critic according to the batch
                    value_loss, policy_loss = self.agent.update_params(batch)
            
                if (step%100) == 0:
                    self.agent.hard_swap()

            print("iter: ", iter, " total_reward: ", total_reward)
            list_iter.append(iter)
            list_total_rewards.append(total_reward.cpu())
            plt.plot(list_iter, list_total_rewards)
            plt.show()
            plt.savefig('reward.png')
            
    


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

    