from env import *
import os
import random
import numpy as np
import torch
from memory import ReplayMemory
from ddpg import *
import matplotlib.pyplot as plt
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from tqdm import tqdm

class Initializer():
    def __init__(self): 
        self.seed = 0
        self.use_cuda = True
        self.replay_size =  1000000
        self.gamma = 0.99
        self.tau = 1e-3
        self.device = torch.device('cuda')
        self.max_iters = 11000
        self.batch_size = 256 + 1  
        self.number_of_stack = 3
        self.results_path = 'placeholder'
        self.statistic_dir = os.path.join(self.results_path, 'statistics/')
        self.gpu_id = 0
        self.hard_swap_interval = 100    
        torch.cuda.set_device(self.gpu_id)


        #if folder do not exists, create it
        os.makedirs(self.statistic_dir, exist_ok=True)

        self.metrics = {'steps': [], 'episodes': [], 'test_episodes': [], 'train_rewards': [], 'test_rewards': [], 'actor_loss': [], 'critic_loss': []} 
        

    
    def start(self):
        self.set_seed()
        self.env = ControlSuite('cheetah-run', 2, 1000, 4, 3)
        self.agent = DDPG(self.gamma, self.tau,self.env.state_space(),self.env,self.device)
        # Initialize replay memory
        self.memory = ReplayMemory(int(self.replay_size), self.env.action_space())
        self.list_total_rewards = []
        self.list_iter = []
        self.step = 0
        self.current_episode = 0
        self.checkpoint_interval = 100
        self.train()

    
    def train(self):
        for episode in tqdm(range(self.max_iters) ):
            self.metrics['episodes'].append(episode)
            self.explore_and_collect(episode)
            self.current_episode = episode

            if (self.current_episode % self.checkpoint_interval) == 0:
                self.test(self.current_episode)
                self.save_checkpoint()

    def explore_and_collect(self, iter):
        
        done = False
        total_reward = 0
        state = self.env.reset()
        flag = False
        
        while not done:
            self.metrics['steps'] = self.step
            self.step += 1
            action = self.agent.get_action(torch.tensor(state).unsqueeze(dim=0),iter, action_noise=True)
            action = action.cpu().numpy()[0]
            next_state, reward, done, _ = self.env.step(action)
            self.memory.push(state, action, done, reward)

            total_reward += reward
            state = next_state

            if len(self.memory) > self.batch_size:
                self.fit_buffer()
                flag = True
            
            if (self.step % self.hard_swap_interval) == 0:   
                self.agent.hard_swap()

        if flag:
            self.list_iter.append(iter)
            self.list_total_rewards.append(total_reward)
            plt.plot(self.list_iter, self.list_total_rewards)
            plt.show()
            plt.savefig('reward.png')
            self.metrics['train_rewards'].append(total_reward)
            self.lineplot(self.metrics['episodes'][-len(self.metrics['train_rewards']):], self.metrics['train_rewards'], 'train_rewards', self.statistic_dir)
            self.lineplot(self.metrics['episodes'][-len(self.metrics['actor_loss']):], self.metrics['actor_loss'], 'actor_loss', self.statistic_dir)
            self.lineplot(self.metrics['episodes'][-len(self.metrics['critic_loss']):], self.metrics['critic_loss'], 'critic_loss', self.statistic_dir)
            torch.save(self.metrics, os.path.join(self.statistic_dir , 'metrics.pth'))

    def save_checkpoint(self):
        self.agent.store_model()
        
    def load_checkpoint(self):
        self.agent.load_model()    
        self.metrics = torch.load(os.path.join(self.statistic_dir, 'metrics.pth'))
        self.current_episode = self.metrics['episodes'][-1]


    def fit_buffer(self):
        transitions = self.memory.sample_batch(self.batch_size)
        
        # Update actor and critic according to the batch
        actor_loss, critic_loss = self.agent.update_params(transitions)
        self.metrics['actor_loss'].append(actor_loss)
        self.metrics['critic_loss'].append(critic_loss)

    def test(self, episode):
        self.agent.eval_mode()
        i = 0 
        total_reward = 0
        done = False
        state = self.env.reset()
        
        while not done:
            action = self.agent.get_action(torch.tensor(state).unsqueeze(dim=0),episode, action_noise=False)
            next_state, reward, done, _ = self.env.step(action.cpu().numpy()[0])

            total_reward += reward
            state = next_state        
            i +=1

        print("Result of test: ", total_reward)
        self.metrics['test_rewards'].append(total_reward)
        self.metrics['test_episodes'].append(episode)        
        self.lineplot(self.metrics['test_episodes'][-len(self.metrics['test_rewards']):], self.metrics['test_rewards'], 'test_rewards', self.statistic_dir)
        
        self.agent.train_mode()

    # Plots min, max and mean + standard deviation bars of a population over time
    def lineplot(self, xs, ys_population, title, path='', xaxis='episode'):
        max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

        if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
            ys = np.asarray(ys_population, dtype=np.float32)
            ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
            ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

            trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
            trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
            trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
            trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
            trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
            trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
            data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
        else:
            data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
        plotly.offline.plot({
            'data': data,
            'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
        }, filename=os.path.join(path, title + '.html'), auto_open=False)

    def set_seed(self):
        print("Setting seed")
        os.environ['PYTHONHASHSEED']=str(self.seed)
        #random.seed(self.seed)
        #torch.random.seed()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)   
        if self.use_cuda:
            torch.cuda.manual_seed(self.seed)
            #torch.backends.cudnn.enabled=False
            #torch.backends.cudnn.deterministic=True


if __name__ == "__main__":
    I = Initializer()
    I.start()

    