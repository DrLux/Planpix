from env import *
import os
import random
import numpy as np
import torch
from memory import ReplayMemory,Transition
from ddpg import *
import matplotlib.pyplot as plt
from parameters import Parameters
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line


class Initializer():
    def __init__(self): 
        self.parms = Parameters()

        ## set cuda 
        if torch.cuda.is_available() and self.parms.use_cuda:
            self.parms.device = torch.device('cuda')
            torch.cuda.set_device(self.parms.gpu_id)
            print("Using gpu: ", torch.cuda.current_device())
        else:
            self.parms.device = torch.device('cpu')
            self.use_cuda = False
            print("Work on: ", self.parms.device)

        if self.parms.seed > 0: 
           self.set_seed()
    
    def start(self):
        self.metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_rewards': [], 'actor_loss': [], 'critic_loss': []} 
        self.env = ControlSuite(self.parms.env_name, self.parms.seed, self.parms.max_episode_length)
        self.memory = ReplayMemory(int(self.parms.buffer_replay_size), self.parms.checkpoint_dir)
        self.step = 0
        self.current_episode = 0
        # Initialize replay memory
        self.agent = DDPG(self.env, self.parms)
        
        #self.load_checkpoint()
        self.train()

    def train(self):
        while self.current_episode <= self.parms.max_iters:
            self.metrics['episodes'].append(self.current_episode)
            self.fill_buffer(self.current_episode)

            if (self.step % self.parms.checkpoint_interval) == 0:
                self.save_checkpoint()

            if (self.current_episode % self.parms.test_interval) == 0:
                self.test(self.current_episode)
            
            self.current_episode += 1

    def fill_buffer(self,episode):
        done = False
        state = torch.Tensor([self.env.reset()]).to(self.parms.device)
        total_reward = 0
        done = False
        while not done:
            self.step += 1
            self.metrics['steps'] = self.step
            action = self.agent.get_action(state,episode)
            next_state, reward, done, _ = self.env.step(action.cpu().numpy()[0])

            mask = torch.Tensor([done]).to(self.parms.device)
            reward = torch.Tensor([reward]).to(self.parms.device)
            next_state = torch.Tensor([next_state]).to(self.parms.device)
            total_reward += reward

            self.memory.push(state, action, mask, next_state, reward)
            state = next_state


            if len(self.memory) > self.parms.batch_size:
                self.fit_buffer()
        
            if (self.step % self.parms.hard_swap_interval) == 0:
                self.agent.hard_swap()
                

        self.metrics['train_rewards'].append(total_reward.item())
        self.lineplot(self.metrics['episodes'][-len(self.metrics['train_rewards']):], self.metrics['train_rewards'], 'train_rewards', self.parms.statistic_dir)
        self.lineplot(self.metrics['episodes'][-len(self.metrics['actor_loss']):], self.metrics['actor_loss'], 'actor_loss', self.parms.statistic_dir)
        self.lineplot(self.metrics['episodes'][-len(self.metrics['critic_loss']):], self.metrics['critic_loss'], 'critic_loss', self.parms.statistic_dir)
        torch.save(self.metrics, os.path.join(self.parms.statistic_dir , 'metrics.pth'))


    def test(self, episode):
        done = False
        state = torch.Tensor([self.env.reset()]).to(self.parms.device)
        total_reward = 0
        done = False
        while not done:
            action = self.agent.get_action(state,episode, action_noise=False)
            next_state, reward, done, _ = self.env.step(action.cpu().numpy()[0])
            total_reward += reward
            state = torch.Tensor([next_state]).to(self.parms.device)
        
        self.metrics['test_rewards'].append(total_reward)
        self.lineplot(self.metrics['episodes'][-len(self.metrics['test_rewards']):], self.metrics['test_rewards'], 'test_rewards', self.parms.statistic_dir)


    def fit_buffer(self):
        transitions = self.memory.sample(self.parms.batch_size)
        # Transpose the batch
        # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))

        # Update actor and critic according to the batch
        actor_loss, critic_loss = self.agent.update_params(batch)
        self.metrics['actor_loss'].append(actor_loss)
        self.metrics['critic_loss'].append(critic_loss)

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
    
    def save_checkpoint(self):
        self.agent.store_model()
        
    def load_checkpoint(self):
        self.agent.load_model()    
        self.metrics = torch.load(os.path.join(self.parms.statistic_dir, 'metrics.pth'))
        self.current_episode = self.metrics['episodes'][-1]

    def set_seed(self):
        print("Setting seed")
        os.environ['PYTHONHASHSEED']=str(self.parms.seed)
        random.seed(self.parms.seed)
        #torch.random.seed()
        np.random.seed(self.parms.seed)
        torch.manual_seed(self.parms.seed)   
        torch.manual_seed(self.parms.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.parms.seed)
            torch.backends.cudnn.enabled=False
            torch.backends.cudnn.deterministic=True


if __name__ == "__main__":
    I = Initializer()
    I.start()

    