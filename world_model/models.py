from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d
import math
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np



# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

class VAE(jit.ScriptModule):
    __constants__ = ['latent_size']

    def __init__(self, latent_size):
        super().__init__()
        self.act_fn = getattr(F,'relu')
        self.latent_size = latent_size

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, self.latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, self.latent_size)

        # Decoder
        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    @jit.script_method
    def encode(self, x): 
        x = self.act_fn((self.conv1(x)))
        x = self.act_fn(self.conv2(x))
        x = self.act_fn(self.conv3(x))
        x = self.act_fn(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        # Reparametrization trick
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        return z


    @jit.script_method
    def decode(self, x): 
        x = self.act_fn((self.fc1(x)))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.act_fn(self.deconv1(x))
        x = self.act_fn(self.deconv2(x))
        x = self.act_fn(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction

    @jit.script_method
    def forward(self, img):
        z = self.encode(img)
        recon_img = self.decode(z)
        return recon_img

    
    # MSE loss 
    @jit.script_method
    def get_loss(self,recon_x, x):
        return F.mse_loss(recon_x, x)

'''
class MDRNNcell(jit.ScriptModule):
    __constants__ = ['num_gaussians','size_mu_sigma','latent_size']

    def __init__(self, latent_size, action_size, rnn_hidden_size,gmm_output_size, num_gaussians, min_action, max_action):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.rnn = nn.LSTMCell(latent_size + action_size, rnn_hidden_size)
        self.gmm_layer = nn.Linear(rnn_hidden_size, gmm_output_size) # (mu,sigma,pi,rw,done) for all gaussian 
        self.latent_size = latent_size
        self.size_mu_sigma = latent_size * num_gaussians
        self.min_action = min_action
        self.max_action = max_action
        self.rnn_hidden_size = rnn_hidden_size

    #@jit.script_method
    def forward(self, latents, actions):
        #print("latents: ",latents.shape)
        #print("actions: ", actions.shape)

        inputs = torch.cat([actions, latents], dim=-1)
        inputs = inputs.squeeze(dim=0)
        batch_size = inputs.size(0)
        
        
        self.current_hidden_cell_state = 2 * [torch.zeros(batch_size, self.rnn_hidden_size).to(device=inputs.device)]
        next_hidden, next_cell_state = self.rnn(inputs,self.current_hidden_cell_state)
        self.current_hidden_cell_state = [next_hidden, next_cell_state]

        gmm_outs = self.gmm_layer(next_hidden)

        pred_mus,pred_sigmas,pred_pi,pred_rw,pred_flag = torch.split(gmm_outs,[self.size_mu_sigma,self.size_mu_sigma, self.num_gaussians,1,1], dim=-1)
        pred_mus = pred_mus.view(-1, self.num_gaussians, self.latent_size)
        pred_sigmas = pred_sigmas.view(-1, self.num_gaussians, self.latent_size)
        pred_sigmas = torch.exp(pred_sigmas) 
        pred_pi = pred_pi.view(-1, self.num_gaussians)
        log_pred_pi = F.log_softmax(pred_pi, dim=-1)

        pred_rw = torch.clamp(pred_rw, min=self.min_action, max=self.max_action).int()
        pred_flag = torch.clamp(pred_flag, min=0, max=1)

        return pred_mus, pred_sigmas, log_pred_pi, pred_rw,pred_flag 

    #@jit.script_method
    def get_multiple_prediction(self,mu,sigma,logpi):
        
        mu = mu.squeeze(dim=0)
        logpi = logpi.squeeze(dim=0)
        sigma = sigma.squeeze(dim=0)
        mixt = Categorical(torch.exp(logpi)).sample()

        ######################
        mixt = mixt.unsqueeze(dim=-1).unsqueeze(dim=-1)
        mu = mu.cpu().numpy()
        #sigma = sigma.cpu().numpy()
        inds = mixt.cpu().numpy()

        latent_mu = np.take_along_axis(mu,inds,axis=1)
        latent_mu = torch.from_numpy(latent_mu)
        latent_mu = latent_mu.squeeze(dim=1)

        #latent_sigma = np.take_along_axis(sigma,inds,axis=1)
        #latent_sigma = torch.from_numpy(latent_sigma)
        #latent_sigma = latent_sigma.squeeze(dim=1)
                        
        next_obs = latent_mu  #+ latent_sigma #* torch.randn_like(mu[:, mixt, :])

        return next_obs
'''

#Mixture Density Recurrent Neural Networks
class MDRNN(jit.ScriptModule):
    __constants__ = ['num_gaussians','size_mu_sigma','latent_size']

    def __init__(self, latent_size, action_size, rnn_hidden_size, gmm_output_size, num_gaussians, min_action, max_action):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.rnn = nn.LSTM(latent_size + action_size, rnn_hidden_size)
        self.gmm_layer = nn.Linear(rnn_hidden_size, gmm_output_size) # (mu,sigma,pi,rw,done) for all gaussian 
        self.latent_size = latent_size
        self.size_mu_sigma = (latent_size) * num_gaussians
        self.min_action = min_action
        self.max_action = max_action

    def get_prediction(self,mu,sigma,logpi):       
        mixt = Categorical(torch.exp(logpi)).sample().item()
        latent_mu = torch.split(mu[mixt,:],[self.latent_size], dim=-1)
        latent_sigma = torch.split(sigma[mixt,:],[self.latent_size], dim=-1)
            
        next_obs = latent_mu  #+ latent_sigma #* torch.randn_like(mu[:, mixt, :])

        return next_obs

    @jit.script_method
    def forward(self, latents, actions):
        single_step = False

        seq_len, bs = actions.size(0), actions.size(1)
        inputs = torch.cat([actions, latents], dim=-1)
        outs,(hn, cn) = self.rnn(inputs) 
        gmm_outs = self.gmm_layer(outs) ### -< se qui mettessi hn?
        
        pred_mus,pred_sigmas,pred_pi,pred_rw,pred_flag = torch.split(gmm_outs,[self.size_mu_sigma,self.size_mu_sigma, self.num_gaussians,1,1], dim=-1)
        pred_mus = pred_mus.view(seq_len, bs, self.num_gaussians, self.latent_size)
        pred_sigmas = pred_sigmas.view(seq_len, bs, self.num_gaussians, self.latent_size)
        pred_sigmas = torch.exp(pred_sigmas) 
        pred_pi = pred_pi.view(seq_len, bs, self.num_gaussians)
        log_pred_pi = F.log_softmax(pred_pi, dim=-1)

        pred_rw = torch.clamp(pred_rw, min=self.min_action, max=self.max_action).int()
        pred_rw = pred_rw.squeeze(dim=0)
        pred_flag = torch.clamp(pred_flag, min=0, max=1)

        return pred_mus, pred_sigmas, log_pred_pi, pred_rw,pred_flag     
    
    #@jit.script_method
    def get_loss(self,pi,mu,sigma,next_latent):
        batch = torch.cat([next_latent], dim=-1)
        batch = batch.unsqueeze(-2)

        #Calculate  G = P(pred_mus, pred_sigmas)
        normal_dist = Normal(mu, sigma)
        # Calculate log( P(z' | z,a)  )
        g_log_probs = normal_dist.log_prob(batch)    
        
        gmm_loss = self.get_gmm_loss(pi,g_log_probs)
        return gmm_loss

        

    @jit.script_method
    def get_gmm_loss(self, logpi, g_log_probs):  
        logpi = logpi.unsqueeze(dim=-1)
        ###g_log_probs = torch.sum(g_log_probs, dim=-1)
        loss = g_log_probs + logpi
        loss = torch.sum(loss, dim=-1)
        return -torch.mean(loss)

        # Version from: https://github.com/ctallec/world-models/blob/master/models/mdrnn.py
        # Less variance
        #logpi = logpi.unsqueeze(dim=-1)

        ###g_log_probs = logpi + torch.sum(g_log_probs, dim=-1) # my edit, less variance and lower loss values
        #g_log_probs = logpi + g_log_probs
        #max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
        #g_log_probs = g_log_probs - max_log_probs

        #g_probs = torch.exp(g_log_probs)
        #probs = torch.sum(g_probs, dim=-1)

        #log_prob = max_log_probs.squeeze() + torch.log(probs)
        #return - torch.mean(log_prob)

    #@jit.script_method
    def get_multiple_prediction(self,mu,sigma,logpi):
        
        mu = mu.squeeze(dim=0)
        logpi = logpi.squeeze(dim=0)
        sigma = sigma.squeeze(dim=0)
        mixt = Categorical(torch.exp(logpi)).sample()

        ######################
        mixt = mixt.unsqueeze(dim=-1).unsqueeze(dim=-1)
        mu = mu.cpu().numpy()
        #sigma = sigma.cpu().numpy()
        inds = mixt.cpu().numpy()

        latent_mu = np.take_along_axis(mu,inds,axis=1)
        latent_mu = torch.from_numpy(latent_mu)
        latent_mu = latent_mu.squeeze(dim=1)

        #latent_sigma = np.take_along_axis(sigma,inds,axis=1)
        #latent_sigma = torch.from_numpy(latent_sigma)
        #latent_sigma = latent_sigma.squeeze(dim=1)
                        
        next_obs = latent_mu  #+ latent_sigma #* torch.randn_like(mu[:, mixt, :])

        return next_obs
