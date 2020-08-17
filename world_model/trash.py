##############
## GRU: modello 1-step haed... cellGRU

class RNN_GRU(jit.ScriptModule):
    #__constants__ = ['latent_size', 'action_size', 'rnn_hidden_size]

    def __init__(self, latent_size, action_size, rnn_hidden_size):
        super().__init__()
        self.rnn = nn.GRUCell(latent_size + action_size, rnn_hidden_size   )
        self.act_fn = getattr(F,'relu')
        self.fc_reconstruction = nn.Linear(rnn_hidden_size, 1)#latent_size + action_size)

    #@jit.script_method
    def forward(self, actions, latents, prev_beliefs):
        T = actions.size(0)
        beliefs =  [torch.empty(0)] * (T + 1)

        reconstruction = [torch.empty(0)] * T

        beliefs[0] = prev_beliefs # il primo valore è solo per l' inizializzazione. Non restituirlo
        input = torch.cat([actions, latents], dim=-1)

        for t in range(T):
            beliefs[t+1] = self.rnn(input[t],beliefs[t])
            reconstruction[t] = self.act_fn(self.fc_reconstruction(beliefs[t + 1])) 
        
        beliefs = torch.stack(beliefs[1:], dim=0)
        reconstruction = torch.stack(reconstruction, dim=0)

        return beliefs,reconstruction

### In training
prev_bel = torch.zeros(self.parms.batch_size, self.parms.rnn_hidden_size, device=self.parms.device)
beliefs,reconstruction = self.gru.forward(latent_obs.detach(),actions,prev_bel)
reconstruction = reconstruction.squeeze(dim=2)