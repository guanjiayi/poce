import torch

from torch.distributions import Normal
from omnisafe.common.mlp import MLP

class VAE_CAUG(torch.nn.Module):
    def __init__(self, state_dim, action_dim, vae_features, vae_layers, max_action=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = 2 * self.action_dim
        self.max_action = max_action

        self.encoder = MLP(self.state_dim + self.action_dim, 2 * self.latent_dim, vae_features, vae_layers, hidden_activation='relu')
        self.decoder = MLP(self.state_dim + self.latent_dim, self.action_dim, vae_features, vae_layers, hidden_activation='relu')
        self.noise = MLP(self.state_dim + self.action_dim, self.action_dim, vae_features, vae_layers, hidden_activation='relu')

    def encode(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        mu, logstd = torch.chunk(self.encoder(state_action), 2, dim=-1)
        logstd = torch.clamp(logstd, min=-20, max=2)
        std = torch.exp(logstd)
        return Normal(mu, std)

    # def decode(self, state, z=None):
    #     if z is None:
    #         param = next(self.parameters())
    #         z = torch.randn((*state.shape[:-1], self.latent_dim)).to(param)

    #     action = self.decoder(torch.cat([state, z], dim=-1))
    #     return action
    
    def decode(self, state, z=None):
        if z is None:
            z = Normal(0, 1).sample((*state.shape[:-1], self.latent_dim)).to(state.device)

        action = self.decoder(torch.cat([state, z], dim=-1))
        return action
    
    # def decode_multiple(self, state, z=None, num=10):
    #     if z is None:
    #         param = next(self.parameters())
    #         z = torch.randn((num, *state.shape[:-1], self.latent_dim)).to(param)
    #     state = state.repeat((num,1,1))

    #     action = self.decoder(torch.cat([state, z], dim=-1))
    #     return action
    
    def decode_multiple(self, state, z=None, num=10):
        if z is None:
            z = Normal(0, 1).sample((num, *state.shape[:-1], self.latent_dim)).to(state.device)
        state = state.repeat((num,1,1))
        action = self.decoder(torch.cat([state, z], dim=-1))
        return action

    def forward(self, state, action):
        dist = self.encode(state, action)
        z = dist.rsample()
        action = self.decode(state, z)
        return dist, action