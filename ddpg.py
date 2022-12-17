from torch.optim import Adam
import torch
import numpy as np

from model import Actor, Critic
from global_variables import SCALE, MU, THETA, SIGMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hard_update(target_net, local_net):
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
        target_param.data.copy_(local_param.data)


class OUNoise:

    def __init__(self, action_dimension, scale=SCALE, mu=MU, theta=THETA, sigma=SIGMA):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float().to(device)


class DDPGAgent:
    def __init__(self, state_size, action_size, state_size_all, action_size_all, seed, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size, seed).to(device)
        self.critic = Critic(state_size_all, action_size_all, seed).to(device)
        self.target_actor = Actor(state_size, action_size, seed).to(device)
        self.target_critic = Critic(state_size_all, action_size_all, seed).to(device)

        self.noise = OUNoise(action_size, scale=1.0 )

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)


    def act(self, state, noise=0.0):
        action = self.actor(state) + noise*self.noise.noise()
        return action

    def target_act(self, state, noise=0.0):
        action = self.target_actor(state) + noise*self.noise.noise()
        return action
