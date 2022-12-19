from torch.optim import Adam
import torch
import numpy as np

from model import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, state_size, action_size, lr_actor, lr_critic, gamma, tau, buffer_size, seed, weight_decay, epsilon, eps_decay, eps_min):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size, seed).to(device)
        self.critic = Critic(state_size, action_size, seed).to(device)
        self.target_actor = Actor(state_size, action_size, seed).to(device)
        self.target_critic = Critic(state_size, action_size, seed).to(device)


        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)


    def hard_update(self, target_net, local_net):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(local_param.data)


    def act(self, state, noise=0):
        action = self.actor(state) + noise
        action = action.cpu().data.numpy()
        return np.clip(action, -1, 1)


    def soft_update(self, target_net, local_net, tau):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(local_param.data * tau + target_param.data * (1 - tau))