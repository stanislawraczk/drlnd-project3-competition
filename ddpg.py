from collections import deque, namedtuple
import random

from torch.optim import Adam
import torch
import numpy as np

from model import Actor, Critic
from global_variables import MU, THETA, SIGMA, SEED, BATCH_SIZE, LEARN_EVERY, TAU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hard_update(target_net, local_net):
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
        target_param.data.copy_(local_param.data)


class OUNoise:

    def __init__(self, action_dimension, mu=MU, theta=THETA, sigma=SIGMA):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.seed = random.seed(SEED)
        np.random.seed(SEED)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(size=len(x))
        self.state = x + dx
        return torch.tensor(self.state).float().to(device)


class DDPGAgent:
    def __init__(self, state_size, action_size, lr_actor, lr_critic, gamma, tau, buffer_size, seed, weight_decay, epsilon, eps_decay, eps_min):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size, seed).to(device)
        self.critic = Critic(state_size, action_size, seed).to(device)
        self.target_actor = Actor(state_size, action_size, seed).to(device)
        self.target_critic = Critic(state_size, action_size, seed).to(device)

        self.noise = OUNoise(action_size)
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.gamma = gamma
        self.tau = tau
        self.t_step = 0
        self.memory = ReplayBuffer(buffer_size, seed)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)


    def act(self, state, add_noise=True):
        if add_noise:
            noise = self.epsilon
        action = self.actor(state) + noise*self.noise.noise()
        action = action.cpu().data.numpy()
        return np.clip(action, -1, 1)

    # def target_act(self, state):
    #     action = self.target_actor(state)
    #     return action

    def step(self, state, action, reward, next_state, done, num_agents):
        self.memory.add(state, action, reward, next_state, done, num_agents)
        self.t_step += 1
        if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            experiences = self.memory.sample(BATCH_SIZE)
            self.update(experiences)
            self.soft_update(self.target_actor, self.actor, TAU)
            self.soft_update(self.target_critic, self.critic, TAU)

        self.epsilon = max(self.eps_min, self.eps_decay * self.epsilon)

    def update(self, experiences):
        states, states_full, actions, actions_full, rewards, next_states, next_states_full, dones = experiences
        self.critic_optimizer.zero_grad()
        next_actions = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states_full, next_actions)
        Q_targets = rewards + (Q_targets_next * self.gamma * (1 - dones))

        Q_expected = self.critic(states_full, actions_full)

        critic_loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
        self.critic_optimizer.step()

        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        self.actor_optimizer.step()

    def soft_update(self, target_net, local_net, tau):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(local_param.data * tau + target_param.data * (1 - tau))

class ReplayBuffer:
    def __init__(self,buffer_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'state_full', 'action', 'action_full', 'reward', 'next_state', 'next_state_full', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, num_agents):
        for idx in range(num_agents):
            e = self.experience(state[idx], action[idx], reward[idx], next_state[idx], done[idx])
            self.memory.append(e)

    def sample(self, batch_size):
        # przenieść układanie danych do funkcji uczącej i tam rozpakować dane na poszczególnych agentów
        experiences = random.sample(self.memory, k=batch_size)
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # states_full = torch.from_numpy(np.vstack([e.state_full for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        # actions_full = torch.from_numpy(np.vstack([e.action_full for e in experiences if e is not None])).float().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # next_states_full = torch.from_numpy(np.vstack([e.next_state_full for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        states = [e.state for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = [e.next_state for e in experiences if e is not None]
        dones = [e.done for e in experiences if e is not None]


        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
