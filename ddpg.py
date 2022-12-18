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

    def step(self, state, action, reward, next_state, done, num_agents):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            experiences = self.memory.sample(BATCH_SIZE, num_agents)
            self.update(experiences, num_agents)
            self.soft_update(self.target_actor, self.actor, TAU)
            self.soft_update(self.target_critic, self.critic, TAU)

        self.epsilon = max(self.eps_min, self.eps_decay * self.epsilon)

    def update(self, experiences, num_agents):
        states, actions, rewards, next_states, dones = experiences
        agent_list = [idx for idx in range(2)]
        for agent_idx in range(num_agents):
            next_actions_full = [self.target_actor(next_states[idx]) for idx in agent_list]
            next_actions_full = torch.cat(next_actions_full, dim=1)
            next_states_full = [next_states[idx] for idx in agent_list]
            next_states_full = torch.cat(next_states_full, dim=1)
            Q_targets_next = self.target_critic(next_states_full, next_actions_full)
            Q_targets = rewards[agent_idx] + (Q_targets_next * self.gamma * (1 - dones[agent_idx]))

            states_full = [states[idx] for idx in agent_list]
            states_full = torch.cat(states_full, dim=1)
            actions_full = [actions[idx] for idx in agent_list]
            actions_full = torch.cat(actions_full, dim=1)

            Q_expected = self.critic(states_full, actions_full)

            critic_loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
            self.critic_optimizer.step()

            actions_pred = [self.actor(states[idx]) for idx in agent_list]
            actions_pred = torch.cat(actions_pred, dim=1)
            actor_loss = -self.critic(states_full, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
            self.actor_optimizer.step()

            #reorder the list for observation for next agent
            agent_list = agent_list[1:]
            agent_list.append(agent_idx)

    def soft_update(self, target_net, local_net, tau):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(local_param.data * tau + target_param.data * (1 - tau))

class ReplayBuffer:
    def __init__(self,buffer_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size, num_agents):
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

        states_per_agent = []
        actions_per_agent = []
        rewards_per_agent = []
        next_states_per_agent = []
        dones_per_agent = []

        for idx in range(num_agents):
            states_per_agent.append(torch.from_numpy(np.vstack([state[idx] for state in states])).float().to(device))
            actions_per_agent.append(torch.from_numpy(np.vstack([action[idx] for action in actions])).float().to(device))
            rewards_per_agent.append(torch.from_numpy(np.vstack([reward[idx] for reward in rewards])).float().to(device))
            next_states_per_agent.append(torch.from_numpy(np.vstack([next_state[idx] for next_state in next_states])).float().to(device))
            dones_per_agent.append(torch.from_numpy(np.vstack([done[idx] for done in dones]).astype(np.uint8)).float().to(device))


        return (states_per_agent, actions_per_agent, rewards_per_agent, next_states_per_agent, dones_per_agent)

    def __len__(self):
        return len(self.memory)
