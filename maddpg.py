from collections import deque, namedtuple
import random

import numpy as np
import torch

from ddpg import DDPGAgent
from global_variables import MU, THETA, SIGMA, SEED, BATCH_SIZE, LEARN_EVERY, LEARN_NUM, TAU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG():
    def __init__(self, state_size, action_size, lr_actor, lr_critic, gamma, tau, buffer_size, seed, weight_decay, epsilon, eps_decay, eps_min):
        self.agents = [
            DDPGAgent(state_size, action_size, lr_actor, lr_critic, gamma, tau, buffer_size, seed, weight_decay, epsilon, eps_decay, eps_min),
            DDPGAgent(state_size, action_size, lr_actor, lr_critic, gamma, tau, buffer_size, seed, weight_decay, epsilon, eps_decay, eps_min)
        ]

        self.noise = OUNoise(action_size)
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.gamma = gamma
        self.tau = tau
        self.t_step = 0
        self.memory = ReplayBuffer(buffer_size, seed)

    def act(self, states, add_noise=True):
        if add_noise:
            noise = self.epsilon * self.noise.noise()
        actions = [agent.act(state, noise) for agent, state in zip(self.agents, states)]
        return np.clip(actions, -1, 1)

    def step(self, state, action, reward, next_state, done, num_agents):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample(BATCH_SIZE, num_agents)
                for idx, agent in enumerate(self.agents):
                    self.update(experiences,idx)
                    agent.soft_update(agent.target_actor, agent.actor, TAU)
                    agent.soft_update(agent.target_critic, agent.critic, TAU)

    def update(self, experiences, agent_idx):
        states, actions, rewards, next_states, dones = experiences
        next_actions_full = [agent.target_actor(next_states[idx]) for idx, agent in enumerate(self.agents)]
        next_actions_full = torch.cat(next_actions_full, dim=1)
        next_states_full = torch.cat(next_states, dim=1)
        Q_targets_next = self.agents[agent_idx].target_critic(next_states_full, next_actions_full)
        Q_targets = rewards[agent_idx] + (Q_targets_next * self.gamma * (1 - dones[agent_idx]))

        states_full = torch.cat(states, dim=1)
        actions_full = torch.cat(actions, dim=1)

        Q_expected = self.agents[agent_idx].critic(states_full, actions_full)

        critic_loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        self.agents[agent_idx].critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].critic.parameters(), 1)
        self.agents[agent_idx].critic_optimizer.step()

        actions_pred = [agent.actor(states[idx]) for idx, agent in enumerate(self.agents)]
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -self.agents[agent_idx].critic(states_full, actions_pred).mean()
        self.agents[agent_idx].actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_idx].actor.parameters(), 1)
        self.agents[agent_idx].actor_optimizer.step()


class ReplayBuffer:
    def __init__(self,buffer_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size, num_agents):
        experiences = random.sample(self.memory, k=batch_size)
        
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