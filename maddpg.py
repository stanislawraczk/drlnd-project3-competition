from collections import deque, namedtuple
import random

import numpy as np
import torch

from ddpg import DDPGAgent
from global_variables import BATCH_SIZE, LEARN_EVERY, TAU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def soft_update(target_net, local_net, tau):
#     for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
#         target_param.data.copy_(local_param.data * tau + target_param.data * (1 - tau))

def split_to_agents(state, num_agents):
    states = []
    for idx in range(num_agents):
        states.append(state[idx::2])
    return states

class MADDPG():
    def __init__(self, state_size, action_size, lr_actor, lr_critic, gamma, tau, buffer_size, seed):
        self.agent = DDPGAgent(state_size, action_size, state_size, action_size, seed, lr_actor, lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.t_step = 0
        self.memory = ReplayBuffer(buffer_size, seed)

    def act(self, states, noise=0.0):
        actions = self.agent.act(states, noise)
        actions = actions.cpu().data.numpy()
        return actions

    def target_act(self, states, noise=0.0):
        target_actions = self.agent.target_act(states, noise)
        return target_actions

    def step(self, state, action, reward, next_state, done, num_agents):
        self.memory.add(state, action, reward, next_state, done, num_agents)
        self.t_step += 1
        if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            experiences = self.memory.sample(BATCH_SIZE)
            self.update(experiences)
            self.soft_update(self.agent.target_actor, self.agent.actor, TAU)
            self.soft_update(self.agent.target_critic, self.agent.critic, TAU)

    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        self.agent.critic_optimizer.zero_grad()
        next_actions = self.target_act(next_states)
        with torch.no_grad():
            Q_tagrets_next = self.agent.target_critic(next_states, next_actions)
        Q_targets = rewards + (Q_tagrets_next * self.gamma * (1 - dones))

        Q_expected = self.agent.critic(states, actions)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(Q_expected, Q_targets.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        self.agent.critic_optimizer.step()

        self.agent.actor_optimizer.zero_grad()

        actions_pred = self.agent.actor(states) # [self.agent.actor(state) if i == agent_idx else self.agent.actor(state).detach() for i, state in enumerate(states)]

        actor_loss = self.agent.critic(states, actions_pred).mean()
        actor_loss.backward()
        self.agent.actor_optimizer.step()

    def soft_update(self, target_net, local_net, tau):
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(local_param.data * tau + target_param.data * (1 - tau))

class ReplayBuffer:
    def __init__(self,buffer_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, num_agents):
        for idx in range(num_agents):
            e = self.experience(state[idx], action[idx], reward[idx], next_state[idx], done[idx])
            self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
