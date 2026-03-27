"""PyTorch implementation of the simple episodic policy-gradient method."""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    """Policy network returning action probabilities."""

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    """Vanilla policy-gradient agent using a single episode return."""

    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        """Sample an action and retain its probability."""

        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def add(self, reward, prob):
        self.memory.append((reward, prob))

    def update(self):
        """Use the full-episode return for every policy term."""

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            del prob
            G = reward + self.gamma * G

        for reward, prob in self.memory:
            del reward
            loss += -torch.log(prob) * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


env = gym.make("CartPole-v0")
agent = Agent()
reward_history = []

for episode in range(3000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.add(reward, prob)
        state = next_state
        total_reward += reward

    agent.update()

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))
