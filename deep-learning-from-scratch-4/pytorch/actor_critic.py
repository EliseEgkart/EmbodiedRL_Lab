"""PyTorch implementation of one-step actor-critic."""

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from common.utils import plot_total_reward


class PolicyNet(nn.Module):
    """Actor network producing a categorical policy."""

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    """Critic network estimating V(s)."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    """One-step actor-critic agent."""

    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        """Sample an action and keep its probability for the actor loss."""

        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        probs = self.pi(state)[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        """Update the critic with TD targets and the actor with advantage."""

        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        next_state = torch.tensor(next_state[np.newaxis, :], dtype=torch.float32)

        with torch.no_grad():
            target = reward + self.gamma * self.v(next_state) * (1 - done)

        v = self.v(state)
        loss_v = F.mse_loss(v, target)

        # Advantage estimate. Detaching it prevents the actor update from
        # modifying the critic through this term.
        delta = (target - v).detach()
        loss_pi = -torch.log(action_prob) * delta

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


env = gym.make("CartPole-v0")
agent = Agent()
reward_history = []

for episode in range(2000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

plot_total_reward(reward_history)
