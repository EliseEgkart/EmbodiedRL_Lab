"""Minimal tabular Q-learning implementation."""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.gridworld import GridWorld


class QLearningAgent:
    """Compact epsilon-greedy Q-learning agent."""

    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        """Choose randomly with probability epsilon, else exploit."""

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        qs = [self.Q[state, a] for a in range(self.action_size)]
        return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        """Bellman optimality backup for a single transition."""

        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


env = GridWorld()
agent = QLearningAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_q(agent.Q)
