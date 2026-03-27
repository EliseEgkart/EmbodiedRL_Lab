"""Monte Carlo policy evaluation for a fixed random policy."""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.gridworld import GridWorld


class RandomAgent:
    """Agent that estimates V^pi under a uniform random policy."""

    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        # cnts[s] tracks how many Monte Carlo returns have been averaged
        # into V[s].
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        """Sample an action from the fixed evaluation policy."""

        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """Record the trajectory for episode-end return computation."""

        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        """Compute returns and average them into the state values."""

        G = 0
        for state, action, reward in reversed(self.memory):
            del action
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.eval()
            break

        state = next_state

env.render_v(agent.V)
