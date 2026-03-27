"""Off-policy Monte Carlo control with importance sampling."""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.gridworld import GridWorld
from common.utils import greedy_probs


class McOffPolicyAgent:
    """Monte Carlo control with separate target and behavior policies."""

    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.2
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        """Sample from the exploratory behavior policy b."""

        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """Store one step so the full return can be reconstructed later."""

        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        """Replay the episode backward with importance corrections."""

        G = 0
        rho = 1

        for state, action, reward in reversed(self.memory):
            key = (state, action)

            # rho accumulates the probability ratio between the target
            # policy and the behavior policy along the suffix trajectory.
            G = self.gamma * rho * G + reward
            self.Q[key] += (G - self.Q[key]) * self.alpha
            rho *= self.pi[state][action] / self.b[state][action]

            # The target policy is greedified, while the behavior policy
            # stays epsilon-greedy so data collection remains exploratory.
            self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
            self.b[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McOffPolicyAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)
