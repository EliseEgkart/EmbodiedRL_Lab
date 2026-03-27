"""On-policy Monte Carlo control in GridWorld."""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.gridworld import GridWorld


def greedy_probs(Q, state, epsilon=0, action_size=4):
    """Return an epsilon-greedy policy derived from Q."""

    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1 - epsilon
    return action_probs


class McAgent:
    """Monte Carlo control agent using constant-step updates."""

    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        # Stores one full episode as (state, action, reward) tuples so the
        # return G_t can be computed after the episode terminates.
        self.memory = []

    def get_action(self, state):
        """Sample from the current behavior policy pi."""

        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """Append one transition summary to the episode buffer."""

        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        """Clear the episodic buffer before starting a new rollout."""

        self.memory.clear()

    def update(self):
        """Process the episode backward and improve the policy."""

        G = 0
        for state, action, reward in reversed(self.memory):
            # Standard return recursion:
            #   G_t = R_{t+1} + gamma G_{t+1}
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McAgent()

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
