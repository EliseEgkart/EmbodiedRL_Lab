"""Multi-armed bandit solved with an epsilon-greedy sample-average agent."""

import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    """Stationary Bernoulli bandit.

    Each arm has a fixed success probability sampled once at
    initialization. Pulling arm `a` returns 1 with probability rate[a]
    and 0 otherwise.
    """

    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """Sample a binary reward from the selected arm."""

        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        return 0


class Agent:
    """Epsilon-greedy agent with sample-average value estimates."""

    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        # Q[a] estimates the expected reward E[R | A = a].
        self.Qs = np.zeros(action_size)
        # n[a] counts how many times each arm has been selected.
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        """Apply the incremental sample-average update for one arm."""

        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """Choose an action using epsilon-greedy exploration."""

        if np.random.rand() < self.epsilon:
            # Uniform exploration prevents premature commitment to an arm
            # whose estimate is high only due to early random luck.
            return np.random.randint(0, len(self.Qs))
        # Exploitation selects the empirically best arm so far.
        return np.argmax(self.Qs)


if __name__ == "__main__":
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        # Cumulative reward shows absolute performance.
        total_rewards.append(total_reward)
        # Running average reward approximates the expected reward rate of
        # the learned action-selection strategy.
        rates.append(total_reward / (step + 1))

    print(total_reward)

    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.show()
