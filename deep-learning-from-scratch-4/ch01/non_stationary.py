"""Bandit experiment in a non-stationary environment."""

import matplotlib.pyplot as plt
import numpy as np

from bandit import Agent


class NonStatBandit:
    """Non-stationary Bernoulli bandit.

    After every pull, the latent arm success probabilities drift by
    Gaussian noise. This invalidates plain sample averages because older
    observations should no longer be trusted equally.
    """

    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """Sample a reward, then perturb the environment dynamics."""

        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        return 0


class AlphaAgent:
    """Epsilon-greedy agent with a constant step size.

    The update
        Q <- Q + alpha * (R - Q)
    is an exponential moving average. Recent rewards are weighted more
    heavily, which is desirable when the environment drifts over time.
    """

    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


runs = 200
steps = 1000
epsilon = 0.1
alpha = 0.8
agent_types = ["sample average", "alpha const update"]
results = {}

for agent_type in agent_types:
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        if agent_type == "sample average":
            agent = Agent(epsilon)
        else:
            agent = AlphaAgent(epsilon, alpha)

        bandit = NonStatBandit()
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)
    results[agent_type] = avg_rates

# The expected outcome is that the constant-alpha agent eventually
# outperforms the sample-average agent because it adapts faster to drift.
plt.figure()
plt.ylabel("Average Rates")
plt.xlabel("Steps")
for key, avg_rates in results.items():
    plt.plot(avg_rates, label=key)
plt.legend()
plt.show()
