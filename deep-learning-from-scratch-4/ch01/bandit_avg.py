"""Average the epsilon-greedy bandit performance over many runs."""

import matplotlib.pyplot as plt
import numpy as np

from bandit import Agent, Bandit


runs = 200
steps = 1000
epsilon = 0.1

# all_rates[run, step] stores the empirical average reward up to `step`
# for one independent bandit problem.
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    all_rates[run] = rates

# Averaging across runs reduces the variance of a single lucky or unlucky
# bandit instance and reveals the expected learning curve more clearly.
avg_rates = np.average(all_rates, axis=0)

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates)
plt.show()
