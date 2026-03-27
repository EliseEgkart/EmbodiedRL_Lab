import numpy as np
import matplotlib.pyplot as plt

class NobstatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms) # 노이즈를 추가.
        if rate > np.random.rand():
            return 1
        else:
            return 0

class AlphaAgent:
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

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] +=1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

runs = 200
steps = 3000
epsilon = 0.1
alpha = 0.1

all_rates = np.zeros((runs, steps))
all_alpha_rates = np.zeros((runs, steps))
for run in range(runs):
    bandit = NobstatBandit()
    alpha_agent = AlphaAgent(epsilon, alpha)
    agent = Agent(epsilon)

    total_reward = 0
    total_alpha_reward = 0
    rates = []
    alpha_rates = []

    for step in range(steps):
        action = agent.get_action()
        alpha_action = alpha_agent.get_action()
        reward = bandit.play(action)
        alpha_reward = bandit.play(alpha_action)
        agent.update(action, reward)
        alpha_agent.update(action, reward)
        total_reward += reward
        total_alpha_reward += alpha_reward
        rates.append(total_reward / (step + 1))
        alpha_rates.append(total_alpha_reward / (step +1))

    all_rates[run] = rates
    all_alpha_rates[run] = alpha_rates

avg_rates = np.average(all_rates, axis=0)
alpha_avg_rates = np.average(all_alpha_rates, axis=0)

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.plot(alpha_avg_rates)
plt.grid()
plt.show()