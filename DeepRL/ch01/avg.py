import numpy as np

np.random.seed(0)
rewards = []

for n in range(1, 11):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)


# ---
# incremental impletation - 증분구현 방식의 시간복잡도 완화.
Q = 0

for n in range(1, 10000):
    reward = np.random.rand()
    Q = Q + (reward - Q) / n
    Q += (reward - Q) / n
    print(Q)