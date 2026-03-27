"""Compare two equivalent estimators for the sample mean."""

import numpy as np

# Naive implementation:
# Store every reward and compute the arithmetic mean exactly as
#   Q_n = (1 / n) * sum_{i=1}^n R_i
# This is mathematically transparent but uses O(n) memory.
np.random.seed(0)
rewards = []

for n in range(1, 11):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)

print("---")

# Incremental implementation:
# Rearranging the sample mean gives the classic online update
#   Q_n = Q_{n-1} + (R_n - Q_{n-1}) / n
# The term (R_n - Q_{n-1}) is the prediction error.
# This version is O(1) in memory and is the basis of many RL updates.
np.random.seed(0)
Q = 0

for n in range(1, 11):
    reward = np.random.rand()
    Q = Q + (reward - Q) / n
    print(Q)
