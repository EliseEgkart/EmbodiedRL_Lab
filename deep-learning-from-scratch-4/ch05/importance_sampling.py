"""Toy example of expectation, Monte Carlo, and importance sampling."""

import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# Exact expectation under the target distribution pi:
#   E_pi[x] = sum_i pi_i x_i
e = np.sum(x * pi)
print("E_pi[x]", e)

# Monte Carlo under the target distribution.
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)
print("MC: {:.2f} (var: {:.2f})".format(np.mean(samples), np.var(samples)))

# Importance sampling:
# Draw from behavior distribution b and reweight by
#   rho = pi(x) / b(x)
# so that E_b[rho x] = E_pi[x] as long as b has support everywhere pi does.
b = np.array([0.2, 0.2, 0.6])
samples = []
for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)
print("IS: {:.2f} (var: {:.2f})".format(np.mean(samples), np.var(samples)))
