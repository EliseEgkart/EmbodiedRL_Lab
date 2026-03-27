"""Estimate the expected sum of dice rolls via Monte Carlo sampling."""

import numpy as np


def sample(dices=2):
    """Return one sample of the sum of `dices` fair six-sided dice."""

    x = 0
    for _ in range(dices):
        x += np.random.choice([1, 2, 3, 4, 5, 6])
    return x


trial = 1000
V, n = 0, 0

for _ in range(trial):
    s = sample()
    n += 1
    # Incremental Monte Carlo estimate of E[S].
    V += (s - V) / n
    print(V)
