"""Gradient descent on the Rosenbrock function using DeZero."""

import numpy as np
from dezero import Variable


def rosenbrock(x0, x1):
    """Rosenbrock function.

    This non-convex objective is a classic optimization benchmark because
    it contains a narrow curved valley that is harder to optimize than a
    simple quadratic bowl.
    """

    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001
iters = 10000

for i in range(iters):
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    # Standard gradient descent step:
    # parameter <- parameter - lr * gradient
    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

print(x0, x1)
