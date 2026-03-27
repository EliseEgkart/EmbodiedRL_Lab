"""Linear regression trained with manual DeZero updates."""

import matplotlib.pyplot as plt
import numpy as np
from dezero import Variable
import dezero.functions as F

# Synthetic regression dataset:
# y = 5 + 2x + noise
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

# Parameters of the affine model y_hat = xW + b.
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    """Linear model forward pass."""

    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    """Average squared residual.

    MSE is a natural choice for Gaussian-noise regression and produces a
    smooth differentiable objective.
    """

    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:
        print(loss.data)

print("====")
print("W =", W.data)
print("b =", b.data)

# Compare the learned line with the training samples.
plt.scatter(x.data, y.data, s=10)
plt.xlabel("x")
plt.ylabel("y")
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color="r")
plt.show()
