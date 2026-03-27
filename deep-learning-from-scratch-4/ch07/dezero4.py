"""Nonlinear regression with a small two-layer neural network."""

import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

# Dataset:
# A linear model cannot fit this sinusoidal target well, so a hidden layer
# is introduced to learn a nonlinear basis.
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000


class TwoLayerNet(Model):
    """Two-layer perceptron for scalar regression."""

    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        # The hidden sigmoid layer introduces nonlinearity so the network
        # can approximate curved functions instead of only straight lines.
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(10, 1)
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss.data)

plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = model(t)
plt.plot(t, y_pred.data, color="r")
plt.show()
