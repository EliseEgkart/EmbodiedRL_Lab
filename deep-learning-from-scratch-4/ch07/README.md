# Chapter 7: Neural Networks and Function Approximation

## Why Tables Stop Scaling

Tabular methods assign an independent parameter to each state or state-action pair. This is exact in tiny environments, but it fails when:

- the state space is large,
- the state space is continuous,
- generalization across similar states is needed.

Function approximation replaces lookup tables with parameterized models such as linear maps or neural networks.

## Linear Algebra Foundation

The simplest neural building block is an affine map:

`y = xW + b`

where:

- `x` is the input row vector or minibatch,
- `W` is a weight matrix,
- `b` is a bias vector.

[dezero1.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch07/dezero1.py) isolates matrix multiplication because every dense layer is built from this operation.

## Optimization View

Learning becomes an optimization problem:

`theta^* = argmin_theta L(theta)`

For regression, the standard loss is mean squared error:

`L(theta) = (1 / N) sum_i (y_i - f_theta(x_i))^2`

[dezero3.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch07/dezero3.py) fits a line using this objective.

The gradient-descent step is

`theta <- theta - eta * grad_theta L(theta)`

where `eta` is the learning rate.

## Automatic Differentiation

[dezero2.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch07/dezero2.py) optimizes the Rosenbrock function to show the mechanics of backpropagation:

- build a computation graph,
- compute the forward scalar loss,
- call `backward()`,
- read gradients,
- update parameters.

This matters because deep RL later relies on gradients through value networks and policy networks rather than direct table assignment.

## Nonlinearity

A stack of affine maps without nonlinear activation is still just one affine map. To represent nonlinear functions, we need activations such as sigmoid or ReLU:

- sigmoid: smooth, bounded, historically important,
- ReLU: piecewise linear, simple, effective in deep nets.

[dezero4.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch07/dezero4.py) shows how a hidden nonlinear layer can fit a sinusoidal target that a linear model cannot capture.

## Q-Function Approximation

Once a network outputs a vector of action values,

`Q_theta(s) = [Q_theta(s, a_1), ..., Q_theta(s, a_k)]`

we can approximate tabular Q-learning with gradient descent.

In [q_learning_nn.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch07/q_learning_nn.py), the target is

`y = r + gamma max_{a'} Q_theta(s', a')`

and the loss is

`L(theta) = (Q_theta(s, a) - y)^2`

This is called a semi-gradient method because the target is treated as fixed during differentiation.

## Why One-Hot Encoding Works Here

GridWorld states are discrete coordinates. To feed them into a neural network, the code maps each state to a one-hot vector. This preserves exact identity:

- each state gets its own basis direction,
- similar states are not forced to share features unless the network learns to do so.

In larger problems, one-hot encoding becomes inefficient, and more structured features are needed.

## Semi-Gradient Subtlety

When the target includes `Q(next_state)`, one might accidentally let gradients flow through both the current-state branch and the target branch. The code explicitly detaches the target branch. This is mathematically important because the Bellman target should be treated as a fixed regression target for that update step.

## Implementation Insights

- Chapter 7 is where RL becomes intertwined with numerical optimization.
- Loss minimization does not guarantee stable control; the target itself depends on current parameters, which makes RL harder than ordinary supervised learning.
- The network outputs all action values at once, which is more efficient than fitting a separate model per action.
- Even in a small environment, the code already foreshadows the optimization issues that explode in deep RL: moving targets, correlated data, and bootstrap instability.

## Bridge to Chapter 8

Once we use neural Q-functions in a richer environment such as CartPole, instability becomes the main issue. DQN adds replay buffers and target networks precisely to control that instability.
