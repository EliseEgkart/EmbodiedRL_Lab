# Chapter 2: Markov and MDP Foundations

## Note on This Repository

This repository does not currently contain executable `ch02` source files. This README fills the mathematical gap between bandits and dynamic programming.

## Why Chapter 2 Exists

Bandits have no state transitions. Reinforcement learning becomes fundamentally richer once the agent's action changes the future state distribution. That requires a probabilistic dynamical system description.

## Markov Property

A process is Markov if the future depends on the present state but not on the full past:

`P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t)`

This assumption is what makes value functions computationally tractable. Without it, the sufficient statistic for control would generally be the entire history.

## Markov Reward Process

A Markov reward process is defined by

- state space `S`,
- transition matrix `P`,
- reward function `R`,
- discount factor `gamma in [0, 1)`.

The return from time `t` is

`G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...`

The state value function is

`V(s) = E[G_t | S_t = s]`

The discount factor has three roles:

- it expresses preference for earlier reward,
- it keeps infinite-horizon sums finite,
- it makes Bellman operators contractions under standard assumptions.

## Markov Decision Process

An MDP extends the Markov reward process by adding an action set `A` and controlled dynamics:

`p(s', r | s, a)`

This joint kernel states the probability of moving to `s'` and receiving reward `r` after taking action `a` in state `s`.

## Policy

A policy is a conditional distribution over actions:

`pi(a | s) = P(A_t = a | S_t = s)`

Deterministic policies choose one action with probability 1. Stochastic policies assign nonzero probability to multiple actions. Epsilon-greedy is one concrete stochastic policy family.

## Return and Discounting

The discounted return is

`G_t = sum_{k=0}^inf gamma^k R_{t+k+1}`

Important limiting cases:

- `gamma = 0`: only immediate reward matters.
- `gamma` close to `1`: long-term consequences dominate.

In implementation, `gamma` controls target propagation depth. Large `gamma` lets reward information travel farther but can also increase variance and slow contraction.

## State Value and Action Value

For a fixed policy `pi`,

`V^pi(s) = E_pi[G_t | S_t = s]`

`Q^pi(s, a) = E_pi[G_t | S_t = s, A_t = a]`

`V^pi` tells us how good a state is under the policy. `Q^pi` tells us how good a particular action is in that state.

## Why the MDP Formalism Matters for Code

Every later file in this repository is implementing one of two tasks:

- prediction: estimate `V^pi` or `Q^pi`,
- control: improve the policy toward optimality.

The environment class in [gridworld.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/common/gridworld.py) is a concrete MDP:

- states are grid coordinates,
- actions are `UP`, `DOWN`, `LEFT`, `RIGHT`,
- transitions are deterministic,
- reward is attached to the successor state.

## Implementation Insights

- The environment API `reset()`, `step(action)` is a programming realization of the controlled Markov kernel.
- `next_state, reward, done` is enough for standard RL updates precisely because of the Markov assumption.
- Once the state is Markov, it becomes meaningful to store values indexed by `state` or `(state, action)`.

## Bridge to Chapter 3

The next mathematical step is to express values recursively. Once we define return as a discounted sum of future rewards, we can derive Bellman equations. Those equations are the backbone of every algorithm in chapters 4 through 9.
