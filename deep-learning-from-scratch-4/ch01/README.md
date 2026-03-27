# Chapter 1: Bandit Mathematics

## Overview

`ch01` covers the simplest reinforcement-learning setting: there is no state transition graph, only repeated action selection under uncertainty. The mathematical point is to isolate the exploration-exploitation tradeoff before introducing Markov decision processes.

## Core Random Variables

- Let `A_t` be the action chosen at step `t`.
- Let `R_t` be the reward observed after choosing `A_t`.
- In a Bernoulli bandit, each arm `a` has an unknown success probability `q_*(a)`, and `R_t in {0, 1}` with `P(R_t = 1 | A_t = a) = q_*(a)`.
- The learning problem is to estimate `q_*(a)` from samples and use those estimates to choose better actions.

## Expected Reward and Action Value

The true action value is

`q_*(a) = E[R_t | A_t = a]`

The sample-average estimate after `n` pulls of arm `a` is

`Q_n(a) = (1 / n) * sum_{i=1}^n R_i(a)`

This is an unbiased estimator under stationarity. The implementation in [bandit.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch01/bandit.py) updates this estimate incrementally:

`Q_n(a) = Q_{n-1}(a) + (1 / n) * (R_n - Q_{n-1}(a))`

This matters because RL almost always relies on prediction errors of the form

`target - current_estimate`

and chapter 1 is the first place that algebraic pattern appears.

## Why the Incremental Formula Works

Start from

`Q_n = (R_1 + ... + R_{n-1} + R_n) / n`

Using

`Q_{n-1} = (R_1 + ... + R_{n-1}) / (n - 1)`

we obtain

`Q_n = Q_{n-1} + (R_n - Q_{n-1}) / n`

This turns an `O(n)` memory estimator into an `O(1)` memory estimator. The same pattern reappears later as TD updates, Q-learning updates, and stochastic-gradient steps.

## Exploration vs Exploitation

If the agent always chooses `argmax_a Q(a)`, it may lock into a suboptimal arm because early estimates are noisy. To prevent this, the code uses epsilon-greedy:

- With probability `epsilon`, choose a random action.
- With probability `1 - epsilon`, choose the current best action.

If the action set has size `k`, the induced policy is

`pi(a | s) = epsilon / k` for non-greedy actions,

and

`pi(a_greedy | s) = 1 - epsilon + epsilon / k`

Even though there is no explicit state here, this is already a policy over actions.

## Regret Perspective

An important theoretical quantity is cumulative regret:

`Regret(T) = sum_{t=1}^T [q_*(a^*) - q_*(A_t)]`

where `a^*` is the optimal arm. The code does not compute regret explicitly, but the cumulative reward plots in [bandit.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch01/bandit.py) and the averaged curves in [bandit_avg.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch01/bandit_avg.py) are empirical proxies for how well the agent is minimizing regret.

## Stationary vs Non-Stationary Bandits

In stationary problems, old data remains valid forever, so sample averages are natural. In non-stationary problems, old data becomes stale. Then the update

`Q <- Q + alpha * (R - Q)`

is preferable, where `alpha` is constant.

Expanding the recursion shows

`Q_t = (1 - alpha)^t Q_0 + sum_{i=1}^t alpha (1 - alpha)^{t-i} R_i`

So recent rewards receive exponentially larger weight. This is why [non_stationary.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch01/non_stationary.py) compares sample-average updates against constant-step updates.

## Variance and Multi-Run Averaging

Bandit experiments are noisy because:

- arm means are randomly initialized,
- rewards are stochastic,
- exploration adds randomness.

Therefore [bandit_avg.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch01/bandit_avg.py) averages across many independent runs. This is statistically important because a single run can be misleading.

## Implementation Insights

- `Qs[action] += (reward - Qs[action]) / ns[action]` is not just a coding trick; it is the prototype for nearly every later RL update.
- `rates.append(total_reward / (step + 1))` estimates the empirical expected reward of the entire learning process, not just of one arm.
- In the non-stationary setting, the agent with constant `alpha` is better because it intentionally forgets.
- Chapter 1 teaches the central engineering lesson that estimators must match the data-generating process. If the environment drifts, an estimator that remembers everything equally is mathematically mismatched.

## Connection to Later Chapters

- Chapter 4 generalizes action values from bandit arms to state-action pairs in an MDP.
- Chapter 5 replaces immediate reward averaging with full-return averaging.
- Chapter 6 replaces Monte Carlo returns with one-step bootstrapped targets.
- Chapters 7 to 9 replace tables with differentiable function approximators and policies.
