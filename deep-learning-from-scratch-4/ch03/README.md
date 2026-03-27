# Chapter 3: Bellman Equations and Optimality

## Note on This Repository

This repository does not currently contain executable `ch03` source files. This README summarizes the mathematical results that make the dynamic-programming and TD chapters possible.

## Recursive Structure of Return

Starting from

`G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...`

we can separate the first reward:

`G_t = R_{t+1} + gamma G_{t+1}`

This identity is the source of Bellman equations. It says that long-horizon value can be decomposed into one-step reward plus discounted continuation value.

## Bellman Expectation Equation for V

For a fixed policy `pi`,

`V^pi(s) = sum_a pi(a | s) sum_{s', r} p(s', r | s, a) [r + gamma V^pi(s')]`

Interpretation:

- average over actions chosen by the policy,
- average over next states and rewards induced by the environment,
- add immediate reward and discounted downstream value.

This is exactly what [policy_eval.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/policy_eval.py) computes.

## Bellman Expectation Equation for Q

Also for a fixed policy,

`Q^pi(s, a) = sum_{s', r} p(s', r | s, a) [r + gamma sum_{a'} pi(a' | s') Q^pi(s', a')]`

This form is crucial when we move from state-value prediction to control, because control fundamentally compares actions.

## Bellman Optimality Equation

If we define the optimal value functions

`V^*(s) = max_pi V^pi(s)`

`Q^*(s, a) = max_pi Q^pi(s, a)`

then they satisfy

`V^*(s) = max_a sum_{s', r} p(s', r | s, a) [r + gamma V^*(s')]`

`Q^*(s, a) = sum_{s', r} p(s', r | s, a) [r + gamma max_{a'} Q^*(s', a')]`

The only algebraic change from policy evaluation to optimal control is replacing the policy expectation by a maximization. But that small change is algorithmically enormous.

## Policy Improvement Theorem

Suppose we have a policy `pi` and define a new greedy policy

`pi'(s) in argmax_a Q^pi(s, a)`

Then

`V^{pi'}(s) >= V^pi(s)` for all `s`

This theorem justifies the improve-evaluate loop used in policy iteration.

## Contraction Mapping Insight

For discounted finite MDPs, the Bellman operator is a contraction in the sup norm:

`||T V - T W||_inf <= gamma ||V - W||_inf`

Because `gamma < 1`, repeated Bellman updates converge to a unique fixed point. This is why iterative updates in [dp.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/dp.py), [policy_eval.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/policy_eval.py), and [value_iter.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/value_iter.py) work.

## Synchronous vs In-Place Updates

There are two common iteration styles:

- synchronous: compute `V_{k+1}` from a frozen `V_k`,
- in-place: overwrite entries immediately as they are updated.

In-place updates often converge faster in practice because later states in a sweep use fresher information. [dp_inplace.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/dp_inplace.py) demonstrates this idea.

## Why Bellman Equations Dominate RL Implementations

Almost every RL update in this repository is a noisy or approximate Bellman backup:

- dynamic programming uses exact expectations,
- Monte Carlo uses sampled full returns,
- TD uses sampled one-step bootstraps,
- DQN fits neural networks to Bellman targets,
- actor-critic uses a Bellman-style TD target for the critic.

## Implementation Insights

- When code computes `reward + gamma * something`, it is almost always implementing a Bellman recursion.
- When that `something` is `V(next_state)` or `max_a Q(next_state, a)`, the algorithm is bootstrapping.
- When convergence is checked by `max_s |V_new(s) - V_old(s)|`, the code is using the sup norm motivated by contraction theory.

## Bridge to Chapter 4

Chapter 4 is where these equations stop being abstract. Dynamic programming makes them executable because the environment model is known exactly.
