# Chapter 4: Dynamic Programming

## Core Assumption

Dynamic programming assumes the full MDP model is known:

- transition dynamics `p(s', r | s, a)` are available,
- rewards are known,
- all states can be enumerated.

This assumption is strong, but it lets us compute exact expectations rather than estimating them from sampled trajectories.

## Policy Evaluation

For a fixed policy `pi`, the Bellman expectation equation is

`V^pi(s) = sum_a pi(a | s) sum_{s', r} p(s', r | s, a) [r + gamma V^pi(s')]`

The function [policy_eval.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/policy_eval.py) implements this by repeatedly sweeping over all states until the values stop changing by more than a threshold.

The convergence test is

`delta = max_s |V_{new}(s) - V_{old}(s)|`

and iteration stops when `delta < threshold`.

## Two-State Example

[dp.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/dp.py) and [dp_inplace.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/dp_inplace.py) are miniature Bellman fixed-point solvers. They make the recursion visible without the distraction of a larger environment.

The main mathematical insight is that iterative Bellman updates are fixed-point iterations. We are not solving a linear system directly; we are repeatedly applying a contraction map until the values stabilize.

## Policy Improvement

Once `V^pi` is known, a greedy policy with respect to that value is

`pi'(s) in argmax_a sum_{s', r} p(s', r | s, a) [r + gamma V^pi(s')]`

This is what [policy_iter.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/policy_iter.py) does in `greedy_policy`.

The policy improvement theorem guarantees

`V^{pi'}(s) >= V^pi(s)` for all `s`

so repeated improvement steps cannot make the policy worse.

## Policy Iteration

Policy iteration alternates:

1. policy evaluation,
2. greedy policy improvement.

When the greedy improvement step no longer changes the policy, the policy is optimal.

Mathematically, policy iteration is exact because each evaluation step solves for `V^pi` closely enough before improvement.

## Value Iteration

Value iteration compresses evaluation and improvement into one update:

`V_{k+1}(s) = max_a sum_{s', r} p(s', r | s, a) [r + gamma V_k(s')]`

This is the Bellman optimality operator. [value_iter.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch04/value_iter.py) applies it repeatedly until convergence, then extracts a greedy policy.

Value iteration is usually cheaper than full policy iteration because it avoids fully evaluating intermediate policies.

## Why In-Place Updates Can Be Faster

Suppose a sweep has already improved the value of one state. An in-place update lets the next state immediately use that improved value. In linear-system language, this resembles Gauss-Seidel iteration and often speeds convergence compared with synchronous Jacobi-style updates.

## GridWorld Structure

The environment in [gridworld.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/common/gridworld.py) is deterministic, so the inner expectation over `s'` collapses to a single successor. That is why the code often looks like

`value = reward + gamma * V[next_state]`

instead of summing over many successor states.

## Implementation Insights

- Terminal states are assigned value `0` because there is no discounted future after termination in this setup.
- The reward is attached to the successor state, so the Bellman backup is naturally coded as `r + gamma * V[next_state]`.
- The rendering utilities make policy improvement interpretable: you can see how a value field induces greedy arrows.
- Dynamic programming is computationally feasible only because the state space here is tiny and enumerable.

## Limitations

DP breaks down when:

- the model is unknown,
- the state space is too large to sweep exactly,
- transitions are continuous or high-dimensional.

Those limitations motivate Monte Carlo and temporal-difference methods in the next chapters.
