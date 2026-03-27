# Chapter 9: Policy Gradient and Actor-Critic

## Why Move Beyond Value-Based Control

Value-based methods choose actions indirectly by learning a value function and then taking an argmax. Policy-gradient methods optimize the policy directly. This is attractive when:

- stochastic policies are desirable,
- the action space structure favors parameterized distributions,
- direct control optimization is simpler than value maximization.

## Objective

Let the policy be `pi_theta(a | s)`. The standard episodic objective is

`J(theta) = E_{tau ~ pi_theta}[G_0]`

where `tau` is a full trajectory sampled from the policy.

The key question is how to differentiate an expectation over trajectories that themselves depend on `theta`.

## Log-Derivative Trick

The identity

`grad_theta pi_theta(a | s) = pi_theta(a | s) grad_theta log pi_theta(a | s)`

implies

`grad_theta J(theta) = E[sum_t grad_theta log pi_theta(A_t | S_t) G_t]`

This is the foundation of REINFORCE.

## Simple Policy Gradient

[simple_pg.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch09/simple_pg.py) uses the full episode return `G_0` for every log-probability term:

`sum_t grad log pi_theta(A_t | S_t) G_0`

This estimator is valid but high variance because it assigns the same total outcome to all actions in the episode.

## REINFORCE

[reinforce.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch09/reinforce.py) uses reward-to-go:

`sum_t grad log pi_theta(A_t | S_t) G_t`

This removes rewards that occurred before action `A_t`, which that action could not possibly influence. The estimator remains unbiased and often has lower variance.

## Baselines and Advantage

Subtracting a baseline `b(s)` does not bias the policy gradient as long as the baseline does not depend on the chosen action:

`E[grad log pi(a | s) * b(s)] = 0`

A particularly useful choice is `b(s) = V(s)`, which leads to the advantage:

`A(s, a) = Q(s, a) - V(s)`

In one-step form, a common estimator is the TD error

`delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)`

## Actor-Critic

[actor_critic.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/ch09/actor_critic.py) combines:

- actor: the policy network,
- critic: the value network.

The critic is trained with a Bellman-style squared loss:

`L_V = (V(s) - [r + gamma V(s')])^2`

The actor is trained with

`L_pi = -log pi(a | s) * delta`

where `delta` is used as an advantage-like signal.

This drastically reduces variance compared with pure REINFORCE because the critic supplies a learned baseline and one-step target.

## Bias-Variance Picture

- simple policy gradient: unbiased, very high variance,
- REINFORCE with reward-to-go: unbiased, somewhat lower variance,
- actor-critic: lower variance, but now the critic introduces approximation bias.

This mirrors the Monte Carlo versus TD tradeoff seen earlier, but now on the policy side.

## Stochastic Policies

The policy network outputs probabilities via softmax. This is mathematically natural because:

- probabilities are nonnegative,
- probabilities sum to 1,
- gradients with respect to logits are well behaved.

Sampling from the categorical distribution is essential. If we only chose `argmax`, the policy would stop being differentiably stochastic in the sense needed by the score-function estimator.

## Implementation Insights

- The stored `prob[action]` term is the scalar `pi_theta(a_t | s_t)` needed inside `log pi_theta(a_t | s_t)`.
- The actor loss must not backpropagate through the critic target; that is why the advantage-like signal is detached.
- Actor-critic performs online-style updates every step, unlike REINFORCE which waits for the full episode.
- Policy optimization is often more numerically delicate than supervised classification because the loss weighting signal comes from noisy returns or critic estimates.

## PyTorch Parallels

The corresponding PyTorch implementations live in:

- [simple_pg.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/pytorch/simple_pg.py)
- [reinforce.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/pytorch/reinforce.py)
- [actor_critic.py](/Users/HyeongjinKim/OneDrive/Desktop/EmbodiedRL_Lab/deep-learning-from-scratch-4/pytorch/actor_critic.py)

Reading both versions side by side is useful because the mathematical structure is identical, while the autodiff API differs.

## Broader Perspective

Policy gradients are the entry point to modern RL families such as PPO, TRPO, A2C/A3C, SAC, and many continuous-control methods. Conceptually, chapter 9 marks the shift from value-estimation-first thinking to direct optimization of behavior distributions.
