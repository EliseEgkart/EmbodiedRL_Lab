"""Synchronous policy evaluation on a two-state Markov reward process."""

# V is updated from a frozen copy of the previous iterate. This is the
# textbook Jacobi-style dynamic programming update.
V = {"L1": 0.0, "L2": 0.0}
new_V = V.copy()

cnt = 0
while True:
    # Bellman expectation backup for each state:
    # V(s) = sum_a pi(a|s) sum_{s', r} p(s', r | s, a) [r + gamma V(s')]
    new_V["L1"] = 0.5 * (-1 + 0.9 * V["L1"]) + 0.5 * (1 + 0.9 * V["L2"])
    new_V["L2"] = 0.5 * (0 + 0.9 * V["L1"]) + 0.5 * (-1 + 0.9 * V["L2"])

    # Convergence is monitored by the sup norm ||V_{k+1} - V_k||_inf.
    delta = abs(new_V["L1"] - V["L1"])
    delta = max(delta, abs(new_V["L2"] - V["L2"]))
    V = new_V.copy()

    cnt += 1
    if delta < 0.0001:
        print(V)
        print(cnt)
        break
