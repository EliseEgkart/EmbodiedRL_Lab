"""In-place policy evaluation on a two-state Markov reward process."""

# Here each state update is written back immediately, so later states in
# the same sweep can use fresher values. This is the Gauss-Seidel style
# counterpart to the synchronous update in `dp.py`.
V = {"L1": 0.0, "L2": 0.0}

cnt = 0
while True:
    t = 0.5 * (-1 + 0.9 * V["L1"]) + 0.5 * (1 + 0.9 * V["L2"])
    delta = abs(t - V["L1"])
    V["L1"] = t

    # This update already uses the latest V["L1"], which often accelerates
    # convergence compared with using a fully frozen previous iterate.
    t = 0.5 * (0 + 0.9 * V["L1"]) + 0.5 * (-1 + 0.9 * V["L2"])
    delta = max(delta, abs(t - V["L2"]))
    V["L2"] = t

    cnt += 1
    if delta < 0.0001:
        print(V)
        print(cnt)
        break
