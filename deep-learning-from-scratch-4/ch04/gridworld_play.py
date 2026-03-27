"""Render a random value function on the GridWorld for inspection."""

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from common.gridworld import GridWorld


env = GridWorld()
V = {}
for state in env.states():
    # This file is only for visual familiarization with the environment.
    # Random values make it easy to see how each state is positioned.
    V[state] = np.random.randn()
env.render_v(V)
