"""GridWorld environment used throughout the tabular RL chapters.

The environment is intentionally tiny so that dynamic programming,
Monte Carlo, and temporal-difference methods can be inspected exactly.
Each state is represented as a `(row, col)` tuple on a 3 x 4 grid.
"""

import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    """Deterministic grid navigation environment.

    Action indexing convention:
    - 0: up
    - 1: down
    - 2: left
    - 3: right

    The reward is attached to the *next* state, which matches the
    Bellman-style notation r(s, a, s').
    """

    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        # reward_map[y, x] stores the immediate reward obtained when the
        # agent arrives at that cell. `None` marks the wall cell.
        self.reward_map = np.array(
            [
                [0, 0, 0, 1.0],
                [0, None, 0, -1.0],
                [0, 0, 0, 0],
            ]
        )
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        """Number of rows in the grid."""

        return len(self.reward_map)

    @property
    def width(self):
        """Number of columns in the grid."""

        return len(self.reward_map[0])

    @property
    def shape(self):
        """Convenience accessor used by renderers and one-hot encoders."""

        return self.reward_map.shape

    def actions(self):
        """Return the discrete action set."""

        return self.action_space

    def states(self):
        """Yield every coordinate in row-major order.

        The wall state is yielded as well so higher-level algorithms can
        explicitly decide how they want to handle it.
        """

        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        """Apply a deterministic transition model.

        If the proposed move exits the grid or hits the wall, the agent
        stays in place. This "bounce back" rule is common in textbook
        GridWorld examples because it keeps the transition function simple.
        """

        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        """Return the immediate reward r(s, a, s')."""

        del state, action
        return self.reward_map[next_state]

    def reset(self):
        """Reset the episode to the fixed start state."""

        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        """Advance the environment by one step.

        Returns:
            next_state: state after applying the action
            reward: immediate reward collected at next_state
            done: whether the goal state has been reached
        """

        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        """Visualize a state-value function and optionally a policy."""

        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        """Visualize an action-value function."""

        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_q(q, print_value)
