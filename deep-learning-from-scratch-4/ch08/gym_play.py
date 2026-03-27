"""Play CartPole with random actions to inspect the raw environment."""

import gym
import numpy as np


env = gym.make("CartPole-v0")
state = env.reset()
done = False

while not done:
    env.render()
    # A random policy is a useful sanity check before implementing a
    # learning algorithm, because it reveals the observation and action
    # loop without conflating it with training logic.
    action = np.random.choice([0, 1])
    next_state, reward, done, info = env.step(action)
env.close()
