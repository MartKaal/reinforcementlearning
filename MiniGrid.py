import gymnasium as gym
import numpy as np

# Returns a random action that are available in the enviornment
def random_policy(observation):
    action_space = env.action_space
    return action_space.sample()

# Create the environment
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
observation, info = env.reset()

# Run the environment with the policy
for _ in range(1000):
    action = random_policy(observation) 
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
