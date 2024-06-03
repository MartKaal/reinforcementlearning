import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Create the environment
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")

# Initialize Q-table with default values
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Hyperparameters
learning = 0.1 
discount = 0.99 
epsilon = 1.0 
epsilon_min = 0.1
epsilon_decay = 0.995

# Define epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: select a random action
    else:
        return np.argmax(Q[state])

# Q-learning algorithm
num_episodes = 200
rewards_per_episode = []

for episode in range(num_episodes):
    state, info = env.reset(seed=42)
    state = str(state)
    terminated, truncated = False, False
    total_reward = 0
    step_count = 0  # Step counter to avoid infinite loops

    while not terminated and not truncated:
        action = epsilon_greedy_policy(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = str(next_state)
        
        # Update Q-value
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + discount * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += learning * td_error
        
        state = next_state
        total_reward += reward
        step_count += 1
        
        # Safeguard to prevent infinite loops
        if step_count > 1000:
            print(f"Episode {episode} exceeded 1000 steps, breaking loop.")
            break
        
        if terminated or truncated:
            print(f"Episode {episode} ended after {step_count} steps with total reward: {total_reward}")
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    rewards_per_episode.append(total_reward)

env.close()

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(range(num_episodes), rewards_per_episode)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode over Time')
plt.show()

