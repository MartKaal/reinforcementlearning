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
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995

# Define epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: select a random action
    else:
        return np.argmax(Q[state])  # Exploit: select the action with max value

# Q-learning algorithm
def q_learning(env, num_episodes, learning_rate, discount_factor, initial_epsilon, min_epsilon, epsilon_decay):
    rewards_per_episode = []
    epsilon = initial_epsilon

    for episode in range(num_episodes):
        state, info = env.reset(seed=42)
        state = str(state)
        terminated, truncated = False, False
        total_reward = 0
        step_count = 0

        while not terminated and not truncated:
            action = epsilon_greedy_policy(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = str(next_state)
            
            # Update Q-value
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += learning_rate * td_error
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if step_count > 1000:
                print(f"Episode {episode} exceeded 1000 steps, breaking loop.")
                break

        if epsilon > min_epsilon:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode} ended after {step_count} steps with total reward: {total_reward}")

    return rewards_per_episode

# Main execution
num_episodes = 200
rewards_per_episode = q_learning(env, num_episodes, LEARNING_RATE, DISCOUNT_FACTOR, INITIAL_EPSILON, MIN_EPSILON, EPSILON_DECAY)

env.close()

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(range(num_episodes), rewards_per_episode)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode over Time')
plt.show()