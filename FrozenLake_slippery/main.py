import gymnasium as gym
import numpy as np
from tqdm import tqdm

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

# Q-values
Q = np.zeros((num_states, num_actions))

#Hyperparameters
alpha = 0.8
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.1
decay = 0.99992
episodes = 30000

for episode in tqdm(range(episodes), desc="Episodes completed"):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Q update

        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

    # decaying epsilon
    if epsilon > min_epsilon:
        epsilon *= decay

print('training completed')

# testing agent

np.save("q_values.npy", Q)
print("Q values saved.")
print(Q)
