import gymnasium as gym
import numpy as np
from tqdm import tqdm

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
num_states = env.observation_space.n
num_actions = env.action_space.n

# Q-values
q_values = np.zeros((num_states, num_actions))

#Hyperparameters
alpha = 0.8
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.1
decay = 0.998
episodes = 1000

for episode in tqdm(range(episodes), desc="Episodes completed"):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values[state, :])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q update
        q_values[state, action] = q_values[state, action] + alpha * (reward + gamma * np.max(q_values[next_state, :]) - q_values[state, action])

        state = next_state

    # decaying epsilon
    if epsilon > min_epsilon:
        epsilon *= decay

print('training completed')

np.save("q_values.npy", q_values)
print("Q values saved.")
print(q_values)
