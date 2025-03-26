import gymnasium as gym
import numpy as np

Q = np.load("q_values.npy")
print("Q values loaded.")

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human")

total_reward = 0

for episode in range(10):
    done = False
    state, _ = env.reset()

    while not done:
        action = np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state



print(f"total reward: {total_reward}")