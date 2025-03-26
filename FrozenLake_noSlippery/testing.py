import gymnasium as gym
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo

Q = np.load("q_values.npy")
print("Q values loaded.")

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array")
env = RecordVideo(env, video_folder="FrozenLake-agent", name_prefix="eval", episode_trigger=lambda x: True)
total_reward = 0 # accumulated rewards over episodes

for episode in range(10):
    done = False
    state, _ = env.reset()

    while not done:
        action = np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

env.close()
print(f"total reward: {total_reward}")
