# ❄️ FrozenLake Q-Learning Agent

This project implements a simple **Q-learning agent** for the classic `FrozenLake-v1` environment from [Gymnasium](https://gymnasium.farama.org/).  
It demonstrates how tabular reinforcement learning can be applied to a discrete gridworld problem.

---

## 📌 Project Overview

- **Environment:** `FrozenLake-v1` (4x4 grid, non-slippery)
- **Algorithm:** Tabular Q-learning
- **Training:** Epsilon-greedy exploration with exponential decay
- **Evaluation:** Agent tested over 10 episodes
- **Extras:** Video recording using `RecordVideo` wrapper

---

## 🎯 Objectives

- Learn to apply Q-learning on discrete environments
- Understand Q-table updates via the Bellman equation
- Balance exploration and exploitation using epsilon decay
- Use `gymnasium` for training and evaluation
- Record and visualize agent behavior

---

## 📁 Files

 - `main.py`    | Trains the Q-learning agent and saves the resulting Q-table 
 - `testing.py` | Loads the saved Q-table and evaluates the agent over multiple episodes, recording videos 
 - `q_values.npy` | (auto-generated) Saved Q-table in NumPy format 
 - `FrozenLake-agent/` | (auto-generated) Contains recorded videos of the agent in action 

---

## Note

 In this example, I used a gymnasium wrapper `RecordVideo` to record my agent and tqdm to create a training progress bar.
 Both are optional and can be removed if wanted.
