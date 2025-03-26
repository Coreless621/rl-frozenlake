# 🧊 FrozenLake-v1 (Slippery) – Q-Learning Agent

This project trains a **tabular Q-learning agent** on the **slippery** variant of the `FrozenLake-v1` environment from [Gymnasium](https://gymnasium.farama.org/).  
The goal is to teach an agent to navigate across a frozen lake without falling into holes — despite the randomness of slippery movement.

---

## 📌 Project Overview

- **Environment:** `FrozenLake-v1` (4x4 grid, non-slippery)
- **Algorithm:** Tabular Q-learning
- **Training:** Epsilon-greedy exploration with exponential decay
- **Evaluation:** Agent tested over 10 episodes

---

## 🎯 Objectives

- Learn to apply Q-learning on discrete environments
- Understand Q-table updates via the Bellman equation
- Balance exploration and exploitation using epsilon decay
- Use `gymnasium` for training and evaluation
- Understand the influence of stochasticity on training

---

## 📂 Files

- `main.py`    | Trains the Q-learning agent on the slippery FrozenLake environment and saves the learned Q-table (`q_values.npy`) 
- `testing.py` | Loads the saved Q-table and runs 10 evaluation episodes using the greedy policy 
- `q_values.npy` | (Auto-generated) The learned Q-table as a NumPy array 

## Note

  In this example I used tqdm to create a training progress bar. It is completely optional and not needed to train your agent.
  It is just nice to see progress going, although here training is typically fast.
