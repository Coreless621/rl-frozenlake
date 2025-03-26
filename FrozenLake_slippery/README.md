# 🧊 FrozenLake-v1 (Slippery) – Q-Learning Agent

This project trains a **tabular Q-learning agent** on the **slippery** variant of the `FrozenLake-v1` environment from [Gymnasium](https://gymnasium.farama.org/).  
The goal is to teach an agent to navigate across a frozen lake without falling into holes — despite the randomness of slippery movement.

---

## ❄️ Environment Setup

- **Environment:** `FrozenLake-v1`
- **Map:** `4x4`
- **Slippery:** `True` (stochastic transitions)
- **Observation space:** 16 discrete states
- **Action space:** 4 discrete actions (left, down, right, up)

---

## 🧠 Algorithm

- **Type:** Tabular Q-learning
- **Exploration:** Epsilon-greedy with exponential decay
- **Updates:** Bellman equation-based Q-value updates
- **Goal:** Maximize expected cumulative reward

---

## ⚙️ Hyperparameters

- Learning rate   | `alpha = 0.8` 
- Discount factor | `gamma = 0.99` 
-Initial epsilon | `1.0` 
- Min epsilon     | `0.1` 
- Epsilon decay   | `0.99992` 
- Episodes        | `30,000` 

---

## 📂 Project Structure

- `main.py`    | Trains the Q-learning agent on the slippery FrozenLake environment and saves the learned Q-table (`q_values.npy`) 
- `testing.py` | Loads the saved Q-table and runs 10 evaluation episodes using the greedy policy 
- `q_values.npy` | (Auto-generated) The learned Q-table as a NumPy array 
