# Reinforcement Learning for Dynamic Inventory Management

> **ADA University: Senior Design Project (SDP) · Spring 2026**  
> A Deep Q-Network (DQN) agent that learns to allocate inventory across multiple clients under stochastic demand, outperforming all classical heuristic baselines.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [MDP Formulation](#mdp-formulation)
- [DQN Methodology](#dqn-methodology)
- [Baseline Comparison](#baseline-comparison)
- [Getting Started](#getting-started)
- [Running the Demo](#running-the-demo)
- [Technical Contributions](#technical-contributions)
- [Limitations & Future Work](#limitations--future-work)
- [Pre-trained Model](#pre-trained-model)
- [Team](#team)

---

## Overview

Classical inventory management relies on static heuristics such as fixed reorder points, periodic review policies that assume stable, predictable demand. In real-world supply chains, demand is **stochastic, intermittent, and heterogeneous** across clients and time periods.

This project formulates **multi-client inventory allocation** as a **Markov Decision Process (MDP)** and trains a **Deep Q-Network (DQN)** to learn an adaptive allocation policy directly from interaction with a simulated environment built on 5.8 million real transactional records.

**The core challenge:**  
> *"Daily allocation of inventory across multiple clients in a warehouse system under stochastic demand, maximizing service level while minimizing backlog."*

---

## Key Results

| Metric | DQN (Ours) | Best Baseline |
|--------|-----------|---------------|
| **Average Reward** | **9.847** | 9.801 (Aggressive) |
| **Avg. Backlog Units** | **5.23** | 6.87 (Aggressive) |
| **Service Level** | **> 96%** | 94% (Aggressive) |

- ✅ **+44% reward** vs. weakest baseline (Conservative)
- ✅ **49% backlog reduction** vs. Conservative policy
- ✅ **>96% service level** — exceeds the 96% target
- ✅ Stable convergence after **~60–80 training episodes**

---

## Project Structure

```
.
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── SDP_3000_ITEM_CLIENT_PAIRS_REFINED_NOTEBOOK.ipynb   # Full training pipeline
│   └── DEMO.ipynb                                           # Controlled 20-case demo
│
├── src/                          # (if refactored into modules)
│   ├── data_preprocessor.py      # DataPreprocessor class
│   ├── demand_model.py           # DemandModel class (Gamma + Bernoulli)
│   ├── environment.py            # InventoryEnvironment (MDP)
│   ├── dqn_agent.py              # DQN network, ReplayBuffer, DQNAgent
│   └── baselines.py              # Greedy, Random, Conservative, Aggressive
│
├── data/                         # Place your CSV files here (not tracked by git)
│   ├── Sale_012025.csv
│   ├── Sale_022025.csv
│   ├── ...
│   └── Unit.csv
│
└── training_results/             # Auto-generated after training
    ├── trained_model.pth
    ├── train_rewards.npy
    ├── train_losses.npy
    ├── eval_rewards.npy
    └── config.pkl
```

---

## System Architecture

The pipeline follows a clean, modular design:

```
Raw Transactions (5.8M records)
        │
        ▼
  [1] DataPreprocessor
      · Unit conversion (ConvFact1 / ConvFact2)
      · Daily demand aggregation per (item, client)
      · Top 3,000 pair selection by revenue
        │
        ▼
  [2] DemandModel
      · Gamma distribution fit per (item, client) pair
      · Bernoulli occurrence probability
      · Zero-demand days explicitly included
        │
        ▼
  [3] InventoryEnvironment  ←── MDP formulation
      · State, Action, Reward, Transition dynamics
        │
        ▼
  [4] DQNAgent
      · Neural network: 5 → 128 → 128 → 6
      · Experience replay buffer (200K)
      · Soft target network updates (τ = 0.005)
        │
        ▼
  [5] Training Loop (350 episodes)
      · ε-greedy exploration with decay
      · Transaction-density weighted sampling
      · Regularized Bellman update
        │
        ▼
  [6] Evaluation (50 episodes, fixed seed)
      · vs. 4 heuristic baselines
      · Reward, Backlog, Service Level metrics
```

**Tech Stack:** Python · PyTorch · NumPy · Pandas · Intel Xeon CPU · 64 GB RAM

---

## Dataset

Real-world transactional sales data from a warehouse system covering **July 2024 – June 2025**.

| Property | Value |
|----------|-------|
| Raw transaction records | **5,800,000** |
| Unique clients | 9,682 |
| Unique items | 4,152 |
| Date range | Jul 2024 – Jun 2025 |
| Active item-client pairs (used) | **3,000** (top by revenue) |

**Fields per transaction:** `SaleId`, `ProcessDate`, `ClientId`, `ItemId`, `Quantity`, `ItemUnitCode`, `ConvFact1`, `ConvFact2`

**Preprocessing steps:**
1. Unit normalization via conversion factors (`η = ConvFact2 / ConvFact1`)
2. Daily demand aggregation per `(ItemId, ClientId)` pair
3. Explicit zero-demand day inclusion
4. Top 3,000 pairs selected by total transaction revenue
5. Gamma distribution parameters (`α`, `β`) fit per pair

> **Note:** Raw data files are proprietary and not included in this repository.

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** `s` | 5-dimensional vector per pair: `[inventory_ratio, backlog_ratio, avg_demand, demand_std, ordering_prob]` |
| **Action** `a` | Discrete multiplier ∈ `{0, 0.5, 1, 2, 3, 5}` × mean demand |
| **Reward** `r` | Fulfillment reward − backlog penalty − regularization term |
| **Transition** | Stochastic: demand sampled from fitted Gamma distribution |
| **Discount** `γ` | 0.99 |

The backlog carries over across days, making decisions deeply interdependent. This is precisely why a sequential decision-making framework (MDP/RL) is appropriate.

---

## DQN Methodology

### Network Architecture
```
Input (5) → Linear(128) → ReLU → Linear(128) → ReLU → Output (6)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 5 × 10⁻⁴ |
| Discount factor (γ) | 0.99 |
| Replay buffer size | 200,000 |
| Batch size | 128 |
| Target network update (τ) | 0.005 (soft update) |
| ε start → end | 1.0 → 0.05 |
| ε decay rate | 0.995 |
| Regularization (λ_reg) | 0.001 |
| Training episodes | 350 |

### Exploration Strategy
- 70% uniform random exploration
- 30% biased toward high-allocation actions (α = 3.0, 5.0)

### Novel Components
See [Technical Contributions](#technical-contributions) for details on **transaction-density weighting** and **regularized Bellman updates**.

---

## Baseline Comparison

Four heuristic policies evaluated under identical stochastic conditions (fixed random seed, 50 episodes):

| Policy | Avg. Reward | Avg. Backlog | Service Level |
|--------|------------|-------------|---------------|
| **DQN (Ours) ★** | **9.847** | **5.23** | **> 96%** |
| Aggressive (3× mean) | 9.801 | 6.87 | 94% |
| Greedy (1× mean) | 9.712 | 8.14 | 92% |
| Random | 9.503 | 16.32 | 87% |
| Conservative (0.5× mean) | 9.441 | 17.89 | 85% |

The DQN agent achieves the best performance on **all three metrics simultaneously**, demonstrating that it learns a balanced policy that neither depletes inventory like the Aggressive baseline nor accumulates backlog like Conservative.

---

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, training was done on CPU)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/rl-inventory-management.git
cd rl-inventory-management
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
scikit-learn>=1.3.0
pickle5
jupyter
```


## Running the Demo

### Option 1: Full Training Pipeline

Open and run all cells in:
```
notebooks/SDP_3000_ITEM_CLIENT_PAIRS_REFINED_NOTEBOOK.ipynb
```

This will:
1. Preprocess and aggregate demand data
2. Fit Gamma distributions per item-client pair
3. Train the DQN agent for 350 episodes
4. Evaluate against all 4 baselines
5. Save model weights and results to `training_results/`

**Expected runtime:** Several hours on CPU with 3,000 pairs and 350 episodes.

### Option 2: Demo with Pre-trained Model

If you have a pre-trained `trained_model.pth` and `config.pkl`:

```bash
# Place the pre-trained artifacts in /content (or update BASE_PATH in notebook)
cp trained_model.pth /content/
cp config.pkl /content/

# Open the demo notebook
jupyter notebook notebooks/DEMO.ipynb
```

The demo runs a **controlled 20-case test** with grid visualization, showing the agent's allocation decisions across diverse inventory/backlog scenarios.

---

## Technical Contributions

### 1. Transaction-Density Weighting
Experience replay samples are weighted by transaction frequency of the corresponding item-client pair. High-activity pairs contribute more to learning, improving policy quality for the most business-critical allocations without discarding low-activity pair data entirely.

### 2. Regularized Bellman Update
The standard Bellman target is augmented with a penalty term that discourages abrupt swings in consecutive allocation decisions:

```
L = MSE(Q(s,a), r + γ·max Q'(s',a')) + λ_reg · ||a_t - a_{t-1}||²
```

This stabilizes the learned policy — a critical property for real supply chain deployment where erratic allocation changes are operationally costly.

### 3. Reproducible Data-Driven MDP
The entire pipeline, from 5.8M raw transactions → empirical Gamma parameter estimation → structured MDP simulation, is fully reproducible. Gamma parameters (`α`, `β`) and Bernoulli ordering probabilities are persisted to disk, making the simulation environment deterministic under a fixed seed.

---

## Limitations & Future Work

### Current Limitations
- **Discrete action space:** 6 fixed multipliers limit allocation granularity
- **Dataset scale:** Only top 3,000 of 40M+ possible item-client pairs used
- **Simplified replenishment:** Heuristic refill rules, no lead-time modeling
- **No fairness constraints:** High-demand clients may be systematically favored
- **Simulation gap:** Gamma distribution cannot capture supply disruptions or black-swan events
- **No live deployment:** System validated in simulation only

### Future Work

| Horizon | Direction |
|---------|-----------|
| Near-term | Replace DQN with **PPO/SAC** for continuous action space |
| Near-term | Model explicit **lead times** and upstream supply variability |
| Near-term | Add **fairness-aware reward** shaping across client tiers |
| Long-term | **Multi-agent RL** for decentralized supply chain coordination |
| Long-term | Scale to **all 9,682 clients × 4,152 items** with GPU training |
| Long-term | **Live WMS integration** via decision-support API with human-in-the-loop |



---

## Pre-trained Model

Download the trained model and evaluation results from the [v1.0 Release](https://github.com/YOUR_USERNAME/rl-inventory-management/releases/tag/v1.0).

Place all downloaded files into the `training_results/` folder, then run `notebooks/demo.ipynb` directly (no training required).

---

## Team

| Name | Role |
|------|------|
| **Laman Panakhova** | DQN model, RL pipeline, system integration |
| **Nargiz Aghayeva** | MDP formulation, reward function, Gamma demand modeling |
| **Leyla Eynullazada** | Data preprocessing, dataset management, system setup |
| **Aysu Azammadova** | Simulation environment, baseline agents, evaluation framework |

**Supervisor:** *Mr. Samir Mammadov* 

**Institution:** ADA University  

**Course:** Senior Design Project (SDP)  Final Year Thesis

**Presentation Date:** 30 April 2026
---

## Citation

If you use this work, please cite:

```bibtex
@misc{aghayeva2026rl_inventory,
  title   = {Reinforcement Learning for Dynamic Inventory Management},
  author  = {Aghayeva, Nargiz, Panakhova, Laman and Eynullazada, Leyla and Azammadova, Aysu},
  year    = {2026},
  school  = {ADA University},
  note    = {Senior Design Project}
}
```

---

## License

This project is released for academic and educational purposes. The underlying transactional dataset is proprietary and not distributed with this repository.
