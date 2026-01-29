# DAVID-GRPO

DAVID-GRPO is a **budget-efficient reinforcement learning framework** that enables **small language model agents** to perform **grounded multi-hop reasoning** under strict compute constraints.

Contrary to the common belief that multi-hop reasoning requires large models and massive rollouts, DAVID-GRPO demonstrates that **with the right inductive biases**, small agents can achieve **high accuracy at low training cost**.

This repository provides the official implementation of DAVID-GRPO.

---

## 🚀 Key Idea

Existing RL-based reasoning agents work well only in **high-cost, high-accuracy regimes**, relying on:
- large models,
- massive rollout budgets,
- dense on-policy exploration.

Under realistic resource constraints, small agents suffer from:
1. **Cold-start instability**
2. **Sparse and delayed rewards**
3. **Inefficient exploration**

DAVID-GRPO addresses all three issues **simultaneously** with a simple but effective design.

---

## 🧠 Core Components

### 1. Few-Shot Warm-Start (Cold-Start Stabilization)
- Uses **only a handful of expert trajectories** (e.g., 4 examples)
- Combines **off-policy expert trajectories** with **on-policy rollouts**
- Prevents early policy collapse without large-scale supervision

### 2. Grounded Retrieval Reward (Dense Credit Assignment)
- Rewards **evidence recall**, not just final answer correctness
- Computes recall over the **union of retrieved documents** across all steps
- Encourages faithful multi-hop retrieval instead of parametric shortcuts

### 3. Grounded Expansion (Efficient Exploration)
- Identifies **near-miss trajectories**
- Truncates at the last grounded step and **resamples continuations**
- Improves sample efficiency without additional full rollouts

---

## 🧪 Experimental Highlights

- Trained on **4× RTX 3090 GPUs**
- Models up to **1.5B parameters**
- Uses only **~4.7% of the rollout budget** of high-cost baselines
- Consistently outperforms prior RL methods on **6 multi-hop QA benchmarks**

Benchmarks include:
- HotpotQA
- 2WikiMultiHopQA
- MuSiQue
- Bamboogle / BamTwoogle
- AntiLeakBench (multi-hop)

---

## 📊 Why DAVID-GRPO Works

DAVID-GRPO reinterprets RL failures in low-budget settings through the lens of **information retrieval**:

| RL Problem | IR Analogy | DAVID-GRPO Solution |
|-----------|-----------|--------------------|
| Cold-start | Zero-shot retrieval | Pseudo-positive warm-start |
| Sparse rewards | Relevance judgments | Grounded retrieval reward |
| Poor exploration | Bounded recall | Adaptive grounded expansion |

This alignment leads to **stable training**, **faithful retrieval**, and **robust multi-hop reasoning**.

---

## 🛠️ Repository Structure (Planned)

```text
david-grpo/
├── grpo/                 # GRPO and mixed off-/on-policy training
├── rewards/              # Grounded retrieval reward
├── expansion/            # Grounded expansion module
├── retriever/            # FAISS + dense retriever setup
├── prompts/              # Agent prompts
├── experiments/          # Training & evaluation scripts
├── configs/              # Hyperparameter configs
└── README.md
