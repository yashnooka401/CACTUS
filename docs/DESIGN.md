# Design Specification for MetaTool-RL

*(A Framework for Meta-Reasoning over Tools in Reinforcement Learning)*

---

## 1. Problem Definition

We formalize the problem as a **sequential decision-making task** under the reinforcement learning (RL) paradigm:

* **Agent:** A policy that selects *subsets of tools* (out of 212 available) to maximize performance.
* **Environment (`metatool_env`):** Provides states (tool context, task requirements), executes actions (selecting tools), and returns rewards (utility of chosen tools).
* **Goal:** Learn a policy that consistently selects the *best subset of tools* (5 tools at a time, 10–20 trials per episode) across domains.

---

## 2. RL Formulation

### 2.1 States / Observations (`s ∈ S`)

At timestep *t*, the environment returns an observation:

$$
s_t = \{ f_{task}, f_{tools}, f_{history} \}
$$

where:

* $f_{task}$: embedding of current task/domain (vector representation)
* $f_{tools}$: features of available tools (212 tools, categorical + description embeddings)
* $f_{history}$: previously chosen tools & outcomes

Representation:

* Use a **feature vector** of size `[task_dim + 212 × tool_dim + history_dim]`.
* Alternative: structured representation (transformer-based encoder).

---

### 2.2 Actions (`a ∈ A`)

* Agent must **select exactly k=5 tools** at each step from `N=212` tools.
* This defines a **combinatorial discrete action space**:

$$
|A| = \binom{212}{5}
$$

which is intractable if enumerated directly.

**Practical Relaxations:**

1. **Multi-discrete formulation:** Select 5 tools sequentially (with masking to avoid duplicates).
2. **Top-k continuous relaxation:** Predict a score vector over all 212 tools, then sample top-5.
3. **Hybrid (recommended):**

   * Policy outputs logits for all tools.
   * Use Gumbel-Top-k trick for differentiability.

---

### 2.3 Policies (`πθ`)

* **Stochastic policy:**

$$
πθ(a|s) = \prod_{i=1}^k \text{softmax}(fθ(s))[a_i]
$$

where $fθ$ is a neural network (e.g., Transformer/MLP).

* **Exploration:** Gumbel sampling, ε-greedy, or entropy regularization.

---

### 2.4 Trajectories

A trajectory τ is a sequence:

$$
τ = (s_0, a_0, r_0, \dots, s_T, a_T, r_T)
$$

where horizon $T \approx 10–20$.

---

### 2.5 Return Formulations

* **Episodic return:**

$$
R(τ) = \sum_{t=0}^{T} γ^t r_t
$$

* Reward design:

  * +1 if tool is among the “best 5” for the task.
  * Penalize irrelevant tools (e.g., -0.1).
  * Bonus for full correct set.

---

### 2.6 RL Optimization Problem

We solve:

$$
θ^* = \arg\max_θ \; \mathbb{E}_{τ \sim πθ} [R(τ)]
$$

---

### 2.7 Value Functions

* **State-value:** $Vπ(s) = \mathbb{E}[R_t | s_t = s, π]$
* **Action-value:** $Qπ(s,a) = \mathbb{E}[R_t | s_t = s, a_t = a, π]$
* **Advantage:** $Aπ(s,a) = Qπ(s,a) - Vπ(s)$

We may implement **actor-critic** (PPO/A2C) to stabilize learning.

---

## 3. Workflow & Code Structure

```
metatool-rl/
│
├── env/
│   ├── metatool_env.py    # Gym-style environment
│   ├── utils.py           # preprocessing, embeddings
│   ├── config_template.yaml # config placeholders
│   └── README.md          # environment doc
│
├── models/
│   ├── policy_network.py  # defines πθ
│   ├── value_network.py   # defines Vπ
│   ├── agent.py           # training loop (PPO, A2C, etc.)
│
├── experiments/
│   ├── train.py           # main training script
│   ├── test_env.py        # sanity checks
│   └── eval.py            # evaluation scripts
│
├── data/
│   ├── tools.json         # 212 tool metadata
│   └── tasks.json         # domain/task definitions
│
├── DESIGN.md              # (this file)
└── README.md              # project overview
```

---

## 4. Training Workflow

1. **Environment setup**

   * Define `MetaToolEnv` with `reset()`, `step()`, `reward()`.
   * Embed tasks + tools.

2. **Agent initialization**

   * Policy net outputs logits over 212 tools.
   * Apply Gumbel-Top-k for differentiable top-5 selection.

3. **Rollouts**

   * Collect trajectories for multiple episodes.

4. **Optimization**

   * Compute returns & advantages.
   * Update policy (PPO or REINFORCE baseline).

5. **Evaluation**

   * Measure % of correct tool selections.
   * Compare single-shot vs. multi-step selection.

6. **Logging & Visualization**

   * TensorBoard: rewards, tool distributions.

---

## 5. Mathematical Background (for Reviewer Ego 😏)

* The combinatorial action space is addressed via *structured stochastic policies*.
* Gumbel-Top-k sampling ensures tractability + differentiability.
* Optimization is framed as maximizing expected reward over **tool subsets**, making it a structured prediction problem.
* Relates to **combinatorial bandits** and **set-based RL**.