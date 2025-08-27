# Evaluation Metrics & Plots — MetaTool RL

## 1 — Core per-episode metrics (compute per episode τ)

Compute these first for every episode, then aggregate across episodes / seeds.

1. **Episode Return** `G(τ)`
   \- Sum of discounted or undiscounted rewards collected this episode.
   \- Use undiscounted for short horizons (T ≤ 20): `G = Σ_t r_t`.

2. **Success Count / Accuracy**
   \- `success = #correct_tools_selected` (integer 0..K).
   \- **Set Accuracy** (final): `set_acc = success / K` (if K=5).

3. **Precision\@k / Recall\@k / F1\@k**
   \- Treat chosen set vs ground-truth set as predicted/true positives.
   \- `precision@K = TP / (TP + FP)` ; `recall@K = TP / (TP + FN)`.

4. **Steps-to-First-Complete (SFC)**
   \- Number of steps taken until all required tools are found. If never completed, set to `max_steps` or `∞` (use censoring in plots).

5. **Avg Steps Used**
   \- Average number of steps per episode (efficiency metric).

6. **Redundancy / Repeats**
   \- Count of repeated selections (should be low with masking).

7. **Tool Diversity**
   \- Over N episodes: entropy or distinct tools selected fraction.
   \- `diversity = (#unique tools chosen across episodes) / 212`.

8. **Policy Entropy (per-step)**
   \- Average entropy of the policy's action distribution (measures exploration).

9. **LLM Contribution (for hybrid)**
   \- Fraction of agent actions that came from LLM shortlist or had LLM-high-score.
   \- Correlation between LLM suggestion rank and reward.

10. **Robustness / Noise Sensitivity**
    \- Performance under noisy LLM outputs (randomized LLM), or with partial tool masking.

---

## 2 — Aggregate statistics (over episodes / seeds)

* Mean ± standard error (SE) for each metric (report SE or 95% CI).
* Median and inter-quartile range for skewed metrics (e.g., steps-to-complete).
* Report results per domain (if tasks grouped by domain) and overall.

**Repetition policy:** at least 5 seeds; ideally 10 independent runs per experiment.

**Statistical tests:**

* Compare two conditions with paired t-test (if normal) or Wilcoxon signed-rank (non-parametric).
* For multiple comparisons use ANOVA + post-hoc (Tukey) or Bonferroni-corrected pairwise tests.

---

## 3 — Recommended Plots

### A. Learning & Sample Efficiency

1. **Learning Curve**: `Mean Return vs Training Steps`

   * X-axis: environment steps or episodes (log scale optional).
   * Y-axis: average episodic return (mean ± 95% CI shaded).
   * One curve per condition (Scripted, RL-only, RL+LLM, Ablated).

2. **Sample Efficiency Plot**: `Success Rate vs #Episodes`

   * Plot how quickly each method reaches thresholds (e.g., 20%, 50%, 80% set accuracy).
   * Use vertical lines for episodes-to-threshold.

### B. Final Performance & Ablations

3. **Bar Chart (Final Metrics)**: `Set Accuracy / Precision@K / Recall@K`

   * Bars = mean over seeds; error bars = 95% CI.
   * Separate panels for seen vs unseen tasks.

4. **Ablation Bar Chart**: show drop in performance when removing LLM or adding noise.

### C. Per-Tool & Behavior Visualizations

5. **Tool Selection Heatmap**: `Tools (rows) × Conditions (cols)` showing selection frequency.

   * Cluster rows by tool domain; visualize normalized frequencies.

6. **Confusion / Co-occurrence Matrix**: how often tool A and B are selected together.

7. **Histogram / CDF of Steps-to-Complete**: show distribution across episodes and conditions.

8. **Reward Distribution**: violin/boxplots per method showing return spread.

### D. Diagnostic / Analysis Plots

9. **Policy Entropy over Time**: how exploration decays during training.
10. **LLM Rank vs Reward Scatter**: plot LLM suggestion rank (1,2,...) vs whether that choice yielded reward.
11. **t-SNE / UMAP of Tool Embeddings**: color points by selected/true tools to show if correct tools cluster.

### E. Survival-style Plot

12. **Kaplan–Meier-like Curve**: `P(not completed) vs Steps` — shows how quickly episodes finish (good for censored SFC).

---

## 4 — Evaluation Protocol (step-by-step)

1. **Data splits**

   * Train / validation / test splits by prompt. Ensure *domain* generalization: hold out whole domains for a generalization test set.

2. **Hyperparam sweeps**

   * Run grid search for learning rate, entropy coeff, PPO clip, advantage λ. Keep same budget across methods for fairness.

3. **Warm-start experiments**

   * (Optional) Behavior cloning pretraining for policy; test RL fine-tune vs training from scratch.

4. **Logging**

   * Save raw trajectories (prompt, chosen tools, rewards, LLM output) for qualitative analysis.
   * Log per-step metrics to TensorBoard (return, entropy, value loss).

5. **Evaluation runs**

   * Run 1000 evaluation episodes per final checkpoint if compute allows (or 200) for stable aggregate metrics. Use deterministic policy (argmax) for final performance and stochastic (sample) for ablation of exploration.

6. **Significance**

   * Report p-values for core comparisons (e.g., RL-only vs RL+LLM).
   * Report effect sizes (Cohen’s d).

---

## 5 — What to show in the paper / supplement

**Main paper (concise):**

* Learning curve (mean ± CI) for return (key claim: faster convergence with LLM).
* Bar chart comparing final set accuracy across conditions.
* Table summarizing sample complexity (#episodes to reach 50% / 80% accuracy).
* One heatmap of tool selection frequencies for main methods.

**Supplementary:**

* Precision\@K / Recall\@K / F1\@K per domain.
* Statistical test details & per-seed results.
* Additional ablations (noise levels, LLM models).
* Example trajectories (prompts + chosen tools + LLM rationale) — qualitative.

---

## 6 — Metric Implementation Snippets (pseudo)

**Precision\@K / Recall\@K (per episode)**

```
pred_set = set(chosen_tools)
true_set = set(ground_truth_tools)
TP = len(pred_set & true_set)
precision = TP / len(pred_set)      # handle zero division
recall = TP / len(true_set)
```

**Sample Complexity**

* For each run, record `episodes_to_reach(p)` = first episode where moving-average success ≥ p. Aggregate mean & SE across seeds.

---

## 7 — Practical thresholds & defaults (suggested)

* Seeds per experiment: **5–10** (≥5 required).
* Eval episodes per checkpoint: **200** (or 1000 if possible).
* Confidence intervals: **95%**.
* Significance: report **p < 0.05** (two-sided) and effect sizes.

---

## 8 — Logging & Reproducibility checklist

* Save random seeds, env/config, and exact code commit hash.
* Save model checkpoints and final policy weights.
* Provide scripts to reproduce main figures (one-click `run_experiment.sh` examples).
* Release dataset preprocessing code and synthetic-tool mapping.

---