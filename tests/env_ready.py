"""
test_env_ready.py

Sanity check for MetaToolEnv using real datasets (all_clean.csv / process.csv).
This script ensures the environment can load data, reset, and step properly.

Author: CACTUS Authors
Date: 2025-08-28
"""

import pandas as pd
import os
from env import metatool_env, config


def build_vocab(df):
    """Extract global tool vocabulary from Tool + New Tools columns."""
    vocab = set()

    # Collect from "Tool"
    if "Tool" in df.columns:
        for val in df["Tool"].dropna():
            for t in str(val).replace(";", ",").split(","):
                t = t.strip()
                if t:
                    vocab.add(t)

    # Collect from "New Tools" if exists
    if "New Tools" in df.columns:
        for val in df["New Tools"].dropna():
            for t in str(val).replace(";", ",").split(","):
                t = t.strip()
                if t:
                    vocab.add(t)

    return sorted(vocab)


def run_env_test(csv_path, mode="multi-step", max_steps=1000, seed=42):
    """Run a short rollout to verify the environment works."""
    print(f"\n=== Testing environment with {os.path.basename(csv_path)} ===")
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', on_bad_lines='skip')

    # Rename Query column if needed (some datasets might lowercase it)
    if "Query" not in df.columns:
        raise ValueError("CSV must contain a 'Query' column")

    tool_vocab = build_vocab(df)
    print(f"Loaded dataset with {len(df)} rows, vocab size={len(tool_vocab)}")

    env = metatool_env.MetaToolEnv(df, tool_vocab=tool_vocab, mode=mode, max_steps=max_steps, seed=seed)

    ts = env.reset()
    print("Initial Observation:", ts.observation)
    print("Ground truth tools:", env.correct_tools)

    for step in range(max_steps):
        action = env.rng.randint(0, len(env.tool_vocab) - 1)
        ts = env.step(action)
        print(f"Step {step+1}: Action={env.tool_vocab[action]} Reward={ts.reward} Done={ts.last()}")
        if ts.last():
            break


if __name__ == "__main__":
    # Test with both multi-step and single-shot modes
    run_env_test(config.PROCESSED_DATASET, mode="multi-step", max_steps=200)
    run_env_test(config.PROCESSED_DATASET, mode="single-shot", max_steps=5)