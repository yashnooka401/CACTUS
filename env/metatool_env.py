"""
metatool_env.py

MetaTool Environment: A DeepMind-style environment for training RL agents
on tool selection tasks. Supports single-shot and multi-step interaction,
with an action space spanning the full tool vocabulary.

Author: CACTUS Authors
Date: 2025-08-28
"""

import random
import dm_env
import pandas as pd
from typing import List, Dict, Optional
from dm_env import specs


class MetaToolEnv(dm_env.Environment):
    """
    MetaTool tool-selection environment.

    Agents must select a subset of tools (typically K=5) from a global
    vocabulary (~212 tools). Rewards are given for selecting correct tools.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tool_vocab: List[str],
        mode: str = "multi-step",
        max_steps: int = 20,
        seed: Optional[int] = None
    ):
        """
        Initialize the MetaTool environment.

        Args:
            dataframe (pd.DataFrame): Must contain ["Query", "Tool"] and optionally ["New Tools"].
            tool_vocab (list[str]): Global vocabulary of all tools.
            mode (str): "single-shot" or "multi-step".
            max_steps (int): Maximum number of steps per episode (for multi-step).
            seed (int, optional): Random seed for reproducibility.
        """
        assert "Query" in dataframe.columns, "DataFrame must contain 'Query' column"
        assert "Tool" in dataframe.columns, "DataFrame must contain 'Tool' column"
        assert mode in ["single-shot", "multi-step"], "Invalid mode"

        self.df = dataframe.reset_index(drop=True)
        self.tool_vocab = list(tool_vocab)
        self.mode = mode
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        # Add a "DONE" token to the action space
        if "DONE" not in self.tool_vocab:
            self.tool_vocab = self.tool_vocab + ["DONE"]

        self.current_index = None
        self.correct_tools = None
        self.selected_tools = []
        self.prompt = None
        self.done = False
        self.step_count = 0

    @staticmethod
    def _normalize_tools(val) -> List[str]:
        """Convert CSV string/list into a clean list of tool names."""
        if pd.isna(val):
            return []
        if isinstance(val, str):
            return [t.strip() for t in val.replace(";", ",").split(",") if t.strip()]
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
        return [str(val).strip()]

    def reset(self) -> dm_env.TimeStep:
        """Reset environment to start a new episode."""
        self.done = False
        self.selected_tools = []
        self.step_count = 0

        self.current_index = self.rng.randint(0, len(self.df) - 1)
        row = self.df.iloc[self.current_index]

        self.prompt = row["Query"]

        # Prefer "New Tools" if available, else fallback to "Tool"
        if "New Tools" in row and not pd.isna(row["New Tools"]):
            self.correct_tools = self._normalize_tools(row["New Tools"])
        else:
            self.correct_tools = self._normalize_tools(row["Tool"])

        return dm_env.restart(self._get_observation())

    def step(self, action: int) -> dm_env.TimeStep:
        """Take one step in the environment."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        self.step_count += 1
        chosen_tool = self.tool_vocab[action]
        reward = 0.0

        # Reward logic
        if chosen_tool == "DONE":
            self.done = True
        elif chosen_tool in self.correct_tools and chosen_tool not in self.selected_tools:
            reward = 1.0  # reward for correct unique tool
            self.selected_tools.append(chosen_tool)
        else:
            reward = 0.0  # can extend to penalize wrong tools

        # End conditions
        if self.mode == "single-shot":
            self.done = True
        else:
            if set(self.selected_tools) == set(self.correct_tools):
                self.done = True
            if self.step_count >= self.max_steps:
                self.done = True

        if self.done:
            # Extra terminal reward: F1 score
            precision = len(set(self.selected_tools) & set(self.correct_tools)) / max(
                1, len(self.selected_tools)
            )
            recall = len(set(self.selected_tools) & set(self.correct_tools)) / len(
                self.correct_tools
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            final_reward = reward + f1
            return dm_env.termination(final_reward, self._get_observation())
        else:
            return dm_env.transition(reward, self._get_observation())

    def _get_observation(self) -> Dict[str, object]:
        """Construct observation dictionary."""
        return {
            "prompt": self.prompt,
            "available_tools": self.tool_vocab,
            "selected_tools": self.selected_tools,
        }

    def observation_spec(self):
        """Define observation spec."""
        return {
            "prompt": specs.Array(shape=(), dtype=object, name="prompt"),
            "available_tools": specs.Array(
                shape=(len(self.tool_vocab),), dtype=object, name="available_tools"
            ),
            "selected_tools": specs.Array(
                shape=(None,), dtype=object, name="selected_tools"
            ),
        }

    def action_spec(self):
        """Define action spec (discrete tool choices)."""
        return specs.DiscreteArray(
            num_values=len(self.tool_vocab),
            dtype=int,
            name="action"
        )


if __name__ == "__main__":
    # Example standalone test
    df = pd.DataFrame({
        "Query": ["Find academic papers on reinforcement learning."],
        "Tool": ["SEOTool"],
        "New Tools": ["Google Scholar, Arxiv"]
    })
    vocab = ["Google Scholar", "Arxiv", "PubMed", "Mendeley", "ResearchGate"]
    env = MetaToolEnv(df, tool_vocab=vocab, mode="multi-step", max_steps=10, seed=42)
    ts = env.reset()
    print("Observation:", ts.observation)
    for _ in range(5):
        action = env.rng.randint(0, len(env.tool_vocab) - 1)
        ts = env.step(action)
        print(f"Action: {env.tool_vocab[action]} Reward: {ts.reward} Done: {ts.last()}")
        if ts.last():
            break