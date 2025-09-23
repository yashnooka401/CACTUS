# models/random_agent.py
import random
import time
import numpy as np

class RandomAgent:
    def __init__(self, env, logdir: str = ".", **kwargs):
        self.env = env
        self.logdir = logdir
        self.rng = random.Random(kwargs.get("seed", 0))

    def train(self, num_episodes: int, writer):
        metrics = []
        for ep in range(num_episodes):
            ts = self.env.reset()
            done = False
            ep_return = 0.0
            steps = 0
            while not done:
                # action is index in env.tool_vocab (global vocab)
                action = self.rng.randint(0, len(self.env.tool_vocab) - 1)
                ts = self.env.step(action)
                reward = ts.reward
                ep_return += reward
                steps += 1
                done = ts.last()
            # compute precision/recall/f1 from env.selected_tools & env.correct_tools
            tp = len(set(self.env.selected_tools) & set(self.env.correct_tools))
            precision = tp / max(1, len(self.env.selected_tools))
            recall = tp / max(1, len(self.env.correct_tools))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            metrics.append({"episode": ep, "return": ep_return, "precision": precision, "recall": recall, "f1": f1, "steps": steps})
            writer.add_scalar("random/return", ep_return, ep)
            writer.add_scalar("random/f1", f1, ep)
        return metrics

    def save(self, path):
        # nothing to save for random agent
        return
