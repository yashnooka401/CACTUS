# models/scripted_agent.py
class ScriptedAgent:
    """
    Simple rule-based agent:
    - If a tool name substring appears in prompt, select its index (first match)
    - Else pick a random tool.
    """
    def __init__(self, env, logdir: str = ".", **kwargs):
        import random
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
                prompt = ts.observation["prompt"].lower()
                chosen_index = None
                for idx, tool in enumerate(self.env.tool_vocab):
                    if tool.lower() in prompt:
                        chosen_index = idx
                        break
                if chosen_index is None:
                    chosen_index = self.rng.randint(0, len(self.env.tool_vocab) - 1)
                ts = self.env.step(chosen_index)
                ep_return += ts.reward
                steps += 1
                done = ts.last()
            tp = len(set(self.env.selected_tools) & set(self.env.correct_tools))
            precision = tp / max(1, len(self.env.selected_tools))
            recall = tp / max(1, len(self.env.correct_tools))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            metrics.append({"episode": ep, "return": ep_return, "precision": precision, "recall": recall, "f1": f1, "steps": steps})
            writer.add_scalar("scripted/return", ep_return, ep)
            writer.add_scalar("scripted/f1", f1, ep)
        return metrics

    def save(self, path):
        return
