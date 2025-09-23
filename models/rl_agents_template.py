# models/rl_agents_template.py
"""_summary_
# TODO: initialize your RL model here (PPO/DQN). You can use Stable-Baselines3:
    # from stable_baselines3 import PPO
    # self.model = PPO(policy='MlpPolicy', env=gym_env, verbose=1, ...)
    # Or implement your own policy/value nets.

Returns:
    _type_: _description_
"""
class RLAgent:
    def __init__(self, env, logdir: str = ".", use_llm: bool = False, algo: str = "PPO", llm_backend: str = "gpt", llm_noise: bool = False, **kwargs):
        self.env = env
        self.logdir = logdir
        self.use_llm = use_llm
        self.algo = algo
        self.llm_backend = llm_backend
        self.llm_noise = llm_noise
        # TODO: initialize your RL model here (PPO/DQN). You can use Stable-Baselines3:
        # from stable_baselines3 import PPO
        # self.model = PPO(policy='MlpPolicy', env=gym_env, verbose=1, ...)
        # Or implement your own policy/value nets.
    def train(self, num_episodes: int, writer):
        """
        Train loop must:
        - collect episodes
        - update model
        - write metrics to `writer.add_scalar(...)`
        - return list of per-episode metric dicts
        """
        metrics = []
        for ep in range(num_episodes):
            # Example pseudo-loop: replace with real rollout & update
            ts = self.env.reset()
            done = False
            ep_return = 0.0
            steps = 0
            while not done:
                # Build observation -> convert to features for your model
                # If using LLM shortlist, ask llm_wrapper for candidate scores and mask
                action = self.policy_action(ts)  # you must implement
                ts = self.env.step(action)
                ep_return += ts.reward
                steps += 1
                done = ts.last()
            # compute F1 etc.
            tp = len(set(self.env.selected_tools) & set(self.env.correct_tools))
            precision = tp / max(1, len(self.env.selected_tools))
            recall = tp / max(1, len(self.env.correct_tools))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            metrics.append({"episode": ep, "return": ep_return, "precision": precision, "recall": recall, "f1": f1, "steps": steps})
            writer.add_scalar(f"{self.algo}/return", ep_return, ep)
            writer.add_scalar(f"{self.algo}/f1", f1, ep)
        return metrics

    def policy_action(self, timestep):
        # placeholder
        import random
        return random.randint(0, len(self.env.tool_vocab) - 1)

    def save(self, path):
        # save model weights/checkpoint
        pass
