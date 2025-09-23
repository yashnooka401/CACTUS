## Logging layout & naming convention

When you run run_experiments.py, it creates:
```bash
logs/
 └── 20250923_142501/           # timestamp (YYYYMMDD_HHMMSS)
     ├── scripted/
     │   ├── tensorboard/       # TensorBoard directory
     │   ├── episode_metrics.csv
     │   └── checkpoint.pkl
     ├── random/
     │   └── ...
     └── rl_llm/
         └── ...
```


## TensorBoard command to view latest logs:

```bash
tensorboard --logdir logs --bind_all
# or for a single experiment:
tensorboard --logdir logs/20250923_142501 --bind_all
```