"""
config.py

Configuration file for the MetaTool RL project.
Centralizes dataset paths, environment parameters, and training hyperparameters.
"""

import os

# === PATHS ===
# TODO: later replace with Remote URL or relative paths
RAW_DATASET = '/Users/pranith/Code/CACTUS/data/Raw/all_clean_data.csv' # Directory for raw data files
PROCESSED_DATASET = '/Users/pranith/Code/CACTUS/data/Processed/editeddataset.csv'  # Directory for processed data files


LOGS_DIR = "../logs"            # Directory for logs and checkpoints
CHECKPOINT_DIR = os.path.join(LOGS_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === ENVIRONMENT ===
MAX_STEPS = 1        # For single-shot predictions (change for multi-step)
OBSERVATION_TYPE = "text"  # Options: "text", "embedding"

# === RL TRAINING ===
RL_ALGO = "PPO"       # Options: "PPO", "DQN", "A2C"
LEARNING_RATE = 3e-4
GAMMA = 0.99
BATCH_SIZE = 32
NUM_EPISODES = 1000

# === REWARD SETTINGS ===
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -0.5

# === SEED ===
SEED = 42

# Create logs directory if missing
os.makedirs(LOGS_DIR, exist_ok=True)
print("Configuration loaded successfully.\n")
# print("Configurations:")
# for key, value in globals().items():
#     if key.isupper():
#         print(f"  {key}: {value}")
