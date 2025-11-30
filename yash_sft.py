from trl import SFTTrainer
from datasets import load_dataset
import pandas as pd
import os
from datasets import Dataset

# Path to your CSV dataset (use raw string on Windows)
csv_path = r"data\Processed\editeddataset.csv"

# Load CSV with pandas
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Normalize columns to 'text' (SFTTrainer expects this field by default)
# Adjust column names based on your actual CSV structure
if "prompt" in df.columns and "response" in df.columns:
    df['text'] = df['prompt'] + " " + df['response']
elif "input" in df.columns and "target" in df.columns:
    df['text'] = df['input'] + " " + df['target']
else:
    # If you have a single text column, rename it
    df['text'] = df.iloc[:, 0].astype(str)  # Use first column as fallback

# Drop rows with missing text
df = df.dropna(subset=['text'])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text']].reset_index(drop=True))

# Create output directory
output_dir = "outputs/sft-qwen"
os.makedirs(output_dir, exist_ok=True)

# Instantiate trainer. Add or adjust arguments supported by SFTTrainer as needed.
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
)

if __name__ == "__main__":
    trainer.train()