#!/usr/bin/env python3
"""
Quick test training script for DomainBERT
"""
import subprocess
import sys
from pathlib import Path

# Training arguments
args = [
    "python", "scripts/training/run_pretraining.py",
    "--model_config_path", "configs/model/domain_bert_base_config.json",
    "--tld_vocab_file", "data/processed/domains/tld_vocab.json",
    "--data_dir", "data/processed/domains",
    "--sample", "tiny",
    "--output_dir", "models/domain-bert-test",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "32",
    "--learning_rate", "5e-4",
    "--warmup_ratio", "0.1",
    "--logging_steps", "10",
    "--save_steps", "500",
    "--eval_steps", "500",
    "--do_train",
    "--do_eval",
    "--overwrite_output_dir",
    "--dataloader_num_workers", "2"
]

print("Starting DomainBERT test training...")
print(f"Command: {' '.join(args)}")

# Run the training
result = subprocess.run(args)

if result.returncode == 0:
    print("\nTraining complete!")
else:
    print(f"\nTraining failed with return code: {result.returncode}")
    sys.exit(result.returncode)