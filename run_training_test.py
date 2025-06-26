#!/usr/bin/env python3
"""Quick training test script with fixes"""
import os
import subprocess

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

# Run training with dataloader_num_workers=0 to avoid multiprocessing issues
cmd = [
    "python", "scripts/training/run_pretraining.py",
    "--model_config_path", "configs/model/domain_bert_base_config.json",
    "--tld_vocab_file", "data/processed/domains/tld_vocab.json",
    "--data_dir", "data/raw/domains_project/data",
    "--sample", "full",
    "--max_samples", "1000",
    "--output_dir", "models/domain-bert-test",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "32",
    "--dataloader_num_workers", "0",  # Disable multiprocessing
    "--logging_steps", "10",
    "--do_train",
    "--overwrite_output_dir"
]

print("Running training with fixes...")
print(" ".join(cmd))
subprocess.run(cmd)