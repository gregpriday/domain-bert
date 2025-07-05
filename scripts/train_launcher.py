#!/usr/bin/env python
"""
Simple training launcher for DomainBERT.
Provides convenient presets for common training scenarios.

Usage:
    # Quick test (10K samples)
    python train.py --preset test
    
    # Small training run (1M samples)
    python train.py --preset small
    
    # Full training
    python train.py --preset full
    
    # Custom settings
    python train.py --batch_size 32 --learning_rate 1e-4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Training presets
PRESETS = {
    "test": {
        "description": "Quick test run (10K samples, ~10 minutes)",
        "args": [
            "--sample", "tiny",
            "--max_samples", "1000",
            "--num_train_epochs", "1",
            "--save_steps", "500",
            "--eval_steps", "500",
            "--logging_steps", "50",
        ]
    },
    "small": {
        "description": "Small training run (1M samples, ~1 hour)",
        "args": [
            "--max_samples", "1000000",
            "--num_train_epochs", "1",
            "--save_steps", "5000",
            "--eval_steps", "5000",
            "--logging_steps", "100",
        ]
    },
    "1hour": {
        "description": "1-hour training run (~23M samples on M1)",
        "args": [
            "--sample", "full",
            "--max_steps", "45000",
            "--save_steps", "5000",
            "--eval_steps", "5000",
            "--logging_steps", "1000",
        ]
    },
    "medium": {
        "description": "Medium training run (100M samples, ~1 day)",
        "args": [
            "--max_samples", "100000000",
            "--num_train_epochs", "2",
            "--save_steps", "10000",
            "--eval_steps", "10000",
            "--logging_steps", "500",
        ]
    },
    "full": {
        "description": "Full training run (all data, 3 epochs)",
        "args": [
            "--sample", "full",
            "--num_train_epochs", "3",
            "--save_steps", "50000",
            "--eval_steps", "50000",
            "--logging_steps", "1000",
        ]
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="DomainBERT training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --preset test                    # Quick test
  python train.py --preset 1hour                   # 1-hour training run  
  python train.py --preset full --fp16             # Full training with mixed precision
  python train.py --batch_size 32 --epochs 5      # Custom settings
        """
    )
    
    # Preset selection
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a predefined training configuration"
    )
    
    # Common training arguments
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--learning_rate", "--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    
    # Parse known args
    args, unknown_args = parser.parse_known_args()
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/training/run_pretraining.py",
        "--do_train",
        "--do_eval",
        "--overwrite_output_dir",
    ]
    
    # Add preset args
    if args.preset:
        print(f"Using preset: {args.preset} - {PRESETS[args.preset]['description']}")
        cmd.extend(PRESETS[args.preset]['args'])
    
    # Add custom args
    if args.batch_size:
        cmd.extend(["--per_device_train_batch_size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.epochs:
        cmd.extend(["--num_train_epochs", str(args.epochs)])
    if args.warmup_ratio:
        cmd.extend(["--warmup_ratio", str(args.warmup_ratio)])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    else:
        # Default output dir based on preset
        output_name = f"domain-bert-{args.preset}" if args.preset else "domain-bert-custom"
        cmd.extend(["--output_dir", f"models/{output_name}"])
    if args.fp16:
        cmd.extend(["--fp16"])
    if args.resume:
        cmd.extend(["--resume_from_checkpoint", args.resume])
    
    # Add any unknown args
    cmd.extend(unknown_args)
    
    # Default paths
    cmd.extend([
        "--model_config_path", "configs/model/domain_bert_base_config.json",
        "--tld_vocab_file", "src/domainbert/data/tld_vocab.json",
        "--data_dir", "data/raw/domains_project/data",
    ])
    
    # Show command
    print("\nRunning command:")
    print(" ".join(cmd))
    print()
    
    # Set environment variables if needed
    env = os.environ.copy()
    if args.wandb_project:
        env['WANDB_PROJECT'] = args.wandb_project
    
    # Run training
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()