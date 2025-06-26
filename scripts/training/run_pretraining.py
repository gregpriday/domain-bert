#!/usr/bin/env python
"""
Main pretraining script for DomainBERT.

Uses Hugging Face Trainer for distributed training on domain data.
"""
import argparse
import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from domainbert.config import DomainBertConfig
from domainbert.model import DomainBertForMaskedLM
from domainbert.tokenizer import DomainBertTokenizerFast
from domainbert.data.collator import DataCollatorForDomainMLM
from domainbert.data.streaming_dataset import MultiFileStreamingDataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Sample file mapping
SAMPLE_FILES = {
    "tiny": ["domains_tiny.txt"],
    "small": ["domains_small.txt"],
    "medium": ["domains_medium.txt"],
    "large": ["domains_large.txt"],
    "full": None,  # Use all available files
}


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_config_path: Optional[str] = field(
        default="configs/model/domain_bert_base_config.json",
        metadata={"help": "Path to model configuration JSON file"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer (will look for vocab files here)"}
    )
    tld_vocab_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to TLD vocabulary file (if not in tokenizer_path)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for storing model artifacts"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input to our model."""
    
    data_dir: str = field(
        default="data/processed/domains",
        metadata={"help": "Directory containing processed domain files"}
    )
    sample: str = field(
        default="small",
        metadata={"help": "Dataset sample size: tiny, small, medium, large, or full"}
    )
    max_seq_length: int = field(
        default=64,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for MLM"}
    )
    tld_mask_probability: float = field(
        default=0.10,
        metadata={"help": "Probability of masking TLD for prediction"}
    )
    buffer_size: int = field(
        default=10000,
        metadata={"help": "Buffer size for tokenization batching"}
    )
    shuffle_buffer_size: int = field(
        default=100000,
        metadata={"help": "Shuffle buffer size for data randomization"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use for training (None = use all)"}
    )


@dataclass
class PretrainingArguments(TrainingArguments):
    """Training arguments with additional pretraining-specific options."""
    
    output_dir: str = field(
        default="models/domain-bert-pretrained",
        metadata={"help": "Output directory for model checkpoints"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"}
    )
    learning_rate: float = field(
        default=5e-4,
        metadata={"help": "Initial learning rate"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of total steps for warmup"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    save_steps: int = field(
        default=5000,
        metadata={"help": "Save checkpoint every N steps"}
    )
    eval_steps: int = field(
        default=5000,
        metadata={"help": "Run evaluation every N steps"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every N steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use mixed precision training"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of data loading workers"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"}
    )


def get_data_files(data_dir: Path, sample: str) -> List[str]:
    """Get list of data files based on sample size."""
    if sample == "full":
        # Use all .txt and .xz files in data directory and subdirectories
        files = []
        # Look for .txt and .xz files
        files.extend(list(data_dir.glob("**/*.txt")))
        files.extend(list(data_dir.glob("**/*.txt.xz")))
        # Exclude stats files
        files = [f for f in files if not f.name.endswith("_stats.txt")]
        return [str(f) for f in sorted(files)]
    else:
        # Use specific sample file
        sample_files = SAMPLE_FILES.get(sample, [])
        files = []
        for filename in sample_files:
            file_path = data_dir / filename
            if file_path.exists():
                files.append(str(file_path))
            else:
                logger.warning(f"Sample file not found: {file_path}")
        return files


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, PretrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from config file
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # Parse from command line
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    # Log basic information
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load config
    logger.info(f"Loading model config from {model_args.model_config_path}")
    with open(model_args.model_config_path, 'r') as f:
        config_dict = json.load(f)
    config = DomainBertConfig(**config_dict)
    
    # Load tokenizer
    tokenizer_path = model_args.tokenizer_path or Path(data_args.data_dir)
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    # Look for vocabulary files
    vocab_file = tokenizer_path / "vocab.txt" if isinstance(tokenizer_path, Path) else Path(tokenizer_path) / "vocab.txt"
    if not vocab_file.exists():
        vocab_file = None  # Will use default character vocabulary
    
    tld_vocab_file = model_args.tld_vocab_file
    if tld_vocab_file is None:
        tld_vocab_path = tokenizer_path / "tld_vocab.json" if isinstance(tokenizer_path, Path) else Path(tokenizer_path) / "tld_vocab.json"
        if tld_vocab_path.exists():
            tld_vocab_file = str(tld_vocab_path)
    
    tokenizer = DomainBertTokenizerFast(
        vocab_file=str(vocab_file) if vocab_file else None,
        tld_vocab_file=tld_vocab_file,
        max_len=data_args.max_seq_length,
    )
    
    # Initialize model
    logger.info("Initializing model from scratch")
    model = DomainBertForMaskedLM(config)
    
    # Log model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    # Get data files
    data_dir = Path(data_args.data_dir)
    train_files = get_data_files(data_dir, data_args.sample)
    
    if not train_files:
        raise ValueError(f"No training files found for sample '{data_args.sample}' in {data_dir}")
    
    logger.info(f"Found {len(train_files)} training files:")
    for f in train_files[:5]:
        logger.info(f"  - {f}")
    if len(train_files) > 5:
        logger.info(f"  ... and {len(train_files) - 5} more")
    
    # Create datasets
    train_dataset = MultiFileStreamingDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        buffer_size=data_args.buffer_size,
        shuffle_buffer_size=data_args.shuffle_buffer_size,
        seed=training_args.seed,
        max_samples=data_args.max_samples,
    )
    
    # For evaluation, use a smaller subset or separate validation files
    eval_files = train_files[:1]  # Use first file for eval (in practice, use separate validation set)
    eval_dataset = MultiFileStreamingDataset(
        file_paths=eval_files,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        buffer_size=data_args.buffer_size,
        shuffle_buffer_size=10000,  # Smaller shuffle buffer for eval
        seed=training_args.seed + 1,
    )
    
    # Create data collator
    data_collator = DataCollatorForDomainMLM(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        tld_mask_probability=data_args.tld_mask_probability,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save model
        trainer.save_model()
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate()
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()