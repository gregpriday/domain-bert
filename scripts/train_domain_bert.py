#!/usr/bin/env python3
"""
Training script for DomainBERT with fixed vocabulary handling
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback
)
from transformers.integrations import WandbCallback
import evaluate

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from domainbert.config import DomainBertConfig
from domainbert.model import DomainBertForMaskedLM
from domainbert.tokenizer import DomainBertTokenizerFast
from domainbert.data.collator import DataCollatorForDomainMLM
from domainbert.data.streaming_dataset import MultiFileStreamingDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """Compute character-level accuracy and perplexity metrics"""
    try:
        # The eval_pred is an EvalPrediction object
        predictions = eval_pred.predictions  # This should be logits
        labels = eval_pred.label_ids
        
        # Handle the case where predictions might be a tuple
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Ensure predictions is a numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
            
        # Handle labels - they might be a list of arrays or a single array
        if isinstance(labels, list):
            # If it's a list, try to stack
            try:
                labels = np.stack(labels)
            except:
                # If stacking fails, convert each element
                labels = np.array([np.array(l) for l in labels])
        elif not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # Ensure we have 2D arrays
        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if len(labels.shape) == 3 and labels.shape[0] == 1:
            labels = labels[0]
            
        # Basic checks
        if predictions.shape[0] != labels.shape[0]:
            logger.error(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}")
            return {"accuracy": 0.0, "perplexity": float('inf')}
        
        # Mask out non-predicted positions
        mask = labels != -100
        
        # Check if we have any valid predictions
        if not np.any(mask):
            return {"accuracy": 0.0, "perplexity": float('inf')}
        
        # Get predictions
        preds = predictions.argmax(axis=-1)
        
        # Separate character and TLD predictions
        # Character tokens are 0-43, TLD tokens are 44+
        char_mask = mask & (labels < 44)
        tld_mask = mask & (labels >= 44)
        
        # Overall accuracy
        accuracy = (preds[mask] == labels[mask]).astype(float).mean()
        
        # Character accuracy
        if np.any(char_mask):
            char_accuracy = (preds[char_mask] == labels[char_mask]).astype(float).mean()
        else:
            char_accuracy = 0.0
        
        # TLD accuracy
        if np.any(tld_mask):
            tld_accuracy = (preds[tld_mask] == labels[tld_mask]).astype(float).mean()
        else:
            tld_accuracy = 0.0
        
        # Calculate perplexity from loss
        # Note: This is approximate since we're using logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        logits_flat = torch.from_numpy(predictions).float().view(-1, predictions.shape[-1])
        labels_flat = torch.from_numpy(labels).long().view(-1)
        
        losses = loss_fct(logits_flat, labels_flat)
        valid_losses = losses[labels_flat != -100]
        
        if valid_losses.numel() > 0:
            avg_loss = valid_losses.mean().item()
            perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        else:
            perplexity = float('inf')
        
        # Count prediction distribution for debugging
        pred_counts = {}
        valid_preds = preds[mask]
        for pred in valid_preds:
            pred = int(pred)
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        # Get top 5 most predicted tokens
        top_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_pred_ratio = top_preds[0][1] / len(valid_preds) if top_preds else 0.0
        
        return {
            "accuracy": float(accuracy),
            "char_accuracy": float(char_accuracy),
            "tld_accuracy": float(tld_accuracy),
            "perplexity": float(perplexity),
            "top_pred_ratio": float(top_pred_ratio),
            "masked_tokens": int(np.sum(mask))
        }
    except Exception as e:
        logger.error(f"Error in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            "accuracy": 0.0,
            "char_accuracy": 0.0,
            "tld_accuracy": 0.0,
            "perplexity": float('inf'),
            "top_pred_ratio": 0.0,
            "masked_tokens": 0
        }


class CharAccuracyCallback(WandbCallback):
    """Custom callback to log character-level metrics"""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Log additional analysis
            logger.info(f"Step {state.global_step}: Eval metrics - {metrics}")
            
            # Check for degenerate predictions
            if metrics.get("eval_top_pred_ratio", 0) > 0.5:
                logger.warning(f"Warning: Model predicting same token {metrics['eval_top_pred_ratio']:.1%} of the time!")
        
        return super().on_evaluate(args, state, control, metrics, **kwargs)


def load_evaluation_dataset(tokenizer, max_length=64):
    """Load the fixed evaluation dataset"""
    eval_path = "/home/ubuntu/domain-bert/data/eval_dataset_10k.json"
    
    with open(eval_path, 'r') as f:
        data = json.load(f)
    
    domains = data['domains']
    
    # Tokenize all domains
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    # Create dataset
    from datasets import Dataset
    eval_dataset = Dataset.from_dict({'text': domains})
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=['text']
    )
    
    return eval_dataset


def main():
    parser = argparse.ArgumentParser(description="Train DomainBERT")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="domain-bert-base")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=False, 
                       help="Data directory (optional - will use pre-shuffled file if available)")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=256)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--wandb_project", type=str, default="domainbert")
    parser.add_argument("--wandb_run_name", type=str)
    
    # System arguments
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize tokenizer
    if args.checkpoint:
        logger.info(f"Loading tokenizer from checkpoint: {args.checkpoint}")
        tokenizer = DomainBertTokenizerFast.from_pretrained(args.checkpoint)
    else:
        # Load tokenizer from the initialized location
        tokenizer = DomainBertTokenizerFast.from_pretrained(
            "/home/ubuntu/domain-bert/models/tokenizer"
        )
    
    # Initialize model
    if args.checkpoint:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = DomainBertForMaskedLM.from_pretrained(args.checkpoint)
    else:
        logger.info("Initializing new model from scratch")
        config = DomainBertConfig()
        model = DomainBertForMaskedLM(config)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Initialize datasets
    # Check if pre-shuffled file exists
    shuffled_file = Path("/home/ubuntu/domain-bert/data/processed/all_domains_shuffled.txt")
    sample_file = Path("/home/ubuntu/domain-bert/data/processed/domains_sample_10m.txt")
    
    if shuffled_file.exists() and not args.data_dir:
        # Use pre-shuffled file for much faster training
        logger.info(f"Using pre-shuffled data file: {shuffled_file}")
        from domainbert.data.simple_dataset import SimpleStreamingDataset
        train_dataset = SimpleStreamingDataset(
            file_path=str(shuffled_file),
            tokenizer=tokenizer,
            max_length=args.max_length,
            buffer_size=100000
        )
    elif sample_file.exists() and not args.data_dir:
        # Use sample file if full file doesn't exist
        logger.info(f"Using sample data file: {sample_file}")
        from domainbert.data.simple_dataset import SimpleStreamingDataset
        train_dataset = SimpleStreamingDataset(
            file_path=str(sample_file),
            tokenizer=tokenizer,
            max_length=args.max_length,
            buffer_size=100000
        )
    else:
        # Fall back to streaming from compressed files
        if not args.data_dir:
            args.data_dir = "/home/ubuntu/domain-bert/data/raw/domains_project/data"
        logger.info(f"Loading training data from {args.data_dir}")
        train_dataset = MultiFileStreamingDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
            num_workers=args.num_workers,
            buffer_size=100000
        )
    
    logger.info("Loading evaluation dataset")
    eval_dataset = load_evaluation_dataset(tokenizer, args.max_length)
    
    # Initialize data collator
    data_collator = DataCollatorForDomainMLM(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        mlm=True
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_char_accuracy",
        greater_is_better=True,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=["wandb", "tensorboard"],
        run_name=args.wandb_run_name or f"domainbert-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_safetensors=False,  # Avoid shared tensor issues
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            CharAccuracyCallback(),
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )
    
    # Log initial info
    logger.info(f"Model parameters: {model.num_parameters():,}")
    logger.info(f"Training samples: ~{len(train_dataset):,}")
    logger.info(f"Evaluation samples: {len(eval_dataset):,}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()}")
    
    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    try:
        metrics = trainer.evaluate()
        
        # Save metrics
        with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training complete! Final metrics: {metrics}")
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        logger.info("Training complete! (evaluation failed)")


if __name__ == "__main__":
    # Set environment variables
    os.environ["WANDB_PROJECT"] = "domainbert"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()