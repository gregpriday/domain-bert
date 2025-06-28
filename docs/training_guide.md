# DomainBERT Training Guide

This guide covers training DomainBERT on various hardware configurations. The training script automatically detects your hardware and applies optimized settings.

## Hardware Auto-Detection

The training script automatically detects and optimizes for:
- **NVIDIA GPUs** (CUDA)
- **Apple Silicon** (M1/M2/M3 with MPS)
- **CPU** (fallback)

## Quick Start

### Using Presets

The easiest way to start training is using presets:

```bash
# Test run (10K samples, ~10 minutes)
python train.py --preset test

# Small run (1M samples, ~1 hour)
python train.py --preset small

# Medium run (100M samples, ~1 day)
python train.py --preset medium

# Full training (all 1.65B domains, 2-7 days)
python train.py --preset full
```

### Custom Training

For more control over training parameters:

```bash
# Basic custom training
python scripts/training/run_pretraining.py \
    --max_samples 1000000 \
    --num_train_epochs 2 \
    --learning_rate 1e-4

# Override auto-detected settings
python scripts/training/run_pretraining.py \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --fp16
```

## Hardware-Specific Optimizations

### NVIDIA GPUs (CUDA)

Auto-applied settings:
- `per_device_train_batch_size`: 128
- `gradient_accumulation_steps`: 2
- `fp16`: True (mixed precision)
- `dataloader_num_workers`: 4

Multi-GPU training:
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 scripts/training/run_pretraining.py \
    --output_dir models/domain-bert-large

# Distributed training across nodes
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/training/run_pretraining.py
```

### Apple Silicon (MPS)

Auto-applied settings:
- `per_device_train_batch_size`: 64
- `gradient_accumulation_steps`: 8
- `gradient_checkpointing`: True
- `fp16`: False (not well supported)
- `dataloader_num_workers`: 0

Expected performance:
- M1: ~50-70 samples/sec
- M1 Pro: ~80-120 samples/sec
- M1 Max: ~150-200 samples/sec
- M1 Ultra: ~300-400 samples/sec

### CPU Training

Auto-applied settings:
- `per_device_train_batch_size`: 16
- `gradient_accumulation_steps`: 16
- `fp16`: False
- `dataloader_num_workers`: 2

Note: CPU training is significantly slower and only recommended for testing.

## Data Configuration

### Dataset Samples

The training script supports different dataset sizes:

- `tiny`: Test dataset (included)
- `small`: 1M domain sample
- `medium`: 100M domain sample
- `large`: 1B domain sample
- `full`: Complete 1.65B domain dataset

### Custom Data Directory

```bash
python train.py --preset full \
    --data_dir /path/to/your/domains
```

## Advanced Options

### Resume Training

```bash
# Resume from checkpoint
python train.py --preset full \
    --resume models/domain-bert-full/checkpoint-50000

# Or with the training script directly
python scripts/training/run_pretraining.py \
    --resume_from_checkpoint models/domain-bert-full/checkpoint-50000
```

### Memory Optimization

If you run out of memory:

1. Reduce batch size:
   ```bash
   python train.py --batch_size 32
   ```

2. Increase gradient accumulation:
   ```bash
   python scripts/training/run_pretraining.py \
       --per_device_train_batch_size 16 \
       --gradient_accumulation_steps 32
   ```

3. Enable gradient checkpointing:
   ```bash
   python scripts/training/run_pretraining.py \
       --gradient_checkpointing
   ```

### Mixed Precision Training

For NVIDIA GPUs with Tensor Cores:
```bash
python train.py --preset full --fp16
```

For AMD GPUs or newer NVIDIA GPUs:
```bash
python scripts/training/run_pretraining.py \
    --bf16 \
    --tf32
```

### Logging and Monitoring

#### Weights & Biases
```bash
python train.py --preset full \
    --report_to wandb \
    --run_name "domainbert-full-run"
```

#### TensorBoard
```bash
python train.py --preset full \
    --report_to tensorboard \
    --logging_dir ./logs
```

### Custom Model Configuration

```bash
# Use a different model size
python scripts/training/run_pretraining.py \
    --model_config_path configs/model/domain_bert_large_config.json

# Or modify settings directly
python scripts/training/run_pretraining.py \
    --hidden_size 512 \
    --num_hidden_layers 24 \
    --num_attention_heads 16
```

## Performance Tips

1. **Data Loading**: The streaming dataset automatically handles the 1.65B domains efficiently
2. **Checkpointing**: Saves every 5000 steps by default (configurable with `--save_steps`)
3. **Evaluation**: Runs every 5000 steps (configurable with `--eval_steps`)
4. **Gradient Clipping**: Applied automatically with `max_grad_norm=1.0`
5. **Learning Rate Schedule**: Linear warmup (10% of steps) + linear decay

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size or enable gradient checkpointing
- Use `torch.cuda.empty_cache()` between runs
- Monitor with `nvidia-smi`

### MPS Issues on Apple Silicon
- Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
- Update to latest PyTorch version
- Reduce batch size if seeing memory pressure

### Slow Data Loading
- Check disk I/O speed
- Ensure data files are on SSD
- For distributed training, use shared filesystem

### Training Instability
- Reduce learning rate
- Increase warmup steps
- Enable gradient clipping

## Expected Training Times

| Hardware | Dataset | Time Estimate |
|----------|---------|---------------|
| 4x A100 | Full (1.65B) | 2-3 days |
| 4x 3090 | Full (1.65B) | 4-5 days |
| M1 Max | Full (1.65B) | 5-7 days |
| 1x 3090 | Medium (100M) | 20-24 hours |
| M1 Pro | Small (1M) | 1-2 hours |

## After Training

Once training completes:

1. **Model saved to**: `models/domain-bert-{preset}/`
2. **Tokenizer saved**: Same directory as model
3. **Training metrics**: `trainer_state.json`
4. **Checkpoints**: Keep last 3 by default

To use the trained model:
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("models/domain-bert-full")
tokenizer = AutoTokenizer.from_pretrained("models/domain-bert-full")
```

## Next Steps

- Fine-tune for specific tasks (phishing detection, similarity, etc.)
- Evaluate on downstream benchmarks
- Export to ONNX for production deployment
- Share on HuggingFace Hub