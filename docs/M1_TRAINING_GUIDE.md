# DomainBERT Training on Apple Silicon (M1/M2/M3)

This guide provides optimized settings and instructions for training DomainBERT on Apple Silicon Macs using Metal Performance Shaders (MPS).

## Prerequisites

1. **macOS 12.3+** (required for MPS support)
2. **PyTorch 2.0+** with MPS support:
   ```bash
   pip install torch torchvision torchaudio
   ```
3. **Python 3.8+**

## Automatic Hardware Detection

The training script automatically detects Apple Silicon and applies MPS-optimized settings. No manual configuration needed!

## Quick Start

### 1. Test MPS Availability
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 2. Quick Test Run (10K samples)
```bash
python train.py --preset test
```
This runs a quick test with 10,000 samples to verify everything works (~5-10 minutes).

### 3. Full Training
```bash
python train.py --preset full
```
This runs the full training pipeline. Expect 2-7 days depending on your M1 variant.

## M1 Optimization Settings

The training script automatically applies these optimizations when MPS is detected:

### Memory Management
- **Batch Size**: 64 (optimized for M1 memory)
- **Gradient Accumulation**: 8 steps (effective batch size: 512)
- **Gradient Checkpointing**: Enabled to save memory
- **Single Worker**: Avoids multiprocessing overhead

### Auto-Applied Settings
```python
# Automatically applied when MPS is detected:
per_device_train_batch_size = 64
gradient_accumulation_steps = 8
gradient_checkpointing = True
dataloader_num_workers = 0  # Single process
fp16 = False  # Not well supported on MPS yet
```

You can override these with command-line arguments if needed.

### Environment Variables
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback for unsupported ops
export TOKENIZERS_PARALLELISM=false   # Avoid warnings
export OMP_NUM_THREADS=8              # Use performance cores
```

## Monitoring Performance

### Built-in Monitoring
The training script logs performance metrics automatically:
- Samples per second
- Memory usage
- Loss values
- GPU utilization (when available)

### Activity Monitor
- Open Activity Monitor > Window > GPU History
- Watch for GPU usage during training

## Expected Performance

### M1 Variants
- **M1**: ~50-70 samples/sec, 5-7 days for full dataset
- **M1 Pro**: ~80-120 samples/sec, 3-5 days for full dataset  
- **M1 Max**: ~150-200 samples/sec, 2-3 days for full dataset
- **M1 Ultra**: ~300-400 samples/sec, 1-2 days for full dataset

### Memory Usage
- Base model: ~4-6GB unified memory
- During training: ~8-12GB unified memory
- Peak usage: ~16GB unified memory

## Troubleshooting

### MPS Not Available
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Out of Memory
1. Reduce batch size:
   ```bash
   --per_device_train_batch_size 32
   ```
2. Increase gradient accumulation:
   ```bash
   --gradient_accumulation_steps 16
   ```
3. Close other applications

### Slow Performance
1. Check thermal throttling:
   ```bash
   sudo pmset -g thermlog
   ```
2. Use a laptop stand/cooling pad
3. Reduce batch size if throttling occurs

### MPS Operations Not Supported
Some operations may fall back to CPU. This is normal and handled by:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Advanced Configuration

### Custom Training Script
```python
import torch

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# M1-optimized settings
config = {
    "batch_size": 64,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-4,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True
}
```

### Memory-Efficient Data Loading
The streaming dataset automatically handles the 1.65B domains without loading everything into memory:
- Streams from compressed .xz files
- Processes data on-the-fly
- Uses efficient buffering

### Resume Training
```bash
python train.py --preset full \
    --resume models/domain-bert-full/checkpoint-50000
```

## Tips for Best Performance

1. **Close unnecessary apps** to free unified memory
2. **Use external SSD** for data if internal storage is limited
3. **Train overnight** when thermal conditions are better
4. **Monitor temperatures** and adjust batch size if throttling
5. **Custom settings** if auto-detection needs adjustment:
   ```bash
   python train.py --preset full --batch_size 32 --gradient_accumulation_steps 16
   ```

## Comparison with GPU Training

| Metric | M1 Max | RTX 3090 | A100 |
|--------|---------|----------|------|
| Samples/sec | 150-200 | 400-500 | 800-1000 |
| Memory | 64GB unified | 24GB VRAM | 40GB VRAM |
| Power | 40W | 350W | 400W |
| Cost | Built-in | $1,500 | $10,000 |

While M1 is slower than dedicated GPUs, it offers excellent performance per watt and allows training large models that wouldn't fit on consumer GPUs due to unified memory architecture.

## Next Steps

After training completes:
1. The model is saved in `models/domain-bert-{preset}/`
2. Fine-tune for specific tasks using the same auto-detection
3. Export for production use

Happy training on Apple Silicon! 🍎