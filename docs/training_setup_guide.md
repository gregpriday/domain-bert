# DomainBERT Training Setup Guide

This guide provides step-by-step instructions for setting up DomainBERT training on a new server after cloning the repository.

## Prerequisites

- **System**: Linux (Ubuntu/Debian) or macOS
- **Python**: 3.8 or higher
- **Disk Space**: ~200GB for the full dataset (compressed)
- **RAM**: 16GB minimum, 32GB+ recommended
- **GPU**: Optional but highly recommended (NVIDIA CUDA-capable GPU)

## Step 1: Install System Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y git git-lfs xz-utils build-essential python3-dev
```

### macOS
```bash
brew install git git-lfs xz
```

### All Systems
After installing git-lfs, initialize it:
```bash
git lfs install
```

## Step 2: Set Up Python Environment

Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows
```

## Step 3: Install Python Dependencies

Install all required Python packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU support, ensure you have the correct PyTorch version:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (M1/M2/M3)
pip install torch  # MPS support is included by default
```

## Step 4: Download The Domains Project Dataset

Download the 2.6+ billion domain dataset:
```bash
python scripts/data/download_domains_project.py \
    --data-dir data/raw/domains_project
```

**Note**: This download requires ~200GB of disk space and may take several hours depending on your internet connection.

### Alternative: Use Existing Dataset
If the dataset is already available on your server or mounted storage:
```bash
# Create symlink to existing dataset
ln -s /path/to/existing/domains_project data/raw/domains_project

# Verify the download
python scripts/data/download_domains_project.py \
    --data-dir data/raw/domains_project \
    --skip-download
```

## Step 5: Build TLD Vocabulary

Extract and build the TLD (Top-Level Domain) vocabulary from the dataset:
```bash
python scripts/data/build_tld_vocabulary.py \
    --data-dir data/raw/domains_project/data \
    --output-dir data/processed/domains \
    --min-count 10 \
    --processes 8  # Adjust based on CPU cores
```

This step:
- Analyzes all domains to find unique TLDs
- Creates `tld_vocab.json` for the tokenizer
- Saves statistics in `tld_stats.json`
- Takes ~30-60 minutes on a modern server

## Step 6: Create Training Samples (Optional)

For testing or smaller-scale training, create stratified samples:
```bash
# Create all sample sizes (tiny, small, medium, large)
python scripts/data/create_domain_samples.py \
    --data-dir data/raw/domains_project/data \
    --output-dir data/processed/domains \
    --samples all

# Or create specific samples
python scripts/data/create_domain_samples.py \
    --data-dir data/raw/domains_project/data \
    --output-dir data/processed/domains \
    --samples small medium
```

Sample sizes:
- `tiny`: 10,000 domains (~10 minutes training)
- `small`: 1 million domains (~1 hour training)
- `medium`: 10 million domains (~10 hours training)
- `large`: 100 million domains (~4 days training)

## Step 7: Verify Setup

Run the training readiness test:
```bash
python scripts/tests/test_training_readiness.py
```

This will verify:
- Dataset is properly downloaded
- TLD vocabulary is built
- Model can be initialized
- Training can start successfully

## Step 8: Start Training

### Quick Test Run
Verify everything works with a 10-minute test:
```bash
python scripts/train_launcher.py --preset test
```

### Full Training
Launch full training on the complete dataset:
```bash
python scripts/train_launcher.py --preset full
```

### Custom Training
For more control over parameters:
```bash
python scripts/training/run_pretraining.py \
    --max_samples 100000000 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 128 \
    --learning_rate 1e-4 \
    --output_dir models/domain-bert-custom
```

## Monitoring Training

### Weights & Biases (Recommended)
```bash
# Set up W&B account (first time only)
wandb login

# Run training with W&B logging
python scripts/train_launcher.py --preset full \
    --report_to wandb \
    --run_name "domainbert-full-$(date +%Y%m%d)"
```

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir models/

# Run training with TensorBoard logging
python scripts/train_launcher.py --preset full \
    --report_to tensorboard
```

## Multi-GPU Training

For servers with multiple GPUs:
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) \
    scripts/training/run_pretraining.py \
    --output_dir models/domain-bert-multi-gpu

# Specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/training/run_pretraining.py
```

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python scripts/train_launcher.py --preset full --batch_size 32

# Enable gradient checkpointing
python scripts/training/run_pretraining.py --gradient_checkpointing
```

### Slow Data Loading
- Ensure data is on SSD, not HDD
- Reduce number of data loader workers if I/O bound
- Check disk usage with `iotop` or `iostat`

### Dataset Download Issues
- Check available disk space: `df -h`
- Verify git-lfs is installed: `git lfs version`
- Resume interrupted download by re-running the download script

## Next Steps

After training completes:
1. Model will be saved to `models/domain-bert-{preset}/`
2. Test the model with inference examples
3. Fine-tune for specific downstream tasks
4. Export to ONNX for production deployment

## Resource Requirements Summary

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Disk Space | ~200GB | For compressed dataset |
| RAM | 16GB minimum | 32GB+ recommended for full dataset |
| GPU | Optional | Highly recommended for reasonable training times |
| Time (Full Dataset) | 2-7 days | Depends on hardware |
| Time (Setup) | 2-4 hours | Including dataset download |

## Support

For issues or questions:
- Check existing documentation in `docs/`
- Review training logs in `models/*/logs/`
- Examine the FAQ in `docs/troubleshooting.md`