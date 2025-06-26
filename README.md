# DomainBERT

A character-level BERT model designed specifically for domain name understanding. DomainBERT is pretrained on a massive dataset of over 2.6 billion domain names from The Domains Project.

## Features

- **Character-Level Tokenization**: Handles any ASCII domain name, including those with unusual characters or patterns
- **Structural Awareness**: Distinguishes between subdomain, domain, and TLD components with different token type IDs
- **Dedicated TLD Embeddings**: Incorporates learnable embeddings for each TLD to capture suffix-specific context
- **Multi-Task Pretraining**: Combines masked language modeling (MLM) with TLD prediction
- **Efficient Data Handling**: Streaming dataset implementation for large-scale training without loading all data into memory

## Project Status

✅ **Implemented**
- Core model architecture (DomainBertModel, DomainBertForMaskedLM)
- Custom tokenizer with structural awareness
- Streaming dataset for 2.6B+ domains
- Data preprocessing pipeline
- Multi-GPU training infrastructure
- TLD vocabulary extraction from dataset

⚠️ **In Progress**
- Full-scale pretraining on complete dataset
- Performance benchmarking

❌ **TODO**
- Unit and integration tests
- Comprehensive documentation
- Fine-tuning examples for downstream tasks

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- ~150GB disk space for full dataset

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/domain-bert.git
cd domain-bert

# Install the package in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

### 1. Download The Domains Project Dataset

```bash
# Download the full dataset (~127GB compressed)
python scripts/data/download_domains_project.py

# Or download a specific subset (e.g., US domains only)
python scripts/data/download_domains_project.py --subset us
```

### 2. Build TLD Vocabulary

```bash
# Extract TLD statistics and build vocabulary
python scripts/data/build_tld_vocabulary.py
```

### 3. Create Training Samples (Optional)

For testing or smaller experiments:

```bash
# Create a stratified sample of 10M domains
python scripts/data/create_domain_samples.py --sample-size 10000000
```

### 4. Run Pretraining

```bash
# Test run with small sample
python scripts/training/run_pretraining.py \
    --config configs/model/domain_bert_base_config.json \
    --max_steps 1000 \
    --per_device_train_batch_size 32

# Full pretraining (multi-GPU recommended)
torchrun --nproc_per_node=4 scripts/training/run_pretraining.py \
    --config configs/model/domain_bert_base_config.json \
    --output_dir models/domain-bert-pretrained \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 4
```

## Usage

### Loading Pretrained Model

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load the pretrained model and tokenizer
model = AutoModel.from_pretrained("./models/domain-bert-pretrained")
tokenizer = AutoTokenizer.from_pretrained("./models/domain-bert-pretrained")

# Encode domains
domains = ["example.com", "subdomain.example.org", "phishing-site.tk"]
encoded_input = tokenizer(domains, padding=True, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**encoded_input)

# Use the pooled output for downstream tasks
embeddings = outputs.pooler_output  # Shape: (batch_size, hidden_size)
```

### Fine-tuning for Classification

```python
from domainbert import DomainBertForSequenceClassification, DomainBertConfig

# Load model for binary classification (e.g., phishing detection)
config = DomainBertConfig.from_pretrained("./models/domain-bert-pretrained")
config.num_labels = 2

model = DomainBertForSequenceClassification.from_pretrained(
    "./models/domain-bert-pretrained",
    config=config
)

# Fine-tune using standard HuggingFace Trainer
# ... (see examples in docs/)
```

## Model Architecture

DomainBERT uses a BERT-like transformer architecture with domain-specific modifications:

- **Vocabulary**: 128 ASCII characters + special tokens
- **Embeddings**: Character + Position + Token Type + TLD embeddings
- **Hidden Size**: 256 (base model)
- **Layers**: 12 transformer blocks
- **Attention Heads**: 8
- **Max Sequence Length**: 128 characters

See [docs/architecture.md](docs/architecture.md) for detailed information.

## Data

The model is trained on [The Domains Project](https://thedomainsproject.org/) dataset:
- **Size**: 2.6+ billion registered domains
- **Coverage**: All TLDs and country codes
- **Format**: Plain text files, one domain per line
- **Updates**: Daily snapshots available

## Training Details

- **Pretraining Tasks**: 
  - Masked Language Modeling (85% weight)
  - TLD Prediction (15% weight)
- **Masking**: 15% of characters masked
- **Batch Size**: 512 (with gradient accumulation)
- **Learning Rate**: 5e-4 with warmup
- **Training Time**: ~72 hours on 4x A100 GPUs

## Project Structure

```
domain-bert/
├── src/domainbert/        # Core library package
│   ├── config.py          # Model configuration
│   ├── model.py           # Model implementations
│   ├── tokenizer.py       # Custom tokenizer
│   └── data/              # Data utilities
├── scripts/               # Training and data scripts
│   ├── data/              # Data processing scripts
│   └── training/          # Training scripts
├── configs/               # Configuration files
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Citation

If you use DomainBERT in your research, please cite:

```bibtex
@software{domainbert2024,
  title = {DomainBERT: A Character-Level BERT Model for Domain Name Understanding},
  year = {2024},
  url = {https://github.com/yourusername/domain-bert}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Domains Project for providing the comprehensive domain dataset
- HuggingFace Transformers library for the excellent framework
- The BERT authors for the foundational architecture