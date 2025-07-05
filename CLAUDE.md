# DomainBERT Project Overview

## 1. Summary

DomainBERT is a character-level BERT model designed specifically for domain name understanding. It is pretrained on a massive dataset of over 2.6 billion domain names from The Domains Project. The architecture is structurally aware, capable of distinguishing between the subdomain, domain, and top-level domain (TLD) components of a name. It uses a multi-task learning objective that combines masked language modeling (MLM) with TLD prediction to build a robust and nuanced representation of domain names. This foundational model can be fine-tuned for a wide variety of downstream tasks, including domain similarity, phishing detection, quality scoring, and price prediction.

## 2. Key Features

* **Character-Level Tokenization**: The model uses a character-level vocabulary with only valid domain characters (a-z, 0-9, hyphen, period), ensuring it focuses on legitimate domain patterns while using an [UNK] token for invalid characters.
* **Structural Awareness**: The tokenizer identifies the subdomain, main domain, and TLD, assigning different token type IDs to each component. This allows the model to learn the distinct roles of each part of a domain name.
* **Dedicated TLD Embeddings**: In addition to character embeddings, the model incorporates a separate, learnable embedding for each TLD, allowing it to capture the specific value and context associated with different suffixes.
* **Multi-Task Pretraining**: DomainBERT is pretrained using a dual-objective loss function:
    * **Masked Language Model (MLM)**: Predicts masked characters within the domain string.
    * **TLD Prediction**: Predicts the domain's TLD from the aggregated context.
* **Efficient Data Handling**: The pretraining pipeline uses an efficient streaming dataset implementation to process the 2.6B+ domain dataset without requiring it all to be loaded into memory, making large-scale training feasible.

## 3. Architecture

### Model Architecture

The core model, `DomainBertModel`, is a `PreTrainedModel` that uses a standard `BertEncoder` but is distinguished by its custom embedding layer, `DomainEmbeddings`. This embedding layer constructs the initial representation by combining four sources:
1.  **Character Embeddings**: For each character in the input sequence.
2.  **Position Embeddings**: To encode character order.
3.  **Token Type Embeddings**: To encode the structural part of the domain (e.g., subdomain, main domain, TLD).
4.  **TLD Embeddings**: A single vector representing the domain's TLD, which is broadcast across the entire sequence.

### Tokenizer

The `DomainBertTokenizerFast` is a custom tokenizer built for this task. Its most critical component is the `DomainPreTokenizer`, which first analyzes the domain string to identify its structure (subdomain, domain, suffix). It then assigns specific token type IDs (0 for domain, 1 for subdomain, 2 for TLD, 3 for separators) to each character before the main tokenization step. The tokenizer also manages a separate TLD vocabulary to map each suffix to a unique ID.

### Pretraining Objective

The pretraining is handled by the `DomainBertForMaskedLM` class, which includes two prediction heads:
1.  An **MLM head** that predicts masked characters across the entire sequence.
2.  A **TLD classification head** that takes the final pooled output and predicts the TLD.

The total loss is a weighted sum of the MLM loss and the TLD prediction loss, controlled by `mlm_weight` and `tld_weight` parameters in the config. The `DataCollatorForDomainMLM` is used to create batches for this multi-task objective, applying masking separately for characters and TLDs.

## 4. Pretraining

### Data

The model is designed to be pretrained on "The Domains Project" dataset, which contains over 2.6 billion domains. To manage this scale, the `MultiFileStreamingDataset` class streams data from compressed files, shuffles it using a large buffer, and tokenizes it on the fly. This approach supports multi-worker data loading and is essential for handling the massive dataset efficiently.

### Training Process

The pretraining process involves:
1.  Downloading The Domains Project dataset.
2.  Optionally creating smaller, stratified samples for testing or smaller-scale training runs.
3.  Executing the training script with configurations tailored for different hardware setups, from a quick test on a single GPU to a full-scale multi-GPU run on the complete dataset.

## 5. Usage

Once pretrained, the model can be loaded and used for inference or fine-tuning.

**Example Usage:**
```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load the pretrained model and tokenizer from a local directory
model = AutoModel.from_pretrained("./models/domain-bert-pretrained")
tokenizer = AutoTokenizer.from_pretrained("./models/domain-bert-pretrained")

# Encode a batch of domains
domains = ["example.com", "subdomain.example.org"]
encoded_input = tokenizer(domains, padding=True, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**encoded_input)

# Use the pooled output for downstream tasks
embeddings = outputs.pooler_output
```
The resulting embeddings can be used as features for various downstream tasks, or the entire `DomainBertForSequenceClassification` model can be fine-tuned for tasks like domain quality scoring or phishing detection.

## 6. Implementation Status

### 6.1 Completed Components

**Core Library (`src/domainbert/`)**
- ✅ **Configuration** (`config.py`): Complete `DomainBertConfig` class with all hyperparameters
- ✅ **Models** (`model.py`): 
  - `DomainEmbeddings`: Custom embedding layer combining character, position, token type, and TLD embeddings
  - `DomainBertModel`: Base transformer model
  - `DomainBertForMaskedLM`: Pretraining model with MLM and TLD prediction heads
  - `DomainBertForSequenceClassification`: Fine-tuning model for classification tasks
- ✅ **Tokenizer** (`tokenizer.py`):
  - `DomainPreTokenizer`: Analyzes domain structure and assigns token types
  - `DomainBertTokenizerFast`: Fast tokenizer with TLD vocabulary management
- ✅ **Data Utilities** (`data/`):
  - `DataCollatorForDomainMLM`: Handles masking for both characters and TLDs
  - `MultiFileStreamingDataset`: Efficient streaming from compressed files

**Data Processing Pipeline**
- ✅ **Dataset Download**: Script to fetch The Domains Project data (2.6B+ domains)
- ✅ **TLD Vocabulary Building**: Extracts and ranks TLDs by frequency (found 12,351 unique TLDs)
- ✅ **Sample Creation**: Stratified sampling for testing and development
- ✅ **Data Organization**: Country-specific subdirectories, Majestic Million integration

**Training Infrastructure**
- ✅ **Pretraining Script**: Full implementation with HuggingFace Trainer
- ✅ **Multi-GPU Support**: Distributed training with gradient accumulation
- ✅ **Monitoring**: Weights & Biases integration, TensorBoard logging
- ✅ **Test Runs**: Successfully completed test training runs

**Training Infrastructure Updates**
- ✅ **Hardware Auto-Detection**: Training script automatically detects CUDA/MPS/CPU and applies optimized settings
- ✅ **Unified Training Script**: Single script replaces platform-specific versions
- ✅ **Training Launcher**: Simple `scripts/train_launcher.py` with presets (test, small, 1hour, medium, full)
- ✅ **Apple Silicon Support**: Full MPS acceleration with optimized settings
- ✅ **Character Vocabulary**: Updated to only include valid domain characters (43 tokens total)

### 6.2 Recent Major Updates

- ✅ **Vocabulary Refinement**: Reduced from 128 ASCII to 43 valid domain characters
- ✅ **Training Script Consolidation**: Unified hardware-agnostic training pipeline
- ✅ **Performance Testing**: Verified ~6,358 samples/second on Apple M1
- ✅ **Documentation**: Created comprehensive training guides
- ✅ **File Organization**: Cleaned up project structure per user feedback

### 6.3 Ready for Production

- ✅ **Dataset Validated**: 1.65B domains across 1,676 compressed files
- ✅ **Training Time Estimates**: 
  - M1: ~6 days for full dataset (2 epochs)
  - H100: ~2 hours for full dataset (2 epochs)
- ✅ **Cost Estimates**: ~$7 on H100 for complete training

### 6.4 Known Issues

- ⚠️ **Multiworker Loading**: tldextract pickling issue (workaround: single worker)
- ⚠️ **Gradient Checkpointing**: Not yet implemented in model
- ⚠️ **Unit Tests**: Still needed for core components

## 7. Technical Details

### 7.1 TLD Vocabulary

The TLD vocabulary was extracted from the full dataset with the following statistics:
- **Total Unique TLDs**: 513 (in current vocabulary)
- **Most Common**: .com (1.2B domains), .net (150M), .org (148M)
- **Coverage**: Top TLDs cover vast majority of domains
- **Special Tokens**: [UNK_TLD] for unknown/new TLDs

### 7.2 Data Pipeline Optimizations

- **Streaming Architecture**: Never loads full dataset into memory
- **Multi-Process Loading**: Parallel decompression and tokenization
- **Efficient Shuffling**: Large buffer (100K domains) for randomization
- **Format Flexibility**: Handles .gz, .txt, and mixed formats

### 7.3 Training Configuration

Default configuration for base model:
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Intermediate Size**: 3072
- **Max Position Embeddings**: 64
- **Vocab Size**: 43 (valid domain chars + special tokens)
- **TLD Vocab Size**: 513
- **Training Speed (M1)**: ~6,358 samples/second
- **Recommended Epochs**: 2 (based on dataset size)

***

## Proposed Project Structure for DomainBERT

Here is a logical folder structure for the dedicated DomainBERT project. This design separates the core library code from operational scripts, data, and configurations, which is a standard best practice for maintainable machine learning projects.

```
domain-bert/
│
├── .gitignore             # Specifies intentionally untracked files to ignore
├── README.md              # High-level project description and quick start
├── CLAUDE.md              # Detailed project overview (the file above)
├── pyproject.toml         # Project metadata and build configuration
├── requirements.txt       # Project dependencies
│
├── configs/               # Configuration files for training and models
│   ├── model/
│   │   └── domain_bert_base_config.json
│   └── training/
│       └── pretrain_config.yaml
│
├── data/                  # For storing raw and processed datasets (should be in .gitignore)
│   ├── raw/
│   │   └── domains_project/
│   └── processed/
│       └── samples/
│
├── docs/                  # Project documentation
│   ├── architecture.md
│   └── pretraining_guide.md
│
├── models/                # Saved model checkpoints and final weights (should be in .gitignore)
│   └── domain-bert-pretrained/
│
├── scripts/               # Standalone scripts for data processing and model training
│   ├── data/
│   │   ├── download_dataset.py
│   │   └── preprocess_domains.py
│   └── training/
│       └── run_pretraining.py
│
├── src/
│   └── domainbert/        # The core, installable Python package
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── tokenizer.py
│       └── data/
│           ├── collator.py
│           └── streaming_dataset.py
│
└── tests/                 # Unit and integration tests
    ├── unit/
    │   ├── test_tokenizer.py
    │   └── test_model.py
    └── integration/
        └── test_pretraining_step.py
```

### Explanation of Key Directories

* **`src/domainbert/`**: This is the heart of the project, containing the Python source code for the model itself (`model.py`), the custom tokenizer (`tokenizer.py`), and the configuration class (`config.py`). Structuring it this way makes the project installable as a package (`pip install .`), allowing it to be easily used as a dependency in other projects (like Humbleworth).
* **`scripts/`**: This folder holds the scripts used to orchestrate the project's workflows, such as downloading data, preprocessing it, and launching the training process. This separates the operational logic from the core, reusable library code in `src/`.
* **`configs/`**: Centralizes all configurations. This makes it easy to manage different model sizes or training experiments without modifying the code.
* **`data/`** and **`models/`**: These directories are intended for large files that should not be tracked by Git. The `.gitignore` file should be configured to exclude their contents.
* **`tests/`**: Contains all tests, separated into `unit` (testing individual components in isolation) and `integration` (testing how components work together). This is crucial for ensuring code quality and reliability.
* **`docs/`**: All project documentation lives here, providing a clear reference for users and developers.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
IMPORTANT: Never create files in the root directory unless absolutely necessary. Follow the project structure and place files in appropriate subdirectories:
  - Scripts go in `scripts/` or subdirectories like `scripts/training/`, `scripts/data/`
  - Documentation goes in `docs/` or subdirectories like `docs/reports/`
  - Test results and reports go in `reports/`
  - Configuration files go in `configs/`
  - Keep the root directory clean with only essential files like README.md, CLAUDE.md, requirements.txt, pyproject.toml, and .gitignore

# Git Best Practices
  - Always commit changes using the user greg@siteorigin.com with the name Greg Priday.
  - Try to not mention Claude Code in your commit messages.