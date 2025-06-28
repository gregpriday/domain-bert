"""
Efficient streaming dataset for large-scale domain pretraining
Handles 2.6B+ domains with minimal memory usage
"""

import os
import subprocess
from pathlib import Path
from typing import Iterator, List, Dict, Optional, Union
import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
from collections import deque
import random
import json
import csv


class MajesticMillionLoader:
    """Load domains from Majestic Million CSV"""
    
    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Majestic Million file not found: {csv_path}")
    
    def iter_domains(self) -> Iterator[str]:
        """Iterate through domains in Majestic Million"""
        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            f.seek(0)
            
            if any(keyword in first_line.lower() for keyword in ['domain', 'rank', 'url']):
                reader = csv.DictReader(f)
                domain_col = None
                for col in reader.fieldnames:
                    if 'domain' in col.lower() or 'url' in col.lower():
                        domain_col = col
                        break
                
                if domain_col:
                    for row in reader:
                        domain = row[domain_col].strip()
                        if '://' in domain:
                            domain = domain.split('://')[-1]
                        if '/' in domain:
                            domain = domain.split('/')[0]
                        if domain:
                            yield domain.lower()
            else:
                f.seek(0)
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        domain = row[0].strip()
                        if '://' in domain:
                            domain = domain.split('://')[-1]
                        if '/' in domain:
                            domain = domain.split('/')[0]
                        if domain:
                            yield domain.lower()


class MultiFileStreamingDataset(IterableDataset):
    """
    Stream domains from multiple compressed files efficiently
    Supports distributed training and multi-worker data loading
    """
    
    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        tokenizer=None,
        max_length: int = 64,
        buffer_size: int = 10000,
        shuffle_buffer_size: int = 100000,
        seed: int = 42,
        max_samples: Optional[int] = None,
        data_dir: Optional[Union[str, Path]] = None,
        files: Optional[List[str]] = None,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Args:
            file_paths: List of .xz or .txt files containing domains
            tokenizer: DomainBertTokenizerFast instance
            max_length: Maximum sequence length
            buffer_size: Size of tokenization buffer
            shuffle_buffer_size: Size of shuffle buffer for randomization
            seed: Random seed for shuffling
            max_samples: Maximum number of samples to yield (None = unlimited)
            data_dir: Directory containing domain files (alternative to file_paths)
            files: Specific file names to use from data_dir
            shuffle: Whether to shuffle the data
            num_workers: Number of workers (for compatibility)
        """
        # Handle data_dir interface
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            if files:
                # Use specific files
                self.file_paths = [str(self.data_dir / f) for f in files]
                self.files = [self.data_dir / f for f in files]
            else:
                # Find all domain files
                self.files = sorted(list(self.data_dir.glob("*.txt")) + 
                                  list(self.data_dir.glob("*.txt.gz")) + 
                                  list(self.data_dir.glob("*.xz")) +
                                  list(self.data_dir.glob("*.csv")))
                self.file_paths = [str(f) for f in self.files]
        elif file_paths:
            self.file_paths = sorted(file_paths)
            self.files = [Path(f) for f in self.file_paths]
            self.data_dir = None
        else:
            raise ValueError("Either file_paths or data_dir must be provided")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size if shuffle else 1
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.num_workers = num_workers
        
        # Estimate total domains based on The Domains Project stats
        # Total domains: 1,766,025,618 (from STATS.md)
        # If we have all files, use the exact count, otherwise estimate
        if len(self.file_paths) > 100:  # Likely the full dataset
            self.estimated_size = 1_766_025_618
        else:
            self.estimated_size = len(self.file_paths) * 50_000_000  # ~50M per file
    
    def __len__(self):
        """Return estimated dataset size"""
        return self.estimated_size
    
    def _open_file(self, file_path: str):
        """Open compressed or text file for streaming"""
        import gzip
        
        if file_path.endswith('.xz'):
            # Stream from compressed file using subprocess
            proc = subprocess.Popen(
                ['xzcat', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1024*1024  # 1MB buffer
            )
            return proc.stdout
        elif file_path.endswith('.gz'):
            # Handle gzipped files
            return gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
        elif file_path.endswith('.csv'):
            # Handle CSV files (like Majestic Million)
            loader = MajesticMillionLoader(file_path)
            # Return an iterator that behaves like a file
            class IteratorFile:
                def __init__(self, iterator):
                    self.iterator = iterator
                def __iter__(self):
                    return self
                def __next__(self):
                    return next(self.iterator) + '\n'
                def close(self):
                    pass
            return IteratorFile(loader.iter_domains())
        else:
            # Regular text file
            return open(file_path, 'r', encoding='utf-8', errors='ignore')
    
    def _get_file_shards(self) -> List[str]:
        """Determine which files this worker should process"""
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single process data loading
            return self.file_paths
        else:
            # Multi-process data loading
            # Distribute files across workers
            per_worker = len(self.file_paths) // worker_info.num_workers
            worker_id = worker_info.id
            
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker
            
            if worker_id == worker_info.num_workers - 1:
                # Last worker gets remaining files
                end_idx = len(self.file_paths)
            
            return self.file_paths[start_idx:end_idx]
    
    def _iter_domains(self) -> Iterator[str]:
        """Iterate through domains from assigned files"""
        file_shards = self._get_file_shards()
        
        for file_path in file_shards:
            try:
                with self._open_file(file_path) as f:
                    for line in f:
                        domain = line.strip().lower()
                        if domain and self._is_valid_domain(domain):
                            yield domain
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Basic domain validation"""
        if len(domain) < 3 or len(domain) > 253:
            return False
        if '.' not in domain:
            return False
        return True
    
    def __iter__(self):
        """Stream and tokenize domains with optional shuffling"""
        # Set random seed based on worker
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        np.random.seed(self.seed + worker_id)
        random.seed(self.seed + worker_id)
        
        samples_yielded = 0
        
        if not self.shuffle:
            # No shuffling - process domains in order
            tokenization_buffer = []
            for domain in self._iter_domains():
                tokenization_buffer.append(domain)
                
                # Tokenize when buffer is full
                if len(tokenization_buffer) >= self.buffer_size:
                    for item in self._tokenize_and_yield(tokenization_buffer):
                        yield item
                        samples_yielded += 1
                        if self.max_samples and samples_yielded >= self.max_samples:
                            return
                    tokenization_buffer = []
            
            # Process remaining
            if tokenization_buffer:
                for item in self._tokenize_and_yield(tokenization_buffer):
                    yield item
                    samples_yielded += 1
                    if self.max_samples and samples_yielded >= self.max_samples:
                        return
        else:
            # With shuffling
            shuffle_buffer = deque(maxlen=self.shuffle_buffer_size)
            tokenization_buffer = []
            
            for domain in self._iter_domains():
                # Add to shuffle buffer
                shuffle_buffer.append(domain)
                
                # Once shuffle buffer is full, start yielding
                if len(shuffle_buffer) >= self.shuffle_buffer_size:
                    # Randomly sample from shuffle buffer
                    idx = random.randint(0, len(shuffle_buffer) - 1)
                    selected_domain = shuffle_buffer[idx]
                    shuffle_buffer[idx] = shuffle_buffer[-1]  # Move last to selected position
                    
                    tokenization_buffer.append(selected_domain)
                    
                    # Tokenize when buffer is full
                    if len(tokenization_buffer) >= self.buffer_size:
                        for item in self._tokenize_and_yield(tokenization_buffer):
                            yield item
                            samples_yielded += 1
                            if self.max_samples and samples_yielded >= self.max_samples:
                                return
                        tokenization_buffer = []
            
            # Process remaining items in shuffle buffer
            remaining = list(shuffle_buffer) + tokenization_buffer
            random.shuffle(remaining)
            
            # Process in batches
            for i in range(0, len(remaining), self.buffer_size):
                if self.max_samples and samples_yielded >= self.max_samples:
                    return
                batch = remaining[i:i + self.buffer_size]
                for item in self._tokenize_and_yield(batch):
                    yield item
                    samples_yielded += 1
                    if self.max_samples and samples_yielded >= self.max_samples:
                        return
    
    def _tokenize_and_yield(self, domains: List[str]):
        """Tokenize batch and yield individual examples"""
        try:
            encoded = self.tokenizer(
                domains,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            for i in range(len(domains)):
                yield {
                    'input_ids': encoded['input_ids'][i],
                    'attention_mask': encoded['attention_mask'][i],
                    'token_type_ids': encoded['token_type_ids'][i],
                    'tld_ids': encoded['tld_ids'][i]
                }
        except Exception as e:
            print(f"Tokenization error: {e}")
            return


class DomainDatasetConfig:
    """Configuration for domain dataset streaming"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load streaming configuration"""
        default_config = {
            'batch_size': 128,
            'max_length': 64,
            'buffer_size': 10000,
            'shuffle_buffer_size': 100000,
            'num_workers': 4,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'pin_memory': True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def get_dataloader_kwargs(self) -> Dict:
        """Get DataLoader keyword arguments"""
        return {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'prefetch_factor': self.config['prefetch_factor'],
            'persistent_workers': self.config['persistent_workers'],
            'pin_memory': self.config['pin_memory'] and torch.cuda.is_available()
        }


def create_streaming_dataloader(
    file_paths: List[str],
    tokenizer,
    config: Optional[DomainDatasetConfig] = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create an efficient DataLoader for streaming domain data
    
    Args:
        file_paths: List of domain files (.xz or .txt)
        tokenizer: DomainBertTokenizerFast instance
        config: Dataset configuration
        **kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader instance configured for efficient streaming
    """
    if config is None:
        config = DomainDatasetConfig()
    
    # Create dataset
    dataset = MultiFileStreamingDataset(
        file_paths=file_paths,
        tokenizer=tokenizer,
        max_length=config.config.get('max_length', 64),
        buffer_size=config.config.get('buffer_size', 10000),
        shuffle_buffer_size=config.config.get('shuffle_buffer_size', 100000),
        **kwargs
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **config.get_dataloader_kwargs()
    )
    
    return dataloader


class DomainSampler:
    """
    Intelligent domain sampler for curriculum learning
    Samples domains based on quality criteria
    """
    
    def __init__(self, quality_scores: Optional[Dict[str, float]] = None):
        self.quality_scores = quality_scores or {}
    
    def compute_domain_quality(self, domain: str) -> float:
        """
        Compute quality score for a domain
        Higher scores = better quality for training
        """
        score = 1.0
        
        # Length penalty
        base = domain.split('.')[0]
        if len(base) < 3 or len(base) > 15:
            score *= 0.8
        
        # Numeric penalty
        if any(c.isdigit() for c in base):
            score *= 0.9
        
        # Hyphen penalty
        if '-' in base:
            score *= 0.95
        
        # Repetition penalty
        for c in set(base):
            if base.count(c) > len(base) // 2:
                score *= 0.7
                break
        
        # TLD bonus
        tld = domain.split('.')[-1]
        tld_weights = {
            'com': 1.2,
            'org': 1.1,
            'net': 1.1,
            'edu': 1.15,
            'gov': 1.15
        }
        score *= tld_weights.get(tld, 1.0)
        
        return score
    
    def should_include(self, domain: str, threshold: float = 0.7) -> bool:
        """Decide whether to include domain in training"""
        if domain in self.quality_scores:
            return self.quality_scores[domain] >= threshold
        
        score = self.compute_domain_quality(domain)
        return score >= threshold


# Example usage
if __name__ == "__main__":
    # Example: Create streaming dataloader
    from pathlib import Path
    
    # Find domain files
    data_dir = Path("data/raw/domains_project/data")
    domain_files = list(data_dir.glob("domains-part-*.xz"))[:5]  # First 5 files
    
    if domain_files:
        print(f"Found {len(domain_files)} domain files")
        
        # Mock tokenizer for testing
        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    'input_ids': [[1] * 64 for _ in texts],
                    'attention_mask': [[1] * 64 for _ in texts],
                    'token_type_ids': [[0] * 64 for _ in texts],
                    'tld_ids': [1 for _ in texts]
                }
        
        # Create dataloader
        dataloader = create_streaming_dataloader(
            file_paths=[str(f) for f in domain_files],
            tokenizer=MockTokenizer()
        )
        
        # Test iteration
        print("Testing dataloader...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: {batch['input_ids'].shape}")
            if i >= 5:
                break
    else:
        print("No domain files found. Run download_domains_project.py first")