"""
Simple streaming dataset for reading from a single shuffled file
"""

import torch
from torch.utils.data import IterableDataset
from typing import Optional, Dict, Iterator
import random


class SimpleStreamingDataset(IterableDataset):
    """Stream domains from a single pre-shuffled text file"""
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 64,
        buffer_size: int = 10000,
    ):
        """
        Args:
            file_path: Path to text file with one domain per line
            tokenizer: DomainBertTokenizerFast instance
            max_length: Maximum sequence length
            buffer_size: Size of tokenization buffer
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        
        # Estimate size (will be updated when file is created)
        self.estimated_size = 1_700_000_000
    
    def __len__(self):
        """Return estimated dataset size"""
        return self.estimated_size
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream and tokenize domains"""
        buffer = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                domain = line.strip()
                if domain:
                    buffer.append(domain)
                    
                    # Tokenize when buffer is full
                    if len(buffer) >= self.buffer_size:
                        # Tokenize batch
                        encoded = self.tokenizer(
                            buffer,
                            max_length=self.max_length,
                            truncation=True,
                            padding='max_length',
                            return_tensors=None
                        )
                        
                        # Yield individual examples
                        for i in range(len(buffer)):
                            yield {
                                'input_ids': encoded['input_ids'][i],
                                'attention_mask': encoded['attention_mask'][i],
                                'token_type_ids': encoded['token_type_ids'][i],
                                'tld_ids': encoded['tld_ids'][i]
                            }
                        
                        buffer = []
            
            # Process remaining buffer
            if buffer:
                encoded = self.tokenizer(
                    buffer,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors=None
                )
                
                for i in range(len(buffer)):
                    yield {
                        'input_ids': encoded['input_ids'][i],
                        'attention_mask': encoded['attention_mask'][i],
                        'token_type_ids': encoded['token_type_ids'][i],
                        'tld_ids': encoded['tld_ids'][i]
                    }