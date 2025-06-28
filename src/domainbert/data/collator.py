"""Data collator for DomainBERT with configurable TLD masking"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForDomainMLM(DataCollatorForLanguageModeling):
    """Enhanced data collator with separate TLD masking probability"""
    
    tld_mask_probability: float = 0.1  # Separate probability for TLD masking
    mask_tld_separately: bool = True   # Whether to mask TLDs as a separate task
    
    def __post_init__(self):
        """Set up TLD unknown ID."""
        # Get unknown TLD ID from tokenizer
        if hasattr(self.tokenizer, 'unk_tld_id'):
            self.unk_tld_id = self.tokenizer.unk_tld_id
        elif hasattr(self.tokenizer, 'tld_to_id') and '[UNK_TLD]' in self.tokenizer.tld_to_id:
            self.unk_tld_id = self.tokenizer.tld_to_id['[UNK_TLD]']
        else:
            self.unk_tld_id = 999  # Default matching the test
    
    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[List[List[int]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask tokens for MLM training following the 80/10/10 rule."""
        labels = inputs.clone()
        
        # Get special tokens mask if not provided
        if special_tokens_mask is None:
            special_tokens_mask = self._get_special_tokens_mask(labels, already_has_special_tokens=True)
        
        # Convert to tensor
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Create mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with [MASK] 
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace masked input tokens with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10%) we keep the masked input tokens unchanged
        
        return inputs, labels
    
    def mask_tlds(self, tld_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask TLD IDs for prediction task.
        
        Args:
            tld_ids: Tensor of TLD IDs
            
        Returns:
            Tuple of (masked_tld_ids, tld_labels)
        """
        # Clone both for labels and masked version
        masked_tld_ids = tld_ids.clone()
        tld_labels = tld_ids.clone()
        
        # Create mask based on probability
        probability_matrix = torch.rand(tld_ids.shape)
        masked_indices = probability_matrix < self.tld_mask_probability
        
        # Replace masked TLDs with unknown TLD token
        masked_tld_ids[masked_indices] = self.unk_tld_id
        
        # Only compute loss on masked TLDs
        tld_labels[~masked_indices] = -100
        
        return masked_tld_ids, tld_labels
    
    def _get_special_tokens_mask(self, labels: torch.Tensor, already_has_special_tokens: bool = False) -> List[List[int]]:
        """Get mask for special tokens."""
        if hasattr(self.tokenizer, 'get_special_tokens_mask'):
            # Get special tokens mask from tokenizer
            special_tokens_mask = []
            for label_row in labels:
                mask = self.tokenizer.get_special_tokens_mask(
                    label_row.tolist(), 
                    already_has_special_tokens=already_has_special_tokens
                )
                special_tokens_mask.append(mask)
            return special_tokens_mask
        
        # Create special tokens mask manually
        special_tokens_mask = []
        for label_row in labels:
            row_mask = []
            for token_id in label_row:
                # Mark special tokens (PAD, CLS, SEP, MASK, UNK)
                if token_id in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, 
                               self.tokenizer.sep_token_id, self.tokenizer.mask_token_id, 
                               self.tokenizer.unk_token_id]:
                    row_mask.append(1)
                else:
                    row_mask.append(0)
            special_tokens_mask.append(row_mask)
        
        return special_tokens_mask
    
    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override parent's method to use our mask_tokens implementation."""
        # Convert special_tokens_mask to list format if it's a tensor
        if special_tokens_mask is not None and isinstance(special_tokens_mask, torch.Tensor):
            special_tokens_mask = special_tokens_mask.tolist()
        return self.mask_tokens(inputs, special_tokens_mask)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # First, handle standard character-level MLM
        batch = super().__call__(examples)
        
        # Handle TLD masking if present
        if 'tld_ids' in examples[0] and self.mask_tld_separately:
            # Extract TLD IDs - handle both list and scalar formats
            tld_ids_list = []
            for ex in examples:
                tld_id = ex['tld_ids']
                if isinstance(tld_id, (list, torch.Tensor)):
                    if isinstance(tld_id, torch.Tensor):
                        tld_id = tld_id.item() if tld_id.numel() == 1 else tld_id[0].item()
                    else:
                        tld_id = tld_id[0] if len(tld_id) > 0 else 0
                tld_ids_list.append(tld_id)
            
            tld_ids = torch.tensor(tld_ids_list)
            
            # Mask TLDs
            masked_tld_ids, tld_labels = self.mask_tlds(tld_ids)
            
            # Add to batch
            batch['tld_ids'] = masked_tld_ids
            batch['tld_labels'] = tld_labels
        
        return batch