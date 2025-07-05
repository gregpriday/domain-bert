"""Data collator for DomainBERT with configurable TLD masking"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForDomainMLM(DataCollatorForLanguageModeling):
    """Enhanced data collator with domain-specific masking strategy"""
    
    mlm_probability: float = 0.30  # 30% masking for non-TLD tokens
    tld_mask_probability: float = 0.1  # Separate probability for TLD masking
    mask_tld_separately: bool = True   # Whether to mask TLDs as a separate task
    mask_tld_tokens_in_sequence: bool = False  # Never mask TLD tokens in the sequence
    
    def __post_init__(self):
        """Set up TLD unknown ID."""
        # Get unknown TLD ID from tokenizer
        if hasattr(self.tokenizer, 'unk_tld_id'):
            self.unk_tld_id = self.tokenizer.unk_tld_id
        elif hasattr(self.tokenizer, 'tld_to_id') and '[UNK_TLD]' in self.tokenizer.tld_to_id:
            self.unk_tld_id = self.tokenizer.tld_to_id['[UNK_TLD]']
        elif hasattr(self.tokenizer, 'tld_to_id') and '<UNK>' in self.tokenizer.tld_to_id:
            self.unk_tld_id = self.tokenizer.tld_to_id['<UNK>']
        else:
            self.unk_tld_id = 1  # Default to 1 which is the UNK token in our vocab
    
    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[List[List[int]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask tokens for MLM training following domain-specific strategy.
        
        - Masks 30% of non-TLD tokens (characters and dots)
        - Never masks TLD tokens in the sequence
        - Uses 90/10 rule (90% mask, 10% keep)
        - Ensures at least 1 token is masked per domain
        """
        labels = inputs.clone()
        
        # Get special tokens mask if not provided
        if special_tokens_mask is None:
            special_tokens_mask = self._get_special_tokens_mask(labels, already_has_special_tokens=True)
        
        # Convert to tensor
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # Create TLD mask to identify TLD tokens in the sequence
        tld_token_mask = self._get_tld_token_mask(inputs)
        
        # Combine masks: don't mask special tokens OR TLD tokens
        combined_mask = special_tokens_mask | tld_token_mask
        
        # For each sequence in the batch
        batch_size, seq_len = inputs.shape
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        
        for i in range(batch_size):
            # Find maskable positions (non-special, non-TLD tokens)
            maskable_positions = ~combined_mask[i]
            maskable_indices = maskable_positions.nonzero(as_tuple=True)[0]
            
            if len(maskable_indices) > 0:
                # Calculate number of tokens to mask (30% of maskable tokens, minimum 1)
                num_to_mask = max(1, int(round(0.30 * len(maskable_indices))))
                
                # Randomly select positions to mask
                mask_positions = maskable_indices[torch.randperm(len(maskable_indices))[:num_to_mask]]
                masked_indices[i, mask_positions] = True
        
        # Set labels to -100 for non-masked tokens
        labels[~masked_indices] = -100
        
        # Apply masking: 90% [MASK], 10% keep original
        replace_mask = torch.bernoulli(torch.full_like(masked_indices.float(), 0.9)).bool()
        indices_replaced = masked_indices & replace_mask
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # The remaining 10% keep their original tokens
        
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
    
    def _get_tld_token_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create mask identifying TLD tokens in the sequence.
        
        Returns tensor with True for TLD tokens, False otherwise.
        """
        tld_mask = torch.zeros_like(inputs, dtype=torch.bool)
        
        # Get all TLD token IDs from vocabulary (excluding special TLD tokens)
        tld_token_ids = set()
        if hasattr(self.tokenizer, 'vocab'):
            vocab = self.tokenizer.vocab
            for token, token_id in vocab.items():
                # TLD tokens are those that match common TLD patterns
                # and are not single characters or special tokens
                if (len(token) > 1 and '.' not in token and 
                    token not in ['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]', '[UNK_TLD]'] and
                    not token.startswith('[') and not token.endswith(']')):
                    # Check if this looks like a TLD (all lowercase letters or contains dot for compound TLDs)
                    if token.isalpha() and token.islower():
                        tld_token_ids.add(token_id)
        
        # Mark positions containing TLD tokens
        for tld_id in tld_token_ids:
            tld_mask |= (inputs == tld_id)
        
        return tld_mask
    
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