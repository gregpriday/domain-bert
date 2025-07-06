"""Fixed data collator for DomainBERT with proper masking"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForDomainMLM(DataCollatorForLanguageModeling):
    """Fixed data collator with correct masking strategy"""
    
    mlm_probability: float = 0.30  # Back to 30% for better GPU utilization
    tld_mask_probability: float = 0.1
    mask_tld_separately: bool = True
    mask_tld_tokens_in_sequence: bool = False  # Never mask TLD tokens in sequence
    
    def __post_init__(self):
        """Set up TLD unknown ID."""
        # Get unknown TLD ID from tokenizer
        if hasattr(self.tokenizer, 'unk_tld_id'):
            self.unk_tld_id = self.tokenizer.unk_tld_id
        elif hasattr(self.tokenizer, 'tld_to_id') and '[UNK_TLD]' in self.tokenizer.tld_to_id:
            self.unk_tld_id = self.tokenizer.tld_to_id['[UNK_TLD]']
        else:
            self.unk_tld_id = 1
    
    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed masking that properly handles the 90/10 split"""
        labels = inputs.clone()
        
        # Get special tokens mask
        if special_tokens_mask is None:
            special_tokens_mask = self._get_special_tokens_mask(labels, already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # Create TLD mask (tokens with ID >= 44, excluding special tokens)
        is_tld_token = (inputs >= 44) & ~special_tokens_mask
        
        # Create dot mask (never mask dots)
        is_dot = inputs == 43  # '.' token
        
        # Combine masks: can't mask special tokens, TLD tokens, or dots
        non_maskable = special_tokens_mask | is_tld_token | is_dot
        
        # For each sequence in the batch
        batch_size, seq_len = inputs.shape
        
        for i in range(batch_size):
            # Find maskable positions
            maskable_positions = ~non_maskable[i]
            maskable_indices = maskable_positions.nonzero(as_tuple=True)[0]
            
            if len(maskable_indices) > 0:
                # Calculate number to mask (15% of maskable, minimum 1)
                num_to_mask = max(1, int(round(self.mlm_probability * len(maskable_indices))))
                
                # Randomly select positions to mask
                selected_indices = maskable_indices[
                    torch.randperm(len(maskable_indices))[:num_to_mask]
                ]
                
                # For each selected position, decide what to do
                for idx in selected_indices:
                    # 90% of the time, replace with [MASK]
                    if torch.rand(1).item() < 0.9:
                        inputs[i, idx] = self.tokenizer.mask_token_id
                    # 10% of the time, keep original token
                    # (but still predict it - label stays as original token)
        
        # Set labels to -100 for all non-masked positions
        # A position is "masked" if we're predicting it, regardless of whether
        # we replaced it with [MASK] or kept the original
        labels[non_maskable] = -100
        
        # Also set labels to -100 for positions we didn't select for masking
        for i in range(batch_size):
            maskable_positions = ~non_maskable[i]
            maskable_indices = maskable_positions.nonzero(as_tuple=True)[0]
            
            if len(maskable_indices) > 0:
                num_to_mask = max(1, int(round(self.mlm_probability * len(maskable_indices))))
                selected_indices = maskable_indices[
                    torch.randperm(len(maskable_indices))[:num_to_mask]
                ]
                
                # Create mask for selected positions
                selected_mask = torch.zeros(seq_len, dtype=torch.bool)
                selected_mask[selected_indices] = True
                
                # Set labels to -100 for non-selected maskable positions
                unselected = maskable_positions & ~selected_mask
                labels[i, unselected] = -100
        
        return inputs, labels
    
    def mask_tlds(self, tld_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask TLD IDs for CLS prediction task."""
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
        special_tokens_mask = []
        for label_row in labels:
            row_mask = []
            for token_id in label_row:
                # Mark special tokens
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
        if special_tokens_mask is not None and isinstance(special_tokens_mask, torch.Tensor):
            special_tokens_mask = special_tokens_mask.tolist()
        return self.mask_tokens(inputs, special_tokens_mask)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # First, handle standard character-level MLM
        batch = super().__call__(examples)
        
        # Handle TLD masking if present
        if 'tld_ids' in examples[0] and self.mask_tld_separately:
            # Extract TLD IDs
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
            
            # Mask TLDs for CLS prediction
            masked_tld_ids, tld_labels = self.mask_tlds(tld_ids)
            
            # Add to batch
            batch['tld_ids'] = masked_tld_ids
            batch['tld_labels'] = tld_labels
        
        return batch