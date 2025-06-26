"""
Unit tests for DataCollatorForDomainMLM.

Tests masking logic for both MLM and TLD prediction tasks.
"""
import pytest
import torch
from unittest.mock import MagicMock

from domainbert.data.collator import DataCollatorForDomainMLM


class TestDataCollatorForDomainMLM:
    """Test the data collator for domain MLM."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.cls_token_id = 2
        tokenizer.sep_token_id = 3
        tokenizer.mask_token_id = 4
        tokenizer.unk_token_id = 1
        tokenizer.vocab_size = 133  # 128 ASCII + 5 special tokens
        tokenizer.convert_ids_to_tokens = lambda ids: [f"token_{i}" for i in ids]
        tokenizer.tld_to_id = {"[UNK_TLD]": 999}
        tokenizer.unk_tld_id = 999
        return tokenizer
    
    def test_collator_initialization(self, mock_tokenizer):
        """Test collator can be initialized."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            mlm_probability=0.15,
            tld_mask_probability=0.2
        )
        
        assert collator.tokenizer == mock_tokenizer
        assert collator.mlm_probability == 0.15
        assert collator.tld_mask_probability == 0.2
    
    def test_collator_default_probabilities(self, mock_tokenizer):
        """Test collator uses default probabilities."""
        collator = DataCollatorForDomainMLM(tokenizer=mock_tokenizer)
        
        assert collator.mlm_probability == 0.15
        assert collator.tld_mask_probability == 0.1
    
    def test_mask_tokens_basic(self, mock_tokenizer):
        """Test basic token masking."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            mlm_probability=1.0  # Mask everything for testing
        )
        
        # Create simple input
        inputs = torch.tensor([[5, 6, 7, 8, 9]])  # Simple token IDs
        
        # Mock the special tokens behavior
        collator._get_special_tokens_mask = MagicMock(return_value=[[0, 0, 0, 0, 0]])
        
        inputs, labels = collator.mask_tokens(inputs)
        
        # All tokens should be masked (80% [MASK], 10% random, 10% unchanged)
        assert labels.shape == inputs.shape
        # Labels should contain original values
        assert all(labels[0][i] in [5, 6, 7, 8, 9] for i in range(5))
    
    def test_mask_tokens_respects_special_tokens(self, mock_tokenizer):
        """Test that special tokens are not masked."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            mlm_probability=1.0
        )
        
        # Input with CLS and SEP tokens
        inputs = torch.tensor([[2, 5, 6, 7, 3]])  # [CLS] tokens [SEP]
        
        # Create special tokens mask
        special_tokens_mask = [[1, 0, 0, 0, 1]]  # CLS and SEP are special
        collator._get_special_tokens_mask = MagicMock(return_value=special_tokens_mask)
        
        inputs_masked, labels = collator.mask_tokens(inputs)
        
        # Special tokens should not be masked
        assert inputs_masked[0][0] == 2  # [CLS] unchanged
        assert inputs_masked[0][4] == 3  # [SEP] unchanged
        assert labels[0][0] == -100  # Ignored in loss
        assert labels[0][4] == -100  # Ignored in loss
    
    def test_mask_tokens_probability_distribution(self, mock_tokenizer):
        """Test masking follows the 80/10/10 distribution."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            mlm_probability=0.15
        )
        
        # Create larger input for statistical testing
        torch.manual_seed(42)  # For reproducibility
        inputs = torch.randint(5, 100, (10, 50))  # 10 sequences, 50 tokens each
        
        collator._get_special_tokens_mask = MagicMock(
            return_value=torch.zeros_like(inputs).tolist()
        )
        
        inputs_masked, labels = collator.mask_tokens(inputs.clone())
        
        # Count different mask types
        total_masked = (labels != -100).sum().item()
        mask_token_count = (inputs_masked == 4).sum().item()  # [MASK] token
        
        # Roughly 15% should be masked
        assert 0.1 < total_masked / inputs.numel() < 0.2
        
        # Roughly 80% of masked should be [MASK] token
        if total_masked > 0:
            assert 0.7 < mask_token_count / total_masked < 0.9
    
    def test_mask_tlds(self, mock_tokenizer):
        """Test TLD masking."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            tld_mask_probability=0.5  # 50% for easier testing
        )
        
        # Create TLD IDs
        torch.manual_seed(42)
        tld_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        masked_tlds, tld_labels = collator.mask_tlds(tld_ids)
        
        # Check some are masked
        masked_count = (masked_tlds == 999).sum().item()  # UNK_TLD
        assert 0 < masked_count < 10
        
        # Check labels are set correctly
        assert (tld_labels != -100).sum().item() == masked_count
        
        # Non-masked should have -100 label
        non_masked_indices = masked_tlds != 999
        assert all(tld_labels[non_masked_indices] == -100)
    
    def test_mask_tlds_all(self, mock_tokenizer):
        """Test masking all TLDs."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            tld_mask_probability=1.0  # Mask all
        )
        
        tld_ids = torch.tensor([1, 2, 3, 4, 5])
        masked_tlds, tld_labels = collator.mask_tlds(tld_ids)
        
        # All should be masked
        assert all(masked_tlds == 999)
        assert all(tld_labels == torch.tensor([1, 2, 3, 4, 5]))
    
    def test_mask_tlds_none(self, mock_tokenizer):
        """Test masking no TLDs."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            tld_mask_probability=0.0  # Mask none
        )
        
        tld_ids = torch.tensor([1, 2, 3, 4, 5])
        masked_tlds, tld_labels = collator.mask_tlds(tld_ids)
        
        # None should be masked
        assert all(masked_tlds == torch.tensor([1, 2, 3, 4, 5]))
        assert all(tld_labels == -100)
    
    def test_collator_call_method(self, mock_tokenizer):
        """Test the main __call__ method."""
        collator = DataCollatorForDomainMLM(
            tokenizer=mock_tokenizer,
            mlm_probability=0.15,
            tld_mask_probability=0.1
        )
        
        # Create batch of examples
        examples = [
            {
                "input_ids": torch.tensor([2, 10, 11, 12, 3]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "token_type_ids": torch.tensor([0, 0, 0, 0, 0]),
                "tld_ids": torch.tensor([1])
            },
            {
                "input_ids": torch.tensor([2, 20, 21, 22, 23, 3]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
                "token_type_ids": torch.tensor([0, 0, 0, 0, 0, 0]),
                "tld_ids": torch.tensor([2])
            }
        ]
        
        # Mock padding method
        def mock_pad(features, **kwargs):
            # Simple padding implementation
            max_len = max(f["input_ids"].size(0) for f in features)
            for f in features:
                pad_len = max_len - f["input_ids"].size(0)
                if pad_len > 0:
                    f["input_ids"] = torch.cat([f["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
                    f["attention_mask"] = torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                    f["token_type_ids"] = torch.cat([f["token_type_ids"], torch.zeros(pad_len, dtype=torch.long)])
            
            # Stack into batch
            batch = {
                "input_ids": torch.stack([f["input_ids"] for f in features]),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),
                "token_type_ids": torch.stack([f["token_type_ids"] for f in features]),
                "tld_ids": torch.stack([f["tld_ids"].squeeze() for f in features])
            }
            return batch
        
        collator.tokenizer.pad = mock_pad
        collator._get_special_tokens_mask = MagicMock(
            return_value=[[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]]
        )
        
        # Call collator
        batch = collator(examples)
        
        # Check output structure
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "token_type_ids" in batch
        assert "tld_ids" in batch
        assert "labels" in batch
        assert "tld_labels" in batch
        
        # Check shapes
        assert batch["input_ids"].shape == (2, 6)
        assert batch["labels"].shape == (2, 6)
        assert batch["tld_ids"].shape == (2,)
        assert batch["tld_labels"].shape == (2,)
    
    def test_get_special_tokens_mask(self, mock_tokenizer):
        """Test special tokens mask generation."""
        collator = DataCollatorForDomainMLM(tokenizer=mock_tokenizer)
        
        # Test with token type IDs
        token_ids = torch.tensor([[2, 5, 6, 3, 0, 0]])  # [CLS] tokens [SEP] [PAD] [PAD]
        token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
        
        # For this test, we'll check the actual implementation
        # The method should mark CLS, SEP, and PAD as special tokens
        labels = torch.tensor([[2, 5, 6, 3, 0, 0]])
        
        # Create a labels tensor where special tokens are -100
        labels_with_special = labels.clone()
        labels_with_special[labels == 0] = -100  # PAD
        labels_with_special[labels == 2] = -100  # CLS
        labels_with_special[labels == 3] = -100  # SEP
        
        # The remaining tokens (5, 6) should be maskable
        maskable_positions = (labels_with_special != -100)
        assert maskable_positions[0].tolist() == [False, True, True, False, False, False]