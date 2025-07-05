"""
Integration tests for the pretraining pipeline.

Tests end-to-end training steps including data loading, model forward/backward passes.
"""
import pytest
import torch
import tempfile
import json
from pathlib import Path

from domainbert.config import DomainBertConfig
from domainbert.model import DomainBertForMaskedLM
from domainbert.tokenizer import DomainBertTokenizerFast
from domainbert.data.collator import DataCollatorForDomainMLM


class TestPretrainingIntegration:
    """Test the complete pretraining pipeline."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory with model files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config
            config = DomainBertConfig(
                vocab_size=133,
                hidden_size=64,  # Small for testing
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=128,
                max_position_embeddings=64,
                tld_vocab_size=10
            )
            config.save_pretrained(temp_dir)
            
            # Create tokenizer config
            tokenizer_config = {
                "model_max_length": 64,
                "tokenizer_class": "DomainBertTokenizerFast"
            }
            with open(temp_path / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f)
            
            # Create tokenizer.json
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Sequence", "pretokenizers": []},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "vocab": {
                        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, 
                        "[SEP]": 3, "[MASK]": 4
                    },
                    "merges": []
                }
            }
            
            # Add ASCII characters
            for i in range(128):
                tokenizer_json["model"]["vocab"][chr(i)] = i + 5
            
            with open(temp_path / "tokenizer.json", "w") as f:
                json.dump(tokenizer_json, f)
            
            # Create TLD vocabulary
            tld_vocab = {
                "com": 0, "net": 1, "org": 2, "edu": 3, "gov": 4,
                "io": 5, "co": 6, "uk": 7, "de": 8, "[UNK_TLD]": 9
            }
            with open(temp_path / "tld_vocab.json", "w") as f:
                json.dump(tld_vocab, f)
            
            # Create special tokens map
            special_tokens = {
                "unk_token": "[UNK]", "sep_token": "[SEP]",
                "pad_token": "[PAD]", "cls_token": "[CLS]",
                "mask_token": "[MASK]"
            }
            with open(temp_path / "special_tokens_map.json", "w") as f:
                json.dump(special_tokens, f)
            
            # Create and save a model
            model = DomainBertForMaskedLM(config)
            model.save_pretrained(temp_dir, safe_serialization=False)
            
            yield temp_path
    
    def test_single_training_step(self, temp_model_dir):
        """Test a single forward/backward pass."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Load model and tokenizer
        model = DomainBertForMaskedLM.from_pretrained(temp_model_dir)
        tokenizer = DomainBertTokenizerFast.from_pretrained(temp_model_dir)
        
        # Create data collator with higher masking probability for testing
        collator = DataCollatorForDomainMLM(
            tokenizer=tokenizer,
            mlm_probability=0.30,  # Higher probability to ensure some tokens are masked
            tld_mask_probability=0.5  # Higher probability to ensure some TLDs are masked
        )
        
        # Create sample batch
        domains = ["example.com", "test.org", "subdomain.test.net", "another.edu"]
        encoded = tokenizer(domains, padding=True, truncation=True, return_tensors="pt")
        
        # Prepare batch with masking
        batch = collator([{
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "token_type_ids": encoded["token_type_ids"][i],
            "tld_ids": encoded["tld_ids"][i:i+1]
        } for i in range(len(domains))])
        
        # Forward pass
        outputs = model(**batch)
        
        # Check outputs
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.loss is not None
        assert outputs.loss.requires_grad
        # Check loss is not NaN and is positive
        assert not torch.isnan(outputs.loss), f"Loss is NaN. Batch keys: {batch.keys()}"
        assert outputs.loss.item() > 0
        
        # Backward pass
        outputs.loss.backward()
        
        # Check gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_training_step_with_optimizer(self, temp_model_dir):
        """Test training step with optimizer update."""
        # Load model and tokenizer
        model = DomainBertForMaskedLM.from_pretrained(temp_model_dir)
        tokenizer = DomainBertTokenizerFast.from_pretrained(temp_model_dir)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Create data collator
        collator = DataCollatorForDomainMLM(tokenizer=tokenizer)
        
        # Get initial loss
        domains = ["example.com", "test.org"]
        encoded = tokenizer(domains, padding=True, return_tensors="pt")
        
        batch = collator([{
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "token_type_ids": encoded["token_type_ids"][i],
            "tld_ids": encoded["tld_ids"][i:i+1]
        } for i in range(len(domains))])
        
        # Initial forward pass
        model.train()
        initial_outputs = model(**batch)
        initial_loss = initial_outputs.loss.item()
        
        # Training step
        optimizer.zero_grad()
        initial_outputs.loss.backward()
        optimizer.step()
        
        # Forward pass after update
        final_outputs = model(**batch)
        final_loss = final_outputs.loss.item()
        
        # Loss should change after optimization step
        assert initial_loss != final_loss
    
    def test_batch_size_handling(self, temp_model_dir):
        """Test model handles different batch sizes."""
        model = DomainBertForMaskedLM.from_pretrained(temp_model_dir)
        tokenizer = DomainBertTokenizerFast.from_pretrained(temp_model_dir)
        collator = DataCollatorForDomainMLM(tokenizer=tokenizer)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            domains = ["example.com"] * batch_size
            encoded = tokenizer(domains, padding=True, return_tensors="pt")
            
            batch = collator([{
                "input_ids": encoded["input_ids"][i],
                "attention_mask": encoded["attention_mask"][i],
                "token_type_ids": encoded["token_type_ids"][i],
                "tld_ids": encoded["tld_ids"][i:i+1]
            } for i in range(batch_size)])
            
            outputs = model(**batch)
            
            assert outputs.loss is not None
            assert outputs.logits.shape[0] == batch_size
    
    def test_gradient_accumulation(self, temp_model_dir):
        """Test gradient accumulation over multiple steps."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        model = DomainBertForMaskedLM.from_pretrained(temp_model_dir)
        tokenizer = DomainBertTokenizerFast.from_pretrained(temp_model_dir)
        collator = DataCollatorForDomainMLM(
            tokenizer=tokenizer,
            mlm_probability=0.30,  # Higher probability to ensure masking
            tld_mask_probability=0.5
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Accumulate gradients over 3 steps
        accumulation_steps = 3
        model.train()
        optimizer.zero_grad()
        
        accumulated_loss = 0
        for step in range(accumulation_steps):
            # Different domains for each step
            domains = [f"test{step}.com", f"example{step}.org"]
            encoded = tokenizer(domains, padding=True, return_tensors="pt")
            
            batch = collator([{
                "input_ids": encoded["input_ids"][i],
                "attention_mask": encoded["attention_mask"][i],
                "token_type_ids": encoded["token_type_ids"][i],
                "tld_ids": encoded["tld_ids"][i:i+1]
            } for i in range(len(domains))])
            
            outputs = model(**batch)
            
            # Skip if loss is NaN (can happen with random masking)
            if torch.isnan(outputs.loss):
                continue
                
            loss = outputs.loss / accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
        
        # Check gradients accumulated
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        assert len(grad_norms) > 0
        # At least some parameters should have non-zero gradients
        assert any(norm > 0 for norm in grad_norms), "No parameters have gradients"
        # Check that the accumulated loss is reasonable
        assert accumulated_loss > 0
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
    
    def test_mixed_precision_compatibility(self, temp_model_dir):
        """Test model works with mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        model = DomainBertForMaskedLM.from_pretrained(temp_model_dir).cuda()
        tokenizer = DomainBertTokenizerFast.from_pretrained(temp_model_dir)
        collator = DataCollatorForDomainMLM(tokenizer=tokenizer)
        
        # Create mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Prepare batch
        domains = ["example.com", "test.org"]
        encoded = tokenizer(domains, padding=True, return_tensors="pt")
        
        batch = collator([{
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "token_type_ids": encoded["token_type_ids"][i],
            "tld_ids": encoded["tld_ids"][i:i+1]
        } for i in range(len(domains))])
        
        # Move to GPU
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # Mixed precision forward/backward
        model.train()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check model still works
        assert not torch.isnan(loss).any()
    
    def test_model_saving_and_loading(self, temp_model_dir):
        """Test model can be saved and loaded during training."""
        model = DomainBertForMaskedLM.from_pretrained(temp_model_dir)
        tokenizer = DomainBertTokenizerFast.from_pretrained(temp_model_dir)
        
        # Train for one step
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        collator = DataCollatorForDomainMLM(tokenizer=tokenizer)
        
        domains = ["example.com", "test.org"]
        encoded = tokenizer(domains, padding=True, return_tensors="pt")
        
        batch = collator([{
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "token_type_ids": encoded["token_type_ids"][i],
            "tld_ids": encoded["tld_ids"][i:i+1]
        } for i in range(len(domains))])
        
        model.train()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        
        # Save model
        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir, safe_serialization=False)
            tokenizer.save_pretrained(save_dir)
            
            # Load model
            loaded_model = DomainBertForMaskedLM.from_pretrained(save_dir)
            loaded_tokenizer = DomainBertTokenizerFast.from_pretrained(save_dir)
            
            # Check predictions match
            model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                original_outputs = model(**batch)
                loaded_outputs = loaded_model(**batch)
            
            assert torch.allclose(
                original_outputs.logits,
                loaded_outputs.logits,
                atol=1e-5
            )