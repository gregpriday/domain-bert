"""
Unit tests for DomainBERT models.

Tests model components including embeddings, forward passes, and loss computation.
"""
import pytest
import torch
import tempfile
import json
from pathlib import Path

from domainbert.config import DomainBertConfig
from domainbert.model import (
    DomainEmbeddings,
    DomainBertModel,
    DomainBertForMaskedLM,
    DomainBertForSequenceClassification
)


class TestDomainEmbeddings:
    """Test the custom domain embeddings layer."""
    
    def test_embedding_initialization(self):
        """Test embedding layer can be initialized."""
        config = DomainBertConfig(
            vocab_size=133,  # 128 ASCII + 5 special tokens
            hidden_size=256,
            max_position_embeddings=128,
            type_vocab_size=4,  # domain, subdomain, TLD, separator
            tld_vocab_size=1000
        )
        
        embeddings = DomainEmbeddings(config)
        
        assert embeddings.char_embeddings.num_embeddings == 133
        assert embeddings.char_embeddings.embedding_dim == 256
        assert embeddings.position_embeddings.num_embeddings == 128
        assert embeddings.token_type_embeddings.num_embeddings == 4
        assert embeddings.tld_embeddings.num_embeddings == 1000
    
    def test_embedding_forward_pass(self):
        """Test forward pass through embeddings."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=256,
            max_position_embeddings=128,
            type_vocab_size=4,
            tld_vocab_size=100
        )
        
        embeddings = DomainEmbeddings(config)
        
        # Create dummy inputs
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 133, (batch_size, seq_length))
        token_type_ids = torch.randint(0, 4, (batch_size, seq_length))
        tld_ids = torch.randint(0, 100, (batch_size,))
        
        # Forward pass
        output = embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            tld_ids=tld_ids
        )
        
        assert output.shape == (batch_size, seq_length, 256)
    
    def test_embedding_without_tld_ids(self):
        """Test embeddings work without TLD IDs."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=256,
            tld_vocab_size=100
        )
        
        embeddings = DomainEmbeddings(config)
        
        input_ids = torch.randint(0, 133, (2, 10))
        output = embeddings(input_ids=input_ids)
        
        assert output.shape == (2, 10, 256)
    
    def test_tld_embedding_broadcast(self):
        """Test TLD embeddings are correctly broadcast."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=128,  # Smaller for easier testing
            tld_vocab_size=10
        )
        
        embeddings = DomainEmbeddings(config)
        
        # Set TLD embeddings to known values for testing
        with torch.no_grad():
            embeddings.tld_embeddings.weight.fill_(0)
            embeddings.tld_embeddings.weight[0] = torch.ones(config.tld_embed_dim)
            embeddings.tld_embeddings.weight[1] = torch.ones(config.tld_embed_dim) * 2
        
        input_ids = torch.zeros((2, 5), dtype=torch.long)
        tld_ids = torch.tensor([0, 1])
        
        output = embeddings(input_ids=input_ids, tld_ids=tld_ids)
        
        # Check that TLD embedding affects the output differently for different TLD IDs
        # The exact values depend on the projection layer, so just check they're different
        assert not torch.allclose(output[0], output[1], atol=1e-5)


class TestDomainBertModel:
    """Test the base DomainBERT model."""
    
    def test_model_initialization(self):
        """Test model can be initialized with config."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=256,
            num_hidden_layers=12,
            num_attention_heads=8,
            intermediate_size=1024
        )
        
        model = DomainBertModel(config)
        
        assert model.config.hidden_size == 256
        assert len(model.encoder.layer) == 12
    
    def test_model_forward_pass(self):
        """Test forward pass through model."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=256,
            num_hidden_layers=2,  # Smaller for testing
            num_attention_heads=8
        )
        
        model = DomainBertModel(config)
        model.eval()
        
        # Create inputs
        batch_size = 2
        seq_length = 20
        input_ids = torch.randint(0, 133, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        assert hasattr(outputs, "last_hidden_state")
        assert hasattr(outputs, "pooler_output")
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, 256)
        assert outputs.pooler_output.shape == (batch_size, 256)
    
    def test_model_with_all_inputs(self):
        """Test model with all possible inputs."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=128,
            num_hidden_layers=2,
            tld_vocab_size=50
        )
        
        model = DomainBertModel(config)
        model.eval()
        
        # Create all inputs
        input_ids = torch.randint(0, 133, (2, 15))
        attention_mask = torch.ones(2, 15)
        token_type_ids = torch.randint(0, 4, (2, 15))
        tld_ids = torch.randint(0, 50, (2,))
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tld_ids=tld_ids
            )
        
        assert outputs.last_hidden_state is not None
        assert outputs.pooler_output is not None


class TestDomainBertForMaskedLM:
    """Test the masked language modeling head."""
    
    def test_mlm_model_initialization(self):
        """Test MLM model initialization."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=256,
            tld_vocab_size=100,
            mlm_weight=0.85,
            tld_weight=0.15
        )
        
        model = DomainBertForMaskedLM(config)
        
        assert hasattr(model, "domain_bert")
        assert hasattr(model, "mlm_predictions")
        assert hasattr(model, "tld_classifier")
        assert model.config.mlm_weight == 0.85
        assert model.config.tld_weight == 0.15
    
    def test_mlm_forward_pass(self):
        """Test forward pass with labels."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=128,
            num_hidden_layers=2,
            tld_vocab_size=10
        )
        
        model = DomainBertForMaskedLM(config)
        model.eval()
        
        # Create inputs
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 133, (batch_size, seq_length))
        labels = input_ids.clone()
        labels[labels == 0] = -100  # Mask padding tokens
        tld_labels = torch.randint(0, 10, (batch_size,))
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                tld_labels=tld_labels
            )
        
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.loss is not None
        assert outputs.loss > 0
        
        assert outputs.logits.shape == (batch_size, seq_length, 133)
    
    def test_mlm_loss_computation(self):
        """Test that losses are computed correctly."""
        config = DomainBertConfig(
            vocab_size=50,
            hidden_size=64,
            num_hidden_layers=1,
            tld_vocab_size=5,
            mlm_weight=0.5,
            tld_weight=0.5
        )
        
        model = DomainBertForMaskedLM(config)
        
        # Create simple inputs
        input_ids = torch.randint(0, 50, (2, 5))
        labels = torch.randint(0, 50, (2, 5))
        tld_labels = torch.randint(0, 5, (2,))
        
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            tld_labels=tld_labels
        )
        
        # Check losses exist and are reasonable
        assert outputs.loss > 0
        assert outputs.logits is not None
        assert outputs.logits.shape == (2, 5, 50)  # batch_size, seq_len, vocab_size
    
    def test_mlm_without_labels(self):
        """Test model works without labels (inference mode)."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=128,
            num_hidden_layers=2,
            tld_vocab_size=10
        )
        
        model = DomainBertForMaskedLM(config)
        model.eval()
        
        input_ids = torch.randint(0, 133, (2, 10))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs.loss is None
        assert outputs.logits is not None
        assert outputs.logits.shape == (2, 10, 133)  # batch_size, seq_len, vocab_size


class TestDomainBertForSequenceClassification:
    """Test the sequence classification model."""
    
    def test_classification_model_initialization(self):
        """Test classification model initialization."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=256,
            num_labels=2  # Binary classification
        )
        
        model = DomainBertForSequenceClassification(config)
        
        assert hasattr(model, "domain_bert")
        assert hasattr(model, "classifier")
        assert model.classifier.out_features == 2
    
    def test_classification_forward_pass(self):
        """Test forward pass for classification."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=128,
            num_hidden_layers=2,
            num_labels=3  # 3-class classification
        )
        
        model = DomainBertForSequenceClassification(config)
        model.eval()
        
        # Create inputs
        input_ids = torch.randint(0, 133, (4, 15))
        labels = torch.randint(0, 3, (4,))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (4, 3)
        assert outputs.loss > 0
    
    def test_classification_without_labels(self):
        """Test classification in inference mode."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=128,
            num_hidden_layers=2,
            num_labels=2
        )
        
        model = DomainBertForSequenceClassification(config)
        model.eval()
        
        input_ids = torch.randint(0, 133, (2, 10))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs.loss is None
        assert outputs.logits.shape == (2, 2)
    
    def test_save_and_load_model(self):
        """Test saving and loading model preserves weights."""
        torch.manual_seed(42)  # Set seed for reproducibility
        
        config = DomainBertConfig(
            vocab_size=50,
            hidden_size=64,
            num_hidden_layers=1,
            num_labels=2
        )
        
        model = DomainBertForSequenceClassification(config)
        model.eval()  # Set to eval mode to avoid dropout
        
        # Get some predictions
        input_ids = torch.randint(0, 50, (2, 5))
        with torch.no_grad():
            original_outputs = model(input_ids=input_ids)
        
        # Save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir)
            
            # Also save config
            config.save_pretrained(temp_dir)
            
            loaded_model = DomainBertForSequenceClassification.from_pretrained(temp_dir)
            loaded_model.eval()  # Set to eval mode
            
            # Check predictions match
            with torch.no_grad():
                loaded_outputs = loaded_model(input_ids=input_ids)
            
            assert torch.allclose(original_outputs.logits, loaded_outputs.logits, atol=1e-5)