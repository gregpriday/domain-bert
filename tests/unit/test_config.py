"""
Unit tests for DomainBertConfig.

Tests configuration loading, validation, and serialization.
"""
import pytest
import tempfile
import json
from pathlib import Path

from domainbert.config import DomainBertConfig


class TestDomainBertConfig:
    """Test the DomainBertConfig class."""
    
    def test_config_initialization(self):
        """Test config can be initialized with default values."""
        config = DomainBertConfig()
        
        # Check default values
        assert config.vocab_size == 133  # 128 ASCII + 5 special tokens
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.intermediate_size == 1024
        assert config.hidden_act == "gelu"
        assert config.hidden_dropout_prob == 0.1
        assert config.attention_probs_dropout_prob == 0.1
        assert config.max_position_embeddings == 128
        assert config.type_vocab_size == 4
        assert config.tld_vocab_size == 1000
        assert config.mlm_weight == 0.85
        assert config.tld_weight == 0.15
        assert config.initializer_range == 0.02
        assert config.layer_norm_eps == 1e-12
    
    def test_config_custom_values(self):
        """Test config initialization with custom values."""
        config = DomainBertConfig(
            vocab_size=200,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=16,
            tld_vocab_size=5000,
            mlm_weight=0.9,
            tld_weight=0.1
        )
        
        assert config.vocab_size == 200
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 16
        assert config.tld_vocab_size == 5000
        assert config.mlm_weight == 0.9
        assert config.tld_weight == 0.1
    
    def test_config_validation(self):
        """Test config validates attention heads."""
        # Hidden size must be divisible by num_attention_heads
        with pytest.raises(ValueError):
            DomainBertConfig(
                hidden_size=256,
                num_attention_heads=7  # 256 not divisible by 7
            )
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = DomainBertConfig(
            vocab_size=150,
            hidden_size=384,
            tld_vocab_size=2000
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["vocab_size"] == 150
        assert config_dict["hidden_size"] == 384
        assert config_dict["tld_vocab_size"] == 2000
        assert config_dict["model_type"] == "domain-bert"
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vocab_size": 160,
            "hidden_size": 320,
            "num_hidden_layers": 8,
            "tld_vocab_size": 3000,
            "mlm_weight": 0.8,
            "tld_weight": 0.2
        }
        
        config = DomainBertConfig(**config_dict)
        
        assert config.vocab_size == 160
        assert config.hidden_size == 320
        assert config.num_hidden_layers == 8
        assert config.tld_vocab_size == 3000
        assert config.mlm_weight == 0.8
        assert config.tld_weight == 0.2
    
    def test_config_save_and_load(self):
        """Test saving and loading config from file."""
        config = DomainBertConfig(
            vocab_size=140,
            hidden_size=512,
            num_attention_heads=16,
            tld_vocab_size=1500,
            mlm_weight=0.88,
            tld_weight=0.12
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save config
            config.save_pretrained(temp_dir)
            
            # Check file exists
            config_file = Path(temp_dir) / "config.json"
            assert config_file.exists()
            
            # Load config
            loaded_config = DomainBertConfig.from_pretrained(temp_dir)
            
            # Check values match
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.hidden_size == config.hidden_size
            assert loaded_config.num_attention_heads == config.num_attention_heads
            assert loaded_config.tld_vocab_size == config.tld_vocab_size
            assert loaded_config.mlm_weight == config.mlm_weight
            assert loaded_config.tld_weight == config.tld_weight
    
    def test_config_json_serialization(self):
        """Test config can be serialized to/from JSON."""
        config = DomainBertConfig(
            vocab_size=133,
            hidden_size=256,
            tld_vocab_size=12351  # Actual TLD vocab size from dataset
        )
        
        # Serialize to JSON string
        json_str = json.dumps(config.to_dict())
        
        # Deserialize
        config_dict = json.loads(json_str)
        loaded_config = DomainBertConfig(**config_dict)
        
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.hidden_size == config.hidden_size
        assert loaded_config.tld_vocab_size == config.tld_vocab_size
    
    def test_config_update(self):
        """Test updating config values."""
        config = DomainBertConfig()
        
        # Update via attribute
        config.hidden_size = 512
        assert config.hidden_size == 512
        
        # Update from dict
        config.update({"vocab_size": 200, "num_hidden_layers": 6})
        assert config.vocab_size == 200
        assert config.num_hidden_layers == 6
    
    def test_config_model_type(self):
        """Test model type is correctly set."""
        config = DomainBertConfig()
        
        assert config.model_type == "domain-bert"
        assert hasattr(config, "architectures")
        assert config.architectures == ["DomainBertModel"]
    
    def test_config_weight_validation(self):
        """Test MLM and TLD weights validation."""
        # Weights should sum to 1.0
        config = DomainBertConfig(mlm_weight=0.7, tld_weight=0.3)
        assert abs((config.mlm_weight + config.tld_weight) - 1.0) < 1e-6
        
        # Test with different weights
        config = DomainBertConfig(mlm_weight=0.95, tld_weight=0.05)
        assert abs((config.mlm_weight + config.tld_weight) - 1.0) < 1e-6
    
    def test_config_backward_compatibility(self):
        """Test config handles missing keys gracefully."""
        # Create config dict without some newer fields
        old_config_dict = {
            "vocab_size": 133,
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "intermediate_size": 1024
            # Missing: tld_vocab_size, mlm_weight, tld_weight
        }
        
        config = DomainBertConfig(**old_config_dict)
        
        # Should use defaults for missing values
        assert config.tld_vocab_size == 1000  # default
        assert config.mlm_weight == 0.85  # default
        assert config.tld_weight == 0.15  # default
    
    def test_config_extra_fields(self):
        """Test config handles extra fields properly."""
        config_dict = {
            "vocab_size": 133,
            "hidden_size": 256,
            "extra_field": "should_be_ignored",
            "another_extra": 123
        }
        
        # Should not raise error
        config = DomainBertConfig(**config_dict)
        
        # Extra fields might be stored but not as main attributes
        assert config.vocab_size == 133
        assert config.hidden_size == 256