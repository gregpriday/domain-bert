"""Configuration for DomainBERT model"""

from transformers import PretrainedConfig


class DomainBertConfig(PretrainedConfig):
    """Configuration for DomainBERT model
    
    Args:
        vocab_size: Size of character vocabulary + special tokens (default: 133)
        char_vocab_size: Size of character vocabulary (default: 128)
        tld_vocab_size: Size of TLD vocabulary (default: 1000)
        hidden_size: Hidden layer dimensions (default: 256)
        num_hidden_layers: Number of transformer layers (default: 12)
        num_attention_heads: Number of attention heads (default: 8)
        intermediate_size: FFN intermediate size (default: 1024)
        hidden_dropout_prob: Dropout probability (default: 0.1)
        attention_probs_dropout_prob: Attention dropout (default: 0.1)
        max_position_embeddings: Maximum sequence length (default: 128)
        type_vocab_size: Token type vocabulary size (default: 4)
        initializer_range: Weight initialization range (default: 0.02)
        layer_norm_eps: LayerNorm epsilon (default: 1e-12)
        mlm_weight: Weight for MLM loss (default: 0.85)
        tld_weight: Weight for TLD prediction loss (default: 0.15)
        tld_embed_dim: TLD embedding dimension (default: 64)
        use_tld_embeddings: Whether to use TLD embeddings (default: True)
        hidden_act: Hidden layer activation function (default: "gelu")
    """
    
    model_type = "domain-bert"
    
    def __init__(
        self,
        vocab_size=43,  # 5 special + 26 letters + 10 digits + hyphen + period
        char_vocab_size=38,  # 26 letters + 10 digits + hyphen + period
        tld_vocab_size=513,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,
        type_vocab_size=4,  # main domain, subdomain, tld, separator
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        mask_token_id=1,
        cls_token_id=2,
        sep_token_id=3,
        unk_token_id=4,
        # Multi-task weights
        mlm_weight=0.85,
        tld_weight=0.15,
        # TLD integration
        tld_embed_dim=64,
        use_tld_embeddings=True,
        hidden_act="gelu",
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        # Validation
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size
        self.tld_vocab_size = tld_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.unk_token_id = unk_token_id
        self.mlm_weight = mlm_weight
        self.tld_weight = tld_weight
        self.tld_embed_dim = tld_embed_dim
        self.use_tld_embeddings = use_tld_embeddings
        self.hidden_act = hidden_act
        
        # Set architectures
        self.architectures = ["DomainBertModel"]
    
    def update(self, config_dict):
        """Update configuration with values from dictionary."""
        for key, value in config_dict.items():
            setattr(self, key, value)