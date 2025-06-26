"""
Unit tests for DomainBertTokenizer.

Tests domain parsing, token type assignment, TLD handling, and tokenization.
"""
import pytest
import tempfile
import json
from pathlib import Path

from domainbert.tokenizer import DomainBertTokenizerFast, DomainPreTokenizer


class TestDomainPreTokenizer:
    """Test the domain pre-tokenizer component."""
    
    def test_parse_simple_domain(self):
        """Test parsing a simple domain without subdomain."""
        pretokenizer = DomainPreTokenizer()
        
        # Test simple domain
        result = pretokenizer.pre_tokenize_str("example.com")
        assert len(result) == 11  # e-x-a-m-p-l-e-.-c-o-m
        
        # Check token assignments
        tokens, token_types = zip(*result)
        assert "".join(tokens) == "example.com"
        
        # First 7 chars should be domain (type 0)
        assert all(tt == 0 for tt in token_types[:7])
        # Dot should be separator (type 3)
        assert token_types[7] == 3
        # Last 3 chars should be TLD (type 2)
        assert all(tt == 2 for tt in token_types[8:])
    
    def test_parse_subdomain(self):
        """Test parsing domain with subdomain."""
        pretokenizer = DomainPreTokenizer()
        
        result = pretokenizer.pre_tokenize_str("sub.example.com")
        tokens, token_types = zip(*result)
        
        assert "".join(tokens) == "sub.example.com"
        
        # Check token type assignments
        assert token_types[:3] == (1, 1, 1)  # sub (subdomain)
        assert token_types[3] == 3  # . (separator)
        assert token_types[4:11] == (0, 0, 0, 0, 0, 0, 0)  # example (domain)
        assert token_types[11] == 3  # . (separator)
        assert token_types[12:] == (2, 2, 2)  # com (TLD)
    
    def test_parse_multiple_subdomains(self):
        """Test parsing domain with multiple subdomain levels."""
        pretokenizer = DomainPreTokenizer()
        
        result = pretokenizer.pre_tokenize_str("a.b.example.com")
        tokens, token_types = zip(*result)
        
        assert "".join(tokens) == "a.b.example.com"
        
        # a.b should all be subdomain
        assert token_types[0] == 1  # a
        assert token_types[1] == 3  # .
        assert token_types[2] == 1  # b
        assert token_types[3] == 3  # .
        # example should be domain
        assert all(tt == 0 for tt in token_types[4:11])
        # com should be TLD
        assert all(tt == 2 for tt in token_types[12:])
    
    def test_parse_long_tld(self):
        """Test parsing domain with longer TLD."""
        pretokenizer = DomainPreTokenizer()
        
        result = pretokenizer.pre_tokenize_str("example.co.uk")
        tokens, token_types = zip(*result)
        
        # example should be domain
        assert all(tt == 0 for tt in token_types[:7])
        # co.uk should be TLD
        assert token_types[8:] == (2, 2, 3, 2, 2)  # c-o-.-u-k
    
    def test_parse_numeric_domain(self):
        """Test parsing domain with numbers."""
        pretokenizer = DomainPreTokenizer()
        
        result = pretokenizer.pre_tokenize_str("123.456.com")
        tokens, token_types = zip(*result)
        
        assert "".join(tokens) == "123.456.com"
        assert token_types[:3] == (1, 1, 1)  # 123 (subdomain)
        assert token_types[4:7] == (0, 0, 0)  # 456 (domain)
        assert token_types[8:] == (2, 2, 2)  # com (TLD)
    
    def test_parse_hyphenated_domain(self):
        """Test parsing domain with hyphens."""
        pretokenizer = DomainPreTokenizer()
        
        result = pretokenizer.pre_tokenize_str("my-site.com")
        tokens, token_types = zip(*result)
        
        assert "".join(tokens) == "my-site.com"
        # All of my-site should be domain
        assert all(tt == 0 for tt in token_types[:7])
        assert all(tt == 2 for tt in token_types[8:])


class TestDomainBertTokenizerFast:
    """Test the full DomainBertTokenizerFast."""
    
    @pytest.fixture
    def temp_tokenizer_dir(self):
        """Create a temporary directory with tokenizer files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create minimal tokenizer config
            tokenizer_config = {
                "model_max_length": 128,
                "tokenizer_class": "DomainBertTokenizerFast"
            }
            with open(temp_path / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f)
            
            # Create minimal tokenizer.json
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": []
                },
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
                        "[PAD]": 0,
                        "[UNK]": 1,
                        "[CLS]": 2,
                        "[SEP]": 3,
                        "[MASK]": 4
                    },
                    "merges": []
                }
            }
            
            # Add ASCII characters to vocab
            for i in range(128):
                tokenizer_json["model"]["vocab"][chr(i)] = i + 5
            
            with open(temp_path / "tokenizer.json", "w") as f:
                json.dump(tokenizer_json, f)
            
            # Create TLD vocabulary
            tld_vocab = {
                "com": 0,
                "net": 1,
                "org": 2,
                "co.uk": 3,
                "edu": 4,
                "gov": 5,
                "[UNK_TLD]": 6
            }
            with open(temp_path / "tld_vocab.json", "w") as f:
                json.dump(tld_vocab, f)
            
            # Create special tokens map
            special_tokens = {
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]"
            }
            with open(temp_path / "special_tokens_map.json", "w") as f:
                json.dump(special_tokens, f)
            
            yield temp_path
    
    def test_tokenizer_initialization(self, temp_tokenizer_dir):
        """Test tokenizer can be initialized and loaded."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        assert tokenizer is not None
        assert tokenizer.model_max_length == 128
        assert tokenizer.tld_to_id is not None
        assert len(tokenizer.tld_to_id) == 7  # 6 TLDs + [UNK_TLD]
    
    def test_tokenize_simple_domain(self, temp_tokenizer_dir):
        """Test tokenizing a simple domain."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        # Tokenize single domain
        encoded = tokenizer("example.com", return_tensors="pt")
        
        assert "input_ids" in encoded
        assert "token_type_ids" in encoded
        assert "attention_mask" in encoded
        assert "tld_ids" in encoded
        
        # Check TLD ID
        assert encoded["tld_ids"][0].item() == 0  # com -> 0
    
    def test_tokenize_batch(self, temp_tokenizer_dir):
        """Test tokenizing multiple domains."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        domains = ["example.com", "test.org", "sub.domain.net"]
        encoded = tokenizer(domains, padding=True, return_tensors="pt")
        
        assert encoded["input_ids"].shape[0] == 3
        assert encoded["tld_ids"].tolist() == [0, 2, 1]  # com, org, net
        
        # Check padding is applied
        assert encoded["attention_mask"].sum(dim=1).tolist() == [13, 11, 17]  # [CLS] + domain + [SEP]
    
    def test_tokenize_unknown_tld(self, temp_tokenizer_dir):
        """Test tokenizing domain with unknown TLD."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        encoded = tokenizer("example.xyz", return_tensors="pt")
        
        # Should use [UNK_TLD] token
        assert encoded["tld_ids"][0].item() == 6  # [UNK_TLD] -> 6
    
    def test_tokenize_with_special_chars(self, temp_tokenizer_dir):
        """Test tokenizing domain with special characters."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        # Domain with hyphen and numbers
        encoded = tokenizer("test-123.com", return_tensors="pt")
        
        assert encoded["input_ids"] is not None
        assert encoded["tld_ids"][0].item() == 0  # com
    
    def test_token_type_ids_structure(self, temp_tokenizer_dir):
        """Test that token type IDs correctly identify domain parts."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        encoded = tokenizer("sub.example.com", return_tensors="pt")
        token_types = encoded["token_type_ids"][0].tolist()
        
        # Skip [CLS] token, check structure
        # Note: exact indices depend on tokenization, but pattern should be:
        # subdomain tokens -> separator -> domain tokens -> separator -> TLD tokens
        assert 1 in token_types  # subdomain type
        assert 0 in token_types  # domain type
        assert 2 in token_types  # TLD type
        assert 3 in token_types  # separator type
    
    def test_max_length_truncation(self, temp_tokenizer_dir):
        """Test that long domains are properly truncated."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        # Create a very long domain
        long_domain = "a" * 200 + ".com"
        encoded = tokenizer(long_domain, truncation=True, return_tensors="pt")
        
        assert encoded["input_ids"].shape[1] <= 128
        assert encoded["tld_ids"][0].item() == 0  # Should still identify .com
    
    def test_save_and_load_tokenizer(self, temp_tokenizer_dir):
        """Test saving and loading tokenizer preserves TLD vocab."""
        tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_tokenizer_dir))
        
        # Save to new location
        with tempfile.TemporaryDirectory() as save_dir:
            tokenizer.save_pretrained(save_dir)
            
            # Load and verify
            loaded_tokenizer = DomainBertTokenizerFast.from_pretrained(save_dir)
            
            assert loaded_tokenizer.tld_to_id == tokenizer.tld_to_id
            assert loaded_tokenizer.model_max_length == tokenizer.model_max_length
            
            # Test functionality is preserved
            test_domain = "example.org"
            original_encoded = tokenizer(test_domain, return_tensors="pt")
            loaded_encoded = loaded_tokenizer(test_domain, return_tensors="pt")
            
            assert original_encoded["tld_ids"].tolist() == loaded_encoded["tld_ids"].tolist()