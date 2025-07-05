"""
Integration tests for the data pipeline.

Tests the streaming dataset with real files and tokenization.
"""
import pytest
import tempfile
import gzip
from pathlib import Path
import json

from domainbert.data.streaming_dataset import MultiFileStreamingDataset
from domainbert.tokenizer import DomainBertTokenizerFast


class TestDataPipelineIntegration:
    """Test the complete data pipeline from files to tokenized output."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test domain files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create uncompressed file
            with open(temp_path / "domains1.txt", "w") as f:
                f.write("example.com\n")
                f.write("test.org\n")
                f.write("subdomain.example.net\n")
                f.write("another-test.edu\n")
                f.write("site123.gov\n")
            
            # Create gzipped file
            with gzip.open(temp_path / "domains2.txt.gz", "wt") as f:
                f.write("compressed.com\n")
                f.write("gzipped.org\n")
                f.write("archive.net\n")
                f.write("data.io\n")
                f.write("website.co.uk\n")
            
            # Create another uncompressed file
            with open(temp_path / "domains3.txt", "w") as f:
                f.write("final.com\n")
                f.write("last.org\n")
                f.write("end.net\n")
            
            yield temp_path
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        # Create minimal tokenizer directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Tokenizer config
            tokenizer_config = {
                "model_max_length": 64,
                "tokenizer_class": "DomainBertTokenizerFast"
            }
            with open(temp_path / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f)
            
            # Tokenizer vocabulary
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
            
            # TLD vocabulary
            tld_vocab = {
                "com": 0, "org": 1, "net": 2, "edu": 3,
                "gov": 4, "io": 5, "co.uk": 6, "[UNK_TLD]": 7
            }
            with open(temp_path / "tld_vocab.json", "w") as f:
                json.dump(tld_vocab, f)
            
            # Special tokens
            special_tokens = {
                "unk_token": "[UNK]", "sep_token": "[SEP]",
                "pad_token": "[PAD]", "cls_token": "[CLS]",
                "mask_token": "[MASK]"
            }
            with open(temp_path / "special_tokens_map.json", "w") as f:
                json.dump(special_tokens, f)
            
            # Load tokenizer
            tokenizer = DomainBertTokenizerFast.from_pretrained(str(temp_path))
            yield tokenizer
    
    def test_dataset_initialization(self, temp_data_dir, mock_tokenizer):
        """Test dataset can be initialized with files."""
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            max_length=64
        )
        
        assert dataset.data_dir == Path(temp_data_dir)
        assert len(dataset.files) == 3  # 3 files created
        assert dataset.max_length == 64
    
    def test_dataset_file_discovery(self, temp_data_dir, mock_tokenizer):
        """Test dataset finds all domain files."""
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer
        )
        
        # Check it found all files
        file_names = [f.name for f in dataset.files]
        assert "domains1.txt" in file_names
        assert "domains2.txt.gz" in file_names
        assert "domains3.txt" in file_names
    
    def test_dataset_iteration(self, temp_data_dir, mock_tokenizer):
        """Test iterating through dataset."""
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            shuffle=False  # Disable shuffle for predictable order
        )
        
        # Collect all domains
        all_domains = []
        for item in dataset:
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "token_type_ids" in item
            assert "tld_ids" in item
            all_domains.append(item)
        
        # Should have 13 total domains (5 + 5 + 3)
        assert len(all_domains) == 13
    
    def test_dataset_tokenization(self, temp_data_dir, mock_tokenizer):
        """Test that domains are properly tokenized."""
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            shuffle=False
        )
        
        # Get first item
        first_item = next(iter(dataset))
        
        # Check structure
        assert isinstance(first_item["input_ids"], list)
        assert isinstance(first_item["attention_mask"], list)
        assert isinstance(first_item["token_type_ids"], list)
        assert isinstance(first_item["tld_ids"], int)
        
        # Check lengths match
        assert len(first_item["input_ids"]) == len(first_item["attention_mask"])
        assert len(first_item["input_ids"]) == len(first_item["token_type_ids"])
        
        # Check TLD ID is valid
        assert 0 <= first_item["tld_ids"] < 8  # Based on our TLD vocab
    
    def test_dataset_shuffling(self, temp_data_dir, mock_tokenizer):
        """Test dataset shuffling with buffer."""
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            shuffle=True,
            shuffle_buffer_size=5,
            seed=42  # First seed
        )
        
        # Collect domains from two iterations
        first_iteration = []
        for i, item in enumerate(dataset):
            if i >= 10:  # Get first 10
                break
            first_iteration.append(item["input_ids"])
        
        # Reset and iterate again with different seed
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            shuffle=True,
            shuffle_buffer_size=5,
            seed=123  # Different seed
        )
        
        second_iteration = []
        for i, item in enumerate(dataset):
            if i >= 10:
                break
            second_iteration.append(item["input_ids"])
        
        # Orders should be different (with high probability)
        assert first_iteration != second_iteration
    
    def test_dataset_max_length_truncation(self, temp_data_dir, mock_tokenizer):
        """Test that long domains are truncated."""
        # Add a very long domain
        with open(temp_data_dir / "long_domain.txt", "w") as f:
            f.write("a" * 100 + ".com\n")
        
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            max_length=32,  # Short max length
            shuffle=False
        )
        
        # Find the long domain
        for item in dataset:
            if len(item["input_ids"]) > 30:  # Should be truncated
                assert len(item["input_ids"]) <= 32
                break
    
    def test_dataset_specific_files(self, temp_data_dir, mock_tokenizer):
        """Test dataset with specific file list."""
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            files=["domains1.txt", "domains3.txt"],  # Skip domains2.txt.gz
            shuffle=False
        )
        
        # Should only have 8 domains (5 + 3)
        count = sum(1 for _ in dataset)
        assert count == 8
    
    def test_dataset_empty_lines_handling(self, temp_data_dir, mock_tokenizer):
        """Test dataset handles empty lines correctly."""
        # Create file with empty lines
        with open(temp_data_dir / "with_empty.txt", "w") as f:
            f.write("domain1.com\n")
            f.write("\n")  # Empty line
            f.write("domain2.org\n")
            f.write("   \n")  # Whitespace line
            f.write("domain3.net\n")
        
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            files=["with_empty.txt"],
            shuffle=False
        )
        
        # Should only get 3 valid domains
        domains = list(dataset)
        assert len(domains) == 3
    
    def test_dataset_multiprocessing(self, temp_data_dir, mock_tokenizer):
        """Test dataset works with multiple workers."""
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            num_workers=2
        )
        
        # Use with DataLoader (simulated here by just iterating)
        items = []
        for item in dataset:
            items.append(item)
            if len(items) >= 5:  # Just test a few
                break
        
        assert len(items) == 5
        # All items should be properly tokenized
        for item in items:
            assert "input_ids" in item
            assert "tld_ids" in item
    
    def test_dataset_error_handling(self, temp_data_dir, mock_tokenizer):
        """Test dataset handles errors gracefully."""
        # Create a file with invalid content
        with open(temp_data_dir / "invalid.txt", "w") as f:
            f.write("valid.com\n")
            f.write("invalid_no_tld\n")  # No TLD
            f.write("another.org\n")
        
        dataset = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            files=["invalid.txt"],
            shuffle=False
        )
        
        # Should skip invalid domains
        domains = list(dataset)
        # Exact count depends on tokenizer's error handling
        assert len(domains) >= 2  # At least the valid ones
    
    def test_dataset_reproducibility(self, temp_data_dir, mock_tokenizer):
        """Test dataset is reproducible with same seed."""
        # First iteration
        dataset1 = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            shuffle=True,
            seed=123
        )
        
        items1 = []
        for i, item in enumerate(dataset1):
            if i >= 5:
                break
            items1.append(item["input_ids"])
        
        # Second iteration with same seed
        dataset2 = MultiFileStreamingDataset(
            data_dir=str(temp_data_dir),
            tokenizer=mock_tokenizer,
            shuffle=True,
            seed=123
        )
        
        items2 = []
        for i, item in enumerate(dataset2):
            if i >= 5:
                break
            items2.append(item["input_ids"])
        
        # Should produce same order
        assert items1 == items2