"""Test multi-worker data loading with new domain parser"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from domainbert.tokenizer import DomainBertTokenizerFast
from domainbert.data.streaming_dataset import MultiFileStreamingDataset
from domainbert.data.collator import DataCollatorForDomainMLM
import time


def test_multiworker_loading():
    """Test that multi-worker loading works with the new parser"""
    print("Testing multi-worker data loading...")
    
    # Create tokenizer
    tokenizer = DomainBertTokenizerFast()
    
    # Find test data
    data_dir = Path("data/raw/domains_project/data")
    test_file = data_dir / "domains_tiny.txt"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        print("Creating a small test file...")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, 'w') as f:
            for domain in ["example.com", "test.org", "subdomain.example.net", 
                          "another.test.co.uk", "deep.sub.domain.com"]:
                f.write(f"{domain}\n")
    
    # Create dataset
    dataset = MultiFileStreamingDataset(
        file_paths=[str(test_file)],
        tokenizer=tokenizer,
        max_length=64,
        buffer_size=2,
        shuffle_buffer_size=5,
        max_samples=20
    )
    
    # Create data collator
    collator = DataCollatorForDomainMLM(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )
    
    # Test with different worker counts
    for num_workers in [0, 2, 4]:
        print(f"\nTesting with {num_workers} workers...")
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
            num_workers=num_workers
        )
        
        try:
            start_time = time.time()
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count == 1:
                    print(f"  First batch shape: {batch['input_ids'].shape}")
                    print(f"  Keys in batch: {list(batch.keys())}")
            
            elapsed = time.time() - start_time
            print(f"  Loaded {batch_count} batches in {elapsed:.2f}s")
            print(f"  ✅ Success with {num_workers} workers!")
            
        except Exception as e:
            print(f"  ❌ Failed with {num_workers} workers: {e}")
            if "pickle" in str(e).lower():
                print("    (Pickling issue detected)")
            return False
    
    return True


if __name__ == "__main__":
    success = test_multiworker_loading()
    if success:
        print("\n✅ Multi-worker loading test PASSED!")
    else:
        print("\n❌ Multi-worker loading test FAILED!")
    sys.exit(0 if success else 1)