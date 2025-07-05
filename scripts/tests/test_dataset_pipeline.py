#!/usr/bin/env python3
"""
Comprehensive test script for DomainBERT dataset pipeline.
Tests all components without running actual training.
"""

import os
import sys
import time
import json
import psutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domainbert.tokenizer import DomainBertTokenizerFast
from domainbert.data.streaming_dataset import MultiFileStreamingDataset
from domainbert.data.collator import DataCollatorForDomainMLM


class DatasetTester:
    """Comprehensive tester for DomainBERT dataset pipeline."""
    
    def __init__(self, data_dir: str, tokenizer_path: str, tld_vocab_path: str):
        self.data_dir = Path(data_dir)
        self.tokenizer_path = Path(tokenizer_path)
        self.tld_vocab_path = Path(tld_vocab_path)
        self.results = {}
        
    def run_all_tests(self):
        """Run all dataset tests."""
        print("=" * 80)
        print("DomainBERT Dataset Pipeline Test Suite")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Start time: {datetime.now()}")
        print("=" * 80)
        
        # Test 1: File discovery
        print("\n1. Testing file discovery...")
        self.test_file_discovery()
        
        # Test 2: Tokenizer loading
        print("\n2. Testing tokenizer...")
        tokenizer = self.test_tokenizer_loading()
        
        # Test 3: Dataset initialization
        print("\n3. Testing dataset initialization...")
        self.test_dataset_initialization(tokenizer)
        
        # Test 4: Data loading performance
        print("\n4. Testing data loading performance...")
        self.test_data_loading_performance(tokenizer)
        
        # Test 5: Tokenization pipeline
        print("\n5. Testing tokenization pipeline...")
        self.test_tokenization_pipeline(tokenizer)
        
        # Test 6: Multi-worker loading
        print("\n6. Testing multi-worker loading...")
        self.test_multiworker_loading(tokenizer)
        
        # Test 7: Memory usage
        print("\n7. Testing memory usage...")
        self.test_memory_usage(tokenizer)
        
        # Test 8: Data quality
        print("\n8. Testing data quality...")
        self.test_data_quality(tokenizer)
        
        # Test 9: Error handling
        print("\n9. Testing error handling...")
        self.test_error_handling(tokenizer)
        
        # Test 10: End-to-end mini batch
        print("\n10. Testing end-to-end pipeline...")
        self.test_end_to_end_pipeline(tokenizer)
        
        # Generate report
        print("\n" + "=" * 80)
        self.generate_report()
        
    def test_file_discovery(self):
        """Test discovering all domain files."""
        try:
            # Find all .xz files
            xz_files = list(self.data_dir.glob("**/*.txt.xz"))
            total_size = sum(f.stat().st_size for f in xz_files)
            
            # Group by TLD category
            tld_groups = defaultdict(list)
            for f in xz_files:
                parent = f.parent.name
                tld_groups[parent].append(f)
            
            self.results['file_discovery'] = {
                'total_files': len(xz_files),
                'total_size_gb': total_size / (1024**3),
                'tld_categories': len(tld_groups),
                'largest_category': max(tld_groups.items(), key=lambda x: len(x[1]))[0],
                'files_by_category': {k: len(v) for k, v in sorted(tld_groups.items())}
            }
            
            print(f"✓ Found {len(xz_files)} files ({total_size / (1024**3):.2f} GB)")
            print(f"✓ Files spread across {len(tld_groups)} TLD categories")
            print(f"✓ Largest category: {self.results['file_discovery']['largest_category']}")
            
        except Exception as e:
            print(f"✗ File discovery failed: {e}")
            self.results['file_discovery'] = {'error': str(e)}
            
    def test_tokenizer_loading(self):
        """Test loading the tokenizer."""
        try:
            # Load tokenizer
            tokenizer = DomainBertTokenizerFast(
                tld_vocab_file=str(self.tld_vocab_path),
                max_len=64
            )
            
            # Test basic tokenization
            test_domains = ["example.com", "subdomain.test.org", "длинное-имя.рф"]
            encoded = tokenizer(test_domains, padding=True, return_tensors=None)
            
            # Load TLD vocabulary for analysis
            with open(self.tld_vocab_path, 'r') as f:
                tld_vocab = json.load(f)
            
            self.results['tokenizer'] = {
                'loaded': True,
                'tld_vocab_size': len(tld_vocab),
                'max_length': tokenizer.model_max_length,
                'test_tokenization': 'success'
            }
            
            print(f"✓ Tokenizer loaded successfully")
            print(f"✓ TLD vocabulary size: {len(tld_vocab)}")
            print(f"✓ Max sequence length: {tokenizer.model_max_length}")
            
            return tokenizer
            
        except Exception as e:
            print(f"✗ Tokenizer loading failed: {e}")
            self.results['tokenizer'] = {'error': str(e)}
            raise
            
    def test_dataset_initialization(self, tokenizer):
        """Test dataset initialization with full data."""
        try:
            # Initialize dataset
            dataset = MultiFileStreamingDataset(
                data_dir=str(self.data_dir),
                tokenizer=tokenizer,
                max_length=64,
                shuffle=False
            )
            
            self.results['dataset_init'] = {
                'initialized': True,
                'num_files': len(dataset.file_paths),
                'estimated_size': dataset.estimated_size,
                'first_5_files': [Path(f).name for f in dataset.file_paths[:5]]
            }
            
            print(f"✓ Dataset initialized with {len(dataset.file_paths)} files")
            print(f"✓ Estimated dataset size: {dataset.estimated_size:,} domains")
            
        except Exception as e:
            print(f"✗ Dataset initialization failed: {e}")
            self.results['dataset_init'] = {'error': str(e)}
            
    def test_data_loading_performance(self, tokenizer):
        """Test data loading performance."""
        try:
            print("  Testing single file loading speed...")
            
            # Test with a single file first
            sample_files = list(self.data_dir.glob("**/domain2multi-*.txt.xz"))[:1]
            if not sample_files:
                raise ValueError("No sample files found")
                
            dataset = MultiFileStreamingDataset(
                file_paths=[str(sample_files[0])],
                tokenizer=tokenizer,
                max_length=64,
                shuffle=False,
                buffer_size=1000
            )
            
            # Time loading 1000 domains
            start_time = time.time()
            domains_loaded = 0
            
            for i, item in enumerate(dataset):
                domains_loaded += 1
                if domains_loaded >= 1000:
                    break
                    
            elapsed = time.time() - start_time
            domains_per_second = domains_loaded / elapsed
            
            self.results['performance'] = {
                'domains_loaded': domains_loaded,
                'elapsed_seconds': elapsed,
                'domains_per_second': domains_per_second,
                'estimated_hours_full': (1_766_025_618 / domains_per_second) / 3600
            }
            
            print(f"✓ Loaded {domains_loaded} domains in {elapsed:.2f} seconds")
            print(f"✓ Performance: {domains_per_second:.0f} domains/second")
            print(f"✓ Estimated time for full dataset: {self.results['performance']['estimated_hours_full']:.1f} hours")
            
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
            self.results['performance'] = {'error': str(e)}
            
    def test_tokenization_pipeline(self, tokenizer):
        """Test tokenization with real domains."""
        try:
            print("  Sampling domains from different TLDs...")
            
            # Get sample of different file types
            sample_files = []
            for pattern in ["**/domain2multi-com*.xz", "**/domain2multi-net*.xz", 
                           "**/domain2multi-org*.xz", "**/domain2multi-de*.xz"]:
                files = list(self.data_dir.glob(pattern))
                if files:
                    sample_files.append(str(files[0]))
                    
            if not sample_files:
                raise ValueError("No sample files found")
                
            dataset = MultiFileStreamingDataset(
                file_paths=sample_files[:2],  # Use just 2 files
                tokenizer=tokenizer,
                max_length=64,
                shuffle=False,
                max_samples=100
            )
            
            # Collect tokenization examples
            examples = []
            tld_distribution = Counter()
            
            for item in dataset:
                examples.append({
                    'input_ids_len': len([x for x in item['input_ids'] if x != tokenizer.pad_token_id]),
                    'tld_id': item['tld_ids'],
                    'has_subdomain': 1 in item['token_type_ids']
                })
                tld_distribution[item['tld_ids']] += 1
                
            self.results['tokenization'] = {
                'samples_tested': len(examples),
                'avg_length': sum(e['input_ids_len'] for e in examples) / len(examples),
                'with_subdomain': sum(e['has_subdomain'] for e in examples),
                'unique_tlds': len(tld_distribution),
                'tld_distribution': dict(tld_distribution.most_common(10))
            }
            
            print(f"✓ Tested {len(examples)} tokenizations")
            print(f"✓ Average token length: {self.results['tokenization']['avg_length']:.1f}")
            print(f"✓ Domains with subdomains: {self.results['tokenization']['with_subdomain']}")
            
        except Exception as e:
            print(f"✗ Tokenization test failed: {e}")
            self.results['tokenization'] = {'error': str(e)}
            traceback.print_exc()
            
    def test_multiworker_loading(self, tokenizer):
        """Test multi-worker data loading."""
        try:
            print("  Testing with 4 workers...")
            
            # Use small subset for testing
            sample_files = list(self.data_dir.glob("**/domain2multi-*.txt.xz"))[:4]
            
            dataset = MultiFileStreamingDataset(
                file_paths=[str(f) for f in sample_files],
                tokenizer=tokenizer,
                max_length=64,
                num_workers=4,
                max_samples=400
            )
            
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )
            
            # Test loading
            start_time = time.time()
            batches_loaded = 0
            
            for batch in dataloader:
                batches_loaded += 1
                if batches_loaded >= 10:
                    break
                    
            elapsed = time.time() - start_time
            
            self.results['multiworker'] = {
                'workers': 4,
                'batches_loaded': batches_loaded,
                'elapsed_seconds': elapsed,
                'batch_shape': {k: list(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}
            }
            
            print(f"✓ Loaded {batches_loaded} batches with 4 workers in {elapsed:.2f}s")
            print(f"✓ Batch shapes: {self.results['multiworker']['batch_shape']}")
            
        except Exception as e:
            print(f"✗ Multi-worker test failed: {e}")
            self.results['multiworker'] = {'error': str(e)}
            
    def test_memory_usage(self, tokenizer):
        """Test memory usage during loading."""
        try:
            print("  Monitoring memory usage...")
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Load with large buffer
            sample_files = list(self.data_dir.glob("**/domain2multi-*.txt.xz"))[:2]
            
            dataset = MultiFileStreamingDataset(
                file_paths=[str(f) for f in sample_files],
                tokenizer=tokenizer,
                buffer_size=10000,
                shuffle_buffer_size=50000,
                max_samples=50000
            )
            
            # Process some data
            peak_memory = initial_memory
            for i, item in enumerate(dataset):
                if i % 1000 == 0:
                    current_memory = process.memory_info().rss / (1024**2)
                    peak_memory = max(peak_memory, current_memory)
                if i >= 10000:
                    break
                    
            final_memory = process.memory_info().rss / (1024**2)
            
            self.results['memory'] = {
                'initial_mb': initial_memory,
                'peak_mb': peak_memory,
                'final_mb': final_memory,
                'increase_mb': peak_memory - initial_memory
            }
            
            print(f"✓ Initial memory: {initial_memory:.1f} MB")
            print(f"✓ Peak memory: {peak_memory:.1f} MB")
            print(f"✓ Memory increase: {peak_memory - initial_memory:.1f} MB")
            
        except Exception as e:
            print(f"✗ Memory test failed: {e}")
            self.results['memory'] = {'error': str(e)}
            
    def test_data_quality(self, tokenizer):
        """Test data quality and validity."""
        try:
            print("  Analyzing domain quality...")
            
            # Sample from different TLDs
            sample_files = []
            for tld in ['com', 'net', 'org', 'de', 'uk']:
                files = list(self.data_dir.glob(f"**/domain2multi-{tld}*.xz"))
                if files:
                    sample_files.append(str(files[0]))
                    
            dataset = MultiFileStreamingDataset(
                file_paths=sample_files[:3],
                tokenizer=tokenizer,
                shuffle=False,
                max_samples=1000
            )
            
            # Analyze domains
            domain_lengths = []
            invalid_count = 0
            tld_counts = Counter()
            
            # Also test raw domain reading
            from domainbert.data.streaming_dataset import MultiFileStreamingDataset
            test_dataset = MultiFileStreamingDataset(
                file_paths=sample_files[:1],
                tokenizer=tokenizer,
                max_samples=100
            )
            
            # Get the unknown TLD ID - check both possible keys
            unk_tld_id = tokenizer.tld_to_id.get('<UNK>', tokenizer.tld_to_id.get('[UNK_TLD]', 1))
            
            for i, item in enumerate(test_dataset):
                if item['tld_ids'] == unk_tld_id:
                    invalid_count += 1
                domain_lengths.append(len([x for x in item['input_ids'] if x != tokenizer.pad_token_id]))
                
            avg_length = sum(domain_lengths) / len(domain_lengths) if domain_lengths else 0
            
            self.results['quality'] = {
                'samples_analyzed': len(domain_lengths),
                'invalid_domains': invalid_count,
                'avg_domain_length': avg_length,
                'min_length': min(domain_lengths) if domain_lengths else 0,
                'max_length': max(domain_lengths) if domain_lengths else 0
            }
            
            print(f"✓ Analyzed {len(domain_lengths)} domains")
            print(f"✓ Invalid domains: {invalid_count}")
            print(f"✓ Average length: {avg_length:.1f} characters")
            
        except Exception as e:
            print(f"✗ Quality test failed: {e}")
            self.results['quality'] = {'error': str(e)}
            traceback.print_exc()
            
    def test_error_handling(self, tokenizer):
        """Test error handling and recovery."""
        try:
            print("  Testing error handling...")
            
            # Test with non-existent file
            try:
                dataset = MultiFileStreamingDataset(
                    file_paths=["/nonexistent/file.xz"],
                    tokenizer=tokenizer
                )
                list(dataset)  # Try to iterate
                error_handled = False
            except:
                error_handled = True
                
            # Test with mix of valid and invalid files
            valid_file = list(self.data_dir.glob("**/domain2multi-*.txt.xz"))[0]
            dataset = MultiFileStreamingDataset(
                file_paths=[str(valid_file), "/nonexistent/file.xz"],
                tokenizer=tokenizer,
                max_samples=10
            )
            
            # Should still work with valid file
            items_loaded = sum(1 for _ in dataset)
            
            self.results['error_handling'] = {
                'nonexistent_file_handled': error_handled,
                'partial_loading_works': items_loaded > 0,
                'items_from_partial': items_loaded
            }
            
            print(f"✓ Non-existent file error handled: {error_handled}")
            print(f"✓ Partial loading works: {items_loaded > 0}")
            
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            self.results['error_handling'] = {'error': str(e)}
            
    def test_end_to_end_pipeline(self, tokenizer):
        """Test end-to-end pipeline with collator."""
        try:
            print("  Testing full pipeline with data collator...")
            
            # Create dataset
            sample_files = list(self.data_dir.glob("**/domain2multi-*.txt.xz"))[:2]
            dataset = MultiFileStreamingDataset(
                file_paths=[str(f) for f in sample_files],
                tokenizer=tokenizer,
                max_samples=128
            )
            
            # Create collator
            collator = DataCollatorForDomainMLM(
                tokenizer=tokenizer,
                mlm_probability=0.15,
                tld_mask_probability=0.1
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=16,
                collate_fn=collator,
                num_workers=0  # Single process for testing
            )
            
            # Test batch creation
            batch = next(iter(dataloader))
            
            self.results['end_to_end'] = {
                'batch_keys': list(batch.keys()),
                'batch_shapes': {k: list(v.shape) for k, v in batch.items() if hasattr(v, 'shape')},
                'has_mlm_labels': 'labels' in batch,
                'has_tld_labels': 'tld_labels' in batch,
                'mlm_mask_count': (batch.get('labels', torch.zeros(1)) != -100).sum().item()
            }
            
            print(f"✓ Created batch with keys: {list(batch.keys())}")
            print(f"✓ MLM masks applied: {self.results['end_to_end']['mlm_mask_count']}")
            print(f"✓ Batch shapes: {self.results['end_to_end']['batch_shapes']}")
            
        except Exception as e:
            print(f"✗ End-to-end test failed: {e}")
            self.results['end_to_end'] = {'error': str(e)}
            traceback.print_exc()
            
    def generate_report(self):
        """Generate final test report."""
        print("\nTest Results Summary")
        print("=" * 80)
        
        # Save detailed results
        report_path = Path("dataset_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Detailed results saved to: {report_path}")
        
        # Print summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if 'error' not in r)
        
        print(f"\nTests passed: {passed_tests}/{total_tests}")
        
        if 'file_discovery' in self.results and 'error' not in self.results['file_discovery']:
            print(f"\nDataset Statistics:")
            print(f"  - Total files: {self.results['file_discovery']['total_files']}")
            print(f"  - Total size: {self.results['file_discovery']['total_size_gb']:.2f} GB")
            print(f"  - TLD categories: {self.results['file_discovery']['tld_categories']}")
            
        if 'performance' in self.results and 'error' not in self.results['performance']:
            print(f"\nPerformance Estimates:")
            print(f"  - Loading speed: {self.results['performance']['domains_per_second']:.0f} domains/sec")
            print(f"  - Full dataset time: {self.results['performance']['estimated_hours_full']:.1f} hours")
            
        if 'memory' in self.results and 'error' not in self.results['memory']:
            print(f"\nMemory Usage:")
            print(f"  - Peak memory: {self.results['memory']['peak_mb']:.1f} MB")
            print(f"  - Memory increase: {self.results['memory']['increase_mb']:.1f} MB")
            
        # List any errors
        errors = [(k, v['error']) for k, v in self.results.items() if 'error' in v]
        if errors:
            print(f"\nErrors encountered:")
            for test_name, error in errors:
                print(f"  - {test_name}: {error}")
                
        print("\n" + "=" * 80)
        print("Testing complete!")


def main():
    """Run the dataset testing suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DomainBERT dataset pipeline")
    parser.add_argument("--data-dir", type=str, default="data/raw/domains_project/data",
                        help="Path to domains data directory")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer",
                        help="Path to tokenizer directory")
    parser.add_argument("--tld-vocab", type=str, default="tokenizer/tld_vocab.json",
                        help="Path to TLD vocabulary file")
    
    args = parser.parse_args()
    
    # Run tests
    tester = DatasetTester(args.data_dir, args.tokenizer_path, args.tld_vocab)
    tester.run_all_tests()


if __name__ == "__main__":
    main()