#!/usr/bin/env python3
"""
Final integration test to verify training readiness.
Simulates a complete training setup without actually training.
"""

import os
import sys
import json
import time
import torch
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import TrainingArguments
from torch.utils.data import DataLoader

from domainbert.config import DomainBertConfig
from domainbert.model import DomainBertForMaskedLM
from domainbert.tokenizer import DomainBertTokenizerFast
from domainbert.data.collator import DataCollatorForDomainMLM
from domainbert.data.streaming_dataset import MultiFileStreamingDataset


class TrainingReadinessTester:
    """Test all components needed for training."""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def run_all_tests(self):
        """Run comprehensive training readiness tests."""
        print("="*80)
        print("DomainBERT Training Readiness Test")
        print("="*80)
        print(f"Start time: {self.start_time}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*80)
        
        # Test 1: Configuration
        print("\n1. Testing Configuration...")
        config = self.test_configuration()
        
        # Test 2: Tokenizer
        print("\n2. Testing Tokenizer...")
        tokenizer = self.test_tokenizer()
        
        # Test 3: Model initialization
        print("\n3. Testing Model Initialization...")
        model = self.test_model_init(config)
        
        # Test 4: Dataset
        print("\n4. Testing Dataset...")
        dataset = self.test_dataset(tokenizer)
        
        # Test 5: Data collator
        print("\n5. Testing Data Collator...")
        collator = self.test_collator(tokenizer)
        
        # Test 6: DataLoader
        print("\n6. Testing DataLoader...")
        dataloader = self.test_dataloader(dataset, collator)
        
        # Test 7: Forward pass
        print("\n7. Testing Forward Pass...")
        self.test_forward_pass(model, dataloader)
        
        # Test 8: Memory and performance
        print("\n8. Testing Memory and Performance...")
        self.test_performance(model, dataloader)
        
        # Test 9: Checkpointing
        print("\n9. Testing Checkpointing...")
        self.test_checkpointing(model, tokenizer)
        
        # Test 10: Multi-GPU readiness
        print("\n10. Testing Multi-GPU Setup...")
        self.test_multi_gpu()
        
        # Generate final report
        self.generate_report()
        
    def test_configuration(self) -> DomainBertConfig:
        """Test model configuration."""
        try:
            # Load from file
            config_path = "configs/model/domain_bert_base_config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            config = DomainBertConfig(**config_dict)
            
            self.results['config'] = {
                'status': 'passed',
                'hidden_size': config.hidden_size,
                'num_layers': config.num_hidden_layers,
                'num_attention_heads': config.num_attention_heads,
                'vocab_size': config.vocab_size,
                'tld_vocab_size': config.tld_vocab_size,
                'max_position_embeddings': config.max_position_embeddings
            }
            
            print("✓ Configuration loaded successfully")
            print(f"  Model size: {config.hidden_size}d, {config.num_hidden_layers} layers")
            
            return config
            
        except Exception as e:
            self.results['config'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Configuration test failed: {e}")
            raise
            
    def test_tokenizer(self) -> DomainBertTokenizerFast:
        """Test tokenizer functionality."""
        try:
            # Look for TLD vocab in different locations
            tld_vocab_paths = [
                Path("src/domainbert/data/tld_vocab.json"),
                Path("data/processed/domains/tld_vocab.json"),
                Path("tokenizer/tld_vocab.json"),
            ]
            
            tld_vocab_file = None
            for path in tld_vocab_paths:
                if path.exists():
                    tld_vocab_file = str(path)
                    break
            
            tokenizer = DomainBertTokenizerFast(
                tld_vocab_file=tld_vocab_file,
                max_len=64
            )
            
            # Test tokenization
            test_domains = [
                "example.com",
                "subdomain.test.org",
                "very-long-domain-name-test.co.uk",
                "short.io"
            ]
            
            encoded = tokenizer(test_domains, padding=True, return_tensors="pt")
            
            self.results['tokenizer'] = {
                'status': 'passed',
                'tld_vocab_size': len(tokenizer.tld_to_id),
                'max_length': tokenizer.model_max_length,
                'test_batch_shape': {k: list(v.shape) for k, v in encoded.items() if hasattr(v, 'shape')}
            }
            
            print("✓ Tokenizer loaded and tested successfully")
            print(f"  TLD vocabulary size: {len(tokenizer.tld_to_id)}")
            
            return tokenizer
            
        except Exception as e:
            self.results['tokenizer'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Tokenizer test failed: {e}")
            raise
            
    def test_model_init(self, config: DomainBertConfig) -> DomainBertForMaskedLM:
        """Test model initialization."""
        try:
            model = DomainBertForMaskedLM(config)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.results['model'] = {
                'status': 'passed',
                'total_params': total_params,
                'trainable_params': trainable_params,
                'total_size_mb': total_params * 4 / (1024**2),  # Assuming float32
                'modules': list(model.children())[0].__class__.__name__
            }
            
            print("✓ Model initialized successfully")
            print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
            print(f"  Model size: {total_params * 4 / (1024**2):.1f} MB")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                print("  Model moved to GPU")
            
            return model
            
        except Exception as e:
            self.results['model'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Model initialization failed: {e}")
            raise
            
    def test_dataset(self, tokenizer) -> MultiFileStreamingDataset:
        """Test dataset loading."""
        try:
            # Load manifest to get some files
            manifest_path = "data/processed/domains/dataset_manifest.json"
            if Path(manifest_path).exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                # Use first 5 files for testing
                test_files = [f['path'] for f in manifest['files'][:5]]
            else:
                # Fallback to finding files directly
                data_dir = Path("data/raw/domains_project/data")
                test_files = list(data_dir.glob("**/*.txt.xz"))[:5]
                test_files = [str(f) for f in test_files]
            
            dataset = MultiFileStreamingDataset(
                file_paths=test_files,
                tokenizer=tokenizer,
                max_length=64,
                max_samples=1000  # Limit for testing
            )
            
            # Test iteration
            sample_count = 0
            for item in dataset:
                sample_count += 1
                if sample_count >= 10:
                    break
            
            self.results['dataset'] = {
                'status': 'passed',
                'test_files': len(test_files),
                'samples_tested': sample_count,
                'estimated_total_size': dataset.estimated_size
            }
            
            print("✓ Dataset created and tested successfully")
            print(f"  Test files: {len(test_files)}")
            print(f"  Samples accessible: Yes")
            
            return dataset
            
        except Exception as e:
            self.results['dataset'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Dataset test failed: {e}")
            raise
            
    def test_collator(self, tokenizer) -> DataCollatorForDomainMLM:
        """Test data collator."""
        try:
            collator = DataCollatorForDomainMLM(
                tokenizer=tokenizer,
                mlm_probability=0.15,
                tld_mask_probability=0.1
            )
            
            # Test with sample data
            sample_batch = [
                {
                    'input_ids': [2] + list(range(5, 20)) + [3] + [0]*43,
                    'attention_mask': [1]*22 + [0]*42,
                    'token_type_ids': [0]*64,
                    'tld_ids': 5
                }
                for _ in range(4)
            ]
            
            collated = collator(sample_batch)
            
            self.results['collator'] = {
                'status': 'passed',
                'output_keys': list(collated.keys()),
                'mlm_probability': 0.15,
                'tld_mask_probability': 0.1
            }
            
            print("✓ Data collator tested successfully")
            print(f"  Output keys: {list(collated.keys())}")
            
            return collator
            
        except Exception as e:
            self.results['collator'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Collator test failed: {e}")
            raise
            
    def test_dataloader(self, dataset, collator) -> DataLoader:
        """Test DataLoader creation."""
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=16,
                collate_fn=collator,
                num_workers=0  # Use 0 for testing to avoid multiprocessing issues
            )
            
            # Test loading a batch
            batch = next(iter(dataloader))
            
            self.results['dataloader'] = {
                'status': 'passed',
                'batch_size': 16,
                'batch_keys': list(batch.keys()),
                'batch_shapes': {k: list(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}
            }
            
            print("✓ DataLoader created successfully")
            print(f"  Batch shapes: {self.results['dataloader']['batch_shapes']}")
            
            return dataloader
            
        except Exception as e:
            self.results['dataloader'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ DataLoader test failed: {e}")
            raise
            
    def test_forward_pass(self, model, dataloader):
        """Test model forward pass."""
        try:
            model.eval()
            
            with torch.no_grad():
                batch = next(iter(dataloader))
                
                # Move to device
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                loss = outputs.loss
                mlm_logits = outputs.logits
                # TLD logits not directly exposed in MaskedLMOutput
            
            self.results['forward_pass'] = {
                'status': 'passed',
                'loss': float(loss.item()),
                'mlm_logits_shape': list(mlm_logits.shape)
            }
            
            print("✓ Forward pass successful")
            print(f"  Loss: {loss.item():.4f}")
            
        except Exception as e:
            self.results['forward_pass'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Forward pass failed: {e}")
            
    def test_performance(self, model, dataloader):
        """Test performance metrics."""
        try:
            model.eval()
            
            # Memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / (1024**2)
            else:
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024**2)
            
            # Time multiple batches
            num_batches = 10
            start_time = time.time()
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    
                    if torch.cuda.is_available():
                        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                    
                    outputs = model(**batch)
            
            elapsed = time.time() - start_time
            
            # Memory after
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / (1024**2)
            else:
                memory_after = process.memory_info().rss / (1024**2)
            
            samples_per_second = (num_batches * 16) / elapsed  # batch_size = 16
            
            self.results['performance'] = {
                'status': 'passed',
                'batches_tested': num_batches,
                'elapsed_seconds': elapsed,
                'samples_per_second': samples_per_second,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_increase_mb': memory_after - memory_before
            }
            
            print("✓ Performance test completed")
            print(f"  Throughput: {samples_per_second:.1f} samples/second")
            print(f"  Memory usage: {memory_after - memory_before:.1f} MB")
            
        except Exception as e:
            self.results['performance'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Performance test failed: {e}")
            
    def test_checkpointing(self, model, tokenizer):
        """Test model checkpointing."""
        try:
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model with safe_serialization=False due to shared tensors
                model.save_pretrained(temp_dir, safe_serialization=False)
                tokenizer.save_pretrained(temp_dir)
                
                # Check files exist
                files_saved = list(Path(temp_dir).glob("*"))
                
                # Try loading
                loaded_model = DomainBertForMaskedLM.from_pretrained(temp_dir)
                loaded_tokenizer = DomainBertTokenizerFast.from_pretrained(temp_dir)
            
            self.results['checkpointing'] = {
                'status': 'passed',
                'files_saved': len(files_saved),
                'can_reload': True
            }
            
            print("✓ Checkpointing test successful")
            print(f"  Files saved: {len(files_saved)}")
            
        except Exception as e:
            self.results['checkpointing'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Checkpointing test failed: {e}")
            
    def test_multi_gpu(self):
        """Test multi-GPU setup."""
        try:
            if not torch.cuda.is_available():
                self.results['multi_gpu'] = {
                    'status': 'skipped',
                    'reason': 'No GPU available'
                }
                print("⚠ Multi-GPU test skipped (no GPU)")
                return
            
            gpu_count = torch.cuda.device_count()
            
            self.results['multi_gpu'] = {
                'status': 'passed',
                'gpu_count': gpu_count,
                'distributed_available': torch.distributed.is_available(),
                'nccl_available': torch.distributed.is_nccl_available()
            }
            
            print(f"✓ Multi-GPU setup ready")
            print(f"  GPUs available: {gpu_count}")
            print(f"  Distributed training: {'Yes' if torch.distributed.is_available() else 'No'}")
            
        except Exception as e:
            self.results['multi_gpu'] = {'status': 'failed', 'error': str(e)}
            print(f"✗ Multi-GPU test failed: {e}")
            
    def generate_report(self):
        """Generate final training readiness report."""
        print("\n" + "="*80)
        print("Training Readiness Report")
        print("="*80)
        
        # Summary
        total_tests = len(self.results)
        passed = sum(1 for r in self.results.values() if r.get('status') == 'passed')
        failed = sum(1 for r in self.results.values() if r.get('status') == 'failed')
        skipped = sum(1 for r in self.results.values() if r.get('status') == 'skipped')
        
        print(f"\nTest Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        
        # Save detailed report
        report = {
            'timestamp': self.start_time.isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'ready_for_training': failed == 0
            },
            'test_results': self.results,
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cpu_count': os.cpu_count()
            }
        }
        
        report_path = "training_readiness_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Training recommendations
        print("\nTraining Recommendations:")
        if failed == 0:
            print("✓ System is ready for training!")
            print("\nRecommended command:")
            print("python scripts/training/run_pretraining.py \\")
            print("  --data_dir data/raw/domains_project/data \\")
            print("  --sample full \\")
            print("  --output_dir models/domain-bert-pretrained \\")
            print("  --num_train_epochs 3 \\")
            print("  --per_device_train_batch_size 128 \\")
            print("  --gradient_accumulation_steps 4 \\")
            print("  --fp16")
        else:
            print("✗ Please fix the failed tests before training")
            for test_name, result in self.results.items():
                if result.get('status') == 'failed':
                    print(f"  - {test_name}: {result.get('error', 'Unknown error')}")


def main():
    """Run training readiness tests."""
    tester = TrainingReadinessTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()