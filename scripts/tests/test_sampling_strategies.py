#!/usr/bin/env python3
"""
Test different sampling strategies for the domain dataset.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domainbert.tokenizer import DomainBertTokenizerFast
from domainbert.data.streaming_dataset import MultiFileStreamingDataset


class StratifiedDomainSampler:
    """Implements various sampling strategies for domain data."""
    
    def __init__(self, manifest_path: str):
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Load TLD weights
        self.tld_weights = {}
        total_domains = self.manifest['total_domains']
        
        for tld, stats in self.manifest['by_tld'].items():
            self.tld_weights[tld] = {
                'count': stats['domains'],
                'files': stats['files'],
                'weight': stats['domains'] / total_domains
            }
    
    def get_balanced_files(self, target_domains: int = 1000000) -> List[str]:
        """Get a balanced set of files covering diverse TLDs."""
        selected_files = []
        domains_selected = 0
        tld_counts = Counter()
        
        # Sort TLDs by rarity (ascending domain count)
        sorted_tlds = sorted(self.tld_weights.items(), key=lambda x: x[1]['count'])
        
        # First pass: ensure coverage of rare TLDs
        for tld, info in sorted_tlds:
            if domains_selected >= target_domains:
                break
            
            # Find files for this TLD
            tld_files = [f for f in self.manifest['files'] 
                        if f.get('primary_tld') == tld]
            
            if tld_files:
                # Take at least one file from each TLD
                file_info = tld_files[0]
                selected_files.append(file_info['path'])
                domains_selected += file_info['domain_count']
                tld_counts[tld] += file_info['domain_count']
        
        # Second pass: add more from common TLDs
        remaining = target_domains - domains_selected
        if remaining > 0:
            # Focus on top TLDs
            top_tlds = sorted(self.tld_weights.items(), 
                            key=lambda x: x[1]['count'], 
                            reverse=True)[:20]
            
            for tld, info in top_tlds:
                if domains_selected >= target_domains:
                    break
                
                tld_files = [f for f in self.manifest['files'] 
                           if f.get('primary_tld') == tld and 
                           f['path'] not in selected_files]
                
                for file_info in tld_files:
                    if domains_selected >= target_domains:
                        break
                    
                    selected_files.append(file_info['path'])
                    domains_selected += file_info['domain_count']
                    tld_counts[tld] += file_info['domain_count']
        
        print(f"Selected {len(selected_files)} files with ~{domains_selected:,} domains")
        print(f"TLD coverage: {len(tld_counts)} different TLDs")
        print(f"Top 5 TLDs: {tld_counts.most_common(5)}")
        
        return selected_files
    
    def get_curriculum_stages(self) -> List[Dict]:
        """Define curriculum learning stages."""
        stages = [
            {
                'name': 'high_quality',
                'description': 'Common TLDs with clean domains',
                'tlds': ['com', 'org', 'net', 'edu', 'gov'],
                'target_domains': 10000000,
                'filters': {
                    'min_length': 3,
                    'max_length': 20,
                    'exclude_numbers': False
                }
            },
            {
                'name': 'diverse_tlds',
                'description': 'Mix of common and uncommon TLDs',
                'tlds': None,  # All TLDs
                'target_domains': 50000000,
                'filters': {
                    'min_length': 3,
                    'max_length': 50
                }
            },
            {
                'name': 'full_diversity',
                'description': 'All domains including edge cases',
                'tlds': None,
                'target_domains': None,  # All domains
                'filters': None
            }
        ]
        
        return stages
    
    def get_importance_weighted_files(self, importance_scores: Optional[Dict[str, float]] = None) -> List[str]:
        """Get files weighted by importance scores."""
        if importance_scores is None:
            # Default importance scores based on TLD popularity and quality
            importance_scores = {
                'com': 1.0,
                'org': 0.9,
                'net': 0.9,
                'edu': 0.95,
                'gov': 0.95,
                'io': 0.8,
                'co': 0.7,
                'info': 0.6,
                'biz': 0.5
            }
        
        weighted_files = []
        
        for file_info in self.manifest['files']:
            tld = file_info.get('primary_tld', 'unknown')
            score = importance_scores.get(tld, 0.3)  # Default score for unknown TLDs
            
            # Adjust score based on file size (prefer medium-sized files)
            size_factor = 1.0
            if file_info['size_mb'] < 1:
                size_factor = 0.5  # Very small files might have quality issues
            elif file_info['size_mb'] > 50:
                size_factor = 0.8  # Very large files might be unwieldy
            
            final_score = score * size_factor
            
            # Probabilistic selection based on score
            if random.random() < final_score:
                weighted_files.append((file_info['path'], final_score))
        
        # Sort by score and return paths
        weighted_files.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in weighted_files]


def test_sampling_strategies():
    """Test different sampling strategies."""
    print("Testing Domain Sampling Strategies")
    print("="*60)
    
    # Load manifest
    manifest_path = "data/processed/domains/dataset_manifest.json"
    if not Path(manifest_path).exists():
        print("Error: Run create_dataset_manifest.py first!")
        return
    
    sampler = StratifiedDomainSampler(manifest_path)
    
    # Test 1: Balanced sampling
    print("\n1. Balanced Sampling Strategy")
    print("-"*40)
    balanced_files = sampler.get_balanced_files(target_domains=5000000)
    print(f"Files selected: {len(balanced_files)}")
    
    # Test 2: Curriculum stages
    print("\n2. Curriculum Learning Stages")
    print("-"*40)
    stages = sampler.get_curriculum_stages()
    for i, stage in enumerate(stages):
        print(f"Stage {i+1}: {stage['name']}")
        print(f"  Description: {stage['description']}")
        print(f"  Target domains: {stage['target_domains'] or 'all'}")
        print(f"  Filters: {stage['filters']}")
    
    # Test 3: Importance weighting
    print("\n3. Importance-Weighted Sampling")
    print("-"*40)
    weighted_files = sampler.get_importance_weighted_files()
    print(f"Files selected: {len(weighted_files)}")
    
    # Test actual loading with a small sample
    print("\n4. Testing Actual Data Loading")
    print("-"*40)
    
    # Use a few files for testing
    test_files = balanced_files[:3]
    
    # Load tokenizer
    tokenizer = DomainBertTokenizerFast(
        tld_vocab_file="tokenizer/tld_vocab.json",
        max_len=64
    )
    
    # Create dataset
    dataset = MultiFileStreamingDataset(
        file_paths=test_files,
        tokenizer=tokenizer,
        max_length=64,
        max_samples=100
    )
    
    # Test loading
    start_time = time.time()
    domains_loaded = []
    tld_distribution = Counter()
    
    for i, item in enumerate(dataset):
        domains_loaded.append(item)
        tld_distribution[item['tld_ids']] += 1
        
        if i >= 99:  # Load 100 samples
            break
    
    elapsed = time.time() - start_time
    
    print(f"Loaded {len(domains_loaded)} domains in {elapsed:.2f}s")
    print(f"TLD distribution: {dict(tld_distribution.most_common(5))}")
    
    # Test domain length distribution
    lengths = [len([x for x in item['input_ids'] if x != tokenizer.pad_token_id]) 
              for item in domains_loaded]
    print(f"Domain length stats:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")


def create_training_file_list(strategy: str = "balanced", output_path: str = "training_files.txt"):
    """Create a file list for training based on strategy."""
    manifest_path = "data/processed/domains/dataset_manifest.json"
    sampler = StratifiedDomainSampler(manifest_path)
    
    if strategy == "balanced":
        files = sampler.get_balanced_files(target_domains=100000000)  # 100M domains
    elif strategy == "importance":
        files = sampler.get_importance_weighted_files()
    elif strategy == "full":
        # All files
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        files = [f['path'] for f in manifest['files']]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Save file list
    with open(output_path, 'w') as f:
        for file_path in files:
            f.write(file_path + '\n')
    
    print(f"Saved {len(files)} files to {output_path}")


def main():
    """Run sampling strategy tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test domain sampling strategies")
    parser.add_argument("--create-file-list", action="store_true",
                        help="Create a training file list")
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=["balanced", "importance", "full"],
                        help="Sampling strategy to use")
    parser.add_argument("--output", type=str, default="training_files.txt",
                        help="Output path for file list")
    
    args = parser.parse_args()
    
    if args.create_file_list:
        create_training_file_list(args.strategy, args.output)
    else:
        test_sampling_strategies()


if __name__ == "__main__":
    main()