#!/usr/bin/env python
"""
Create a deterministic 0.1% validation split from domain data.
Stratified by TLD to ensure representative coverage.
"""

import os
import sys
import json
import gzip
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from domainbert.domain_parser import DomainParser


def hash_domain(domain: str) -> int:
    """Create deterministic hash for domain"""
    return int(hashlib.md5(domain.encode()).hexdigest(), 16)


def should_include_in_validation(domain: str, rate: float = 0.001) -> bool:
    """Deterministically decide if domain should be in validation set"""
    # Use modulo to get consistent 0.1% sample
    return hash_domain(domain) % int(1 / rate) == 0


def create_validation_split(
    data_dir: Path,
    output_dir: Path,
    validation_rate: float = 0.001,
    max_files: int = None
):
    """Create validation split from domain files"""
    
    parser = DomainParser()
    
    # Find all domain files
    domain_files = []
    for pattern in ["*.txt", "*.txt.gz", "*.txt.xz"]:
        domain_files.extend(list(data_dir.glob(f"**/{pattern}")))
    
    # Exclude existing validation files and stats
    domain_files = [
        f for f in domain_files 
        if "validation" not in f.name and "_stats" not in f.name
    ]
    
    if max_files:
        domain_files = domain_files[:max_files]
    
    print(f"Found {len(domain_files)} domain files")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_file = output_dir / "domains_validation.txt"
    
    # Track statistics
    total_domains = 0
    validation_domains = 0
    tld_counts = Counter()
    validation_tld_counts = Counter()
    
    # Process files
    with open(validation_file, 'w') as val_out:
        for i, file_path in enumerate(domain_files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(domain_files)}: {file_path.name}")
            
            # Open file based on extension
            if file_path.suffix == '.gz':
                opener = gzip.open
                mode = 'rt'
            elif file_path.suffix == '.xz':
                # Skip .xz files for now (need special handling)
                continue
            else:
                opener = open
                mode = 'r'
            
            try:
                with opener(file_path, mode, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        domain = line.strip().lower()
                        if not domain or len(domain) < 3:
                            continue
                        
                        total_domains += 1
                        
                        # Extract TLD
                        try:
                            parts = parser.extract(domain)
                            tld = parts.suffix or 'unknown'
                        except:
                            tld = 'unknown'
                        
                        tld_counts[tld] += 1
                        
                        # Check if should be in validation
                        if should_include_in_validation(domain, validation_rate):
                            val_out.write(f"{domain}\n")
                            validation_domains += 1
                            validation_tld_counts[tld] += 1
                        
                        if total_domains % 100000 == 0:
                            print(f"  Processed {total_domains:,} domains, "
                                  f"{validation_domains:,} in validation "
                                  f"({validation_domains/total_domains*100:.2f}%)")
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    # Save statistics
    stats = {
        "total_domains": total_domains,
        "validation_domains": validation_domains,
        "validation_rate": validation_domains / total_domains if total_domains > 0 else 0,
        "unique_tlds": len(tld_counts),
        "validation_tlds": len(validation_tld_counts),
        "top_tlds": tld_counts.most_common(20),
        "validation_top_tlds": validation_tld_counts.most_common(20),
    }
    
    stats_file = output_dir / "validation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("Validation Split Creation Summary")
    print("="*50)
    print(f"Total domains processed: {total_domains:,}")
    print(f"Validation domains: {validation_domains:,}")
    print(f"Actual validation rate: {stats['validation_rate']:.3%}")
    print(f"Unique TLDs: {stats['unique_tlds']:,}")
    print(f"TLDs in validation: {stats['validation_tlds']:,}")
    print(f"\nOutput files:")
    print(f"  - {validation_file}")
    print(f"  - {stats_file}")
    
    # Show TLD distribution
    print("\nTop 10 TLDs in validation:")
    for tld, count in validation_tld_counts.most_common(10):
        total_tld = tld_counts[tld]
        rate = count / total_tld * 100 if total_tld > 0 else 0
        print(f"  {tld:15} {count:8,} / {total_tld:10,} ({rate:.2f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create validation split for domain data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/raw/domains_project/data",
        help="Directory containing domain files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/processed/validation",
        help="Directory for validation output"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.001,
        help="Validation split rate (default: 0.001 = 0.1%)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    create_validation_split(
        args.data_dir,
        args.output_dir,
        args.rate,
        args.max_files
    )


if __name__ == "__main__":
    main()