#!/usr/bin/env python3
"""
Create a manifest of all domain files with statistics.
This helps with efficient data loading and sampling strategies.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures


def count_lines_in_xz(file_path: Path) -> int:
    """Count lines in compressed .xz file."""
    try:
        result = subprocess.run(
            ['xzcat', str(file_path), '|', 'wc', '-l'],
            shell=True,
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip())
    except:
        return 0


def analyze_file(file_path: Path) -> Dict:
    """Analyze a single domain file."""
    stats = {
        'path': str(file_path),
        'name': file_path.name,
        'size_bytes': file_path.stat().st_size,
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'tld_category': file_path.parent.name,
        'domain_count': 0
    }
    
    # Extract TLD from filename (e.g., domain2multi-com00.txt.xz -> com)
    if 'domain2multi-' in file_path.name:
        tld_part = file_path.name.replace('domain2multi-', '').replace('.txt.xz', '')
        # Remove the number suffix (e.g., com00 -> com)
        tld = ''.join(c for c in tld_part if not c.isdigit())
        stats['primary_tld'] = tld
    else:
        stats['primary_tld'] = 'unknown'
    
    # Count domains (sampling for speed)
    if stats['size_mb'] < 10:  # Small files - count all
        stats['domain_count'] = count_lines_in_xz(file_path)
        stats['count_method'] = 'exact'
    else:  # Large files - estimate
        # Sample first 10000 lines
        sample_cmd = f"xzcat '{file_path}' | head -10000 | wc -l"
        result = subprocess.run(sample_cmd, shell=True, capture_output=True, text=True)
        sample_lines = int(result.stdout.strip())
        
        # Get compressed ratio estimate
        sample_cmd2 = f"xzcat '{file_path}' | head -10000 | wc -c"
        result2 = subprocess.run(sample_cmd2, shell=True, capture_output=True, text=True)
        sample_bytes = int(result2.stdout.strip())
        
        # Estimate total lines based on file size
        if sample_bytes > 0:
            bytes_per_line = sample_bytes / sample_lines
            # Assume 10:1 compression ratio for .xz
            estimated_uncompressed = stats['size_bytes'] * 10
            stats['domain_count'] = int(estimated_uncompressed / bytes_per_line)
            stats['count_method'] = 'estimated'
        else:
            stats['domain_count'] = 0
            stats['count_method'] = 'failed'
    
    return stats


def create_manifest(data_dir: Path, output_path: Path):
    """Create manifest of all domain files."""
    print(f"Scanning {data_dir} for domain files...")
    
    # Find all .xz files
    xz_files = list(data_dir.glob("**/*.txt.xz"))
    print(f"Found {len(xz_files)} compressed domain files")
    
    # Analyze files in parallel
    manifest = {
        'total_files': len(xz_files),
        'total_size_gb': 0,
        'total_domains': 0,
        'files': [],
        'by_tld': defaultdict(lambda: {'files': 0, 'domains': 0, 'size_mb': 0}),
        'by_category': defaultdict(lambda: {'files': 0, 'domains': 0, 'size_mb': 0})
    }
    
    print("Analyzing files...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all files for analysis
        future_to_file = {executor.submit(analyze_file, f): f for f in xz_files}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(xz_files)):
            try:
                stats = future.result()
                manifest['files'].append(stats)
                manifest['total_size_gb'] += stats['size_mb'] / 1024
                manifest['total_domains'] += stats['domain_count']
                
                # Update TLD stats
                tld = stats['primary_tld']
                manifest['by_tld'][tld]['files'] += 1
                manifest['by_tld'][tld]['domains'] += stats['domain_count']
                manifest['by_tld'][tld]['size_mb'] += stats['size_mb']
                
                # Update category stats
                category = stats['tld_category']
                manifest['by_category'][category]['files'] += 1
                manifest['by_category'][category]['domains'] += stats['domain_count']
                manifest['by_category'][category]['size_mb'] += stats['size_mb']
                
            except Exception as e:
                print(f"Error processing {future_to_file[future]}: {e}")
    
    # Convert defaultdicts to regular dicts for JSON serialization
    manifest['by_tld'] = dict(manifest['by_tld'])
    manifest['by_category'] = dict(manifest['by_category'])
    
    # Sort files by size for better load balancing
    manifest['files'].sort(key=lambda x: x['size_bytes'], reverse=True)
    
    # Add summary statistics
    manifest['summary'] = {
        'total_files': manifest['total_files'],
        'total_size_gb': round(manifest['total_size_gb'], 2),
        'total_domains': manifest['total_domains'],
        'avg_file_size_mb': round(manifest['total_size_gb'] * 1024 / manifest['total_files'], 2),
        'avg_domains_per_file': int(manifest['total_domains'] / manifest['total_files']),
        'unique_tlds': len(manifest['by_tld']),
        'unique_categories': len(manifest['by_category'])
    }
    
    # Find largest TLDs
    top_tlds = sorted(
        manifest['by_tld'].items(),
        key=lambda x: x[1]['domains'],
        reverse=True
    )[:20]
    manifest['top_tlds'] = {tld: stats for tld, stats in top_tlds}
    
    # Save manifest
    print(f"\nSaving manifest to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Manifest Summary")
    print("="*60)
    print(f"Total files: {manifest['summary']['total_files']:,}")
    print(f"Total size: {manifest['summary']['total_size_gb']:.2f} GB")
    print(f"Total domains: {manifest['summary']['total_domains']:,}")
    print(f"Average file size: {manifest['summary']['avg_file_size_mb']:.2f} MB")
    print(f"Average domains per file: {manifest['summary']['avg_domains_per_file']:,}")
    print(f"Unique TLDs: {manifest['summary']['unique_tlds']}")
    print(f"Unique categories: {manifest['summary']['unique_categories']}")
    
    print("\nTop 10 TLDs by domain count:")
    for i, (tld, stats) in enumerate(top_tlds[:10]):
        print(f"  {i+1}. .{tld}: {stats['domains']:,} domains in {stats['files']} files")
    
    print("\nFile size distribution:")
    sizes = [f['size_mb'] for f in manifest['files']]
    print(f"  Min: {min(sizes):.2f} MB")
    print(f"  Max: {max(sizes):.2f} MB")
    print(f"  Median: {sorted(sizes)[len(sizes)//2]:.2f} MB")
    
    return manifest


def create_balanced_file_groups(manifest: Dict, num_workers: int) -> List[List[str]]:
    """Create balanced file groups for multi-worker loading."""
    files = manifest['files']
    
    # Sort by size (largest first)
    files.sort(key=lambda x: x['size_bytes'], reverse=True)
    
    # Create worker groups
    groups = [[] for _ in range(num_workers)]
    group_sizes = [0] * num_workers
    
    # Assign files to groups using greedy algorithm
    for file_info in files:
        # Find group with smallest total size
        min_group = min(range(num_workers), key=lambda i: group_sizes[i])
        groups[min_group].append(file_info['path'])
        group_sizes[min_group] += file_info['size_bytes']
    
    # Print balance statistics
    print(f"\nFile distribution across {num_workers} workers:")
    for i, (group, size) in enumerate(zip(groups, group_sizes)):
        print(f"  Worker {i}: {len(group)} files, {size/(1024**3):.2f} GB")
    
    return groups


def create_sampling_strategy(manifest: Dict) -> Dict:
    """Create a sampling strategy based on TLD distribution."""
    total_domains = manifest['total_domains']
    
    # Calculate sampling weights
    weights = {}
    for tld, stats in manifest['by_tld'].items():
        # Weight based on domain count (could be adjusted)
        weight = stats['domains'] / total_domains
        weights[tld] = {
            'weight': weight,
            'domains': stats['domains'],
            'files': stats['files'],
            'probability': min(1.0, weight * 10)  # Boost rare TLDs
        }
    
    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1]['weight'], reverse=True)
    
    strategy = {
        'method': 'weighted_sampling',
        'total_domains': total_domains,
        'tld_weights': dict(sorted_weights[:100]),  # Top 100 TLDs
        'sampling_rules': {
            'common_tlds': [tld for tld, w in sorted_weights[:10]],
            'rare_tlds': [tld for tld, w in sorted_weights[-100:]],
            'boost_rare': True
        }
    }
    
    return strategy


def main():
    """Create dataset manifest."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create manifest of domain dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw/domains_project/data",
                        help="Path to domains data directory")
    parser.add_argument("--output", type=str, default="data/processed/domains/dataset_manifest.json",
                        help="Output path for manifest")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers for balanced loading")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest = create_manifest(data_dir, output_path)
    
    # Create balanced file groups
    if args.num_workers > 1:
        groups = create_balanced_file_groups(manifest, args.num_workers)
        
        # Save groups
        groups_path = output_path.parent / "worker_file_groups.json"
        with open(groups_path, 'w') as f:
            json.dump({'num_workers': args.num_workers, 'groups': groups}, f, indent=2)
        print(f"\nWorker file groups saved to {groups_path}")
    
    # Create sampling strategy
    strategy = create_sampling_strategy(manifest)
    strategy_path = output_path.parent / "sampling_strategy.json"
    with open(strategy_path, 'w') as f:
        json.dump(strategy, f, indent=2)
    print(f"Sampling strategy saved to {strategy_path}")


if __name__ == "__main__":
    main()