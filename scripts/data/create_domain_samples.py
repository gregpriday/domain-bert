#!/usr/bin/env python
"""
Create stratified domain samples for training.

This script creates various sized samples from the full domain dataset,
maintaining proportional TLD representation through stratified sampling.
"""
import argparse
import sys
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Iterator, List, Tuple, Optional
import tldextract
from tqdm import tqdm

# Sample sizes for different datasets
SAMPLE_SIZES = {
    "tiny": 10_000,
    "small": 1_000_000,
    "medium": 10_000_000,
    "large": 100_000_000,
}


def iter_domain_files(data_dir: Path) -> Iterator[Tuple[str, Optional[str]]]:
    """
    Iterate through all domain files in a directory.
    
    Yields:
        Tuple of (domain, file_name) for progress tracking
    """
    # Handle both .txt and .xz compressed files
    patterns = ["*.txt", "*.xz"]
    files = []
    
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))
    
    # Exclude stats files
    files = [f for f in files if not f.name.endswith("_stats.json") and not f.name.endswith("_stats.txt")]
    
    if not files:
        raise ValueError(f"No domain files found in {data_dir}")
    
    for file_path in files:
        if file_path.suffix == '.xz':
            # Use xz command to stream compressed file
            import subprocess
            proc = subprocess.Popen(
                ['xzcat', str(file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1024*1024  # 1MB buffer
            )
            for line in proc.stdout:
                domain = line.strip().lower()
                if domain and '.' in domain:
                    yield domain, file_path.name
        else:
            # Regular text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    domain = line.strip().lower()
                    if domain and '.' in domain:
                        yield domain, file_path.name


def load_tld_distribution(stats_file: Path) -> Dict[str, float]:
    """Load TLD distribution from stats file."""
    if not stats_file.exists():
        raise FileNotFoundError(
            f"TLD stats file not found: {stats_file}\n"
            "Please run build_tld_vocabulary.py first"
        )
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Convert to distribution dictionary
    distribution = {}
    total = stats["total_domains"]
    
    for tld_info in stats["top_tlds"]:
        distribution[tld_info["tld"]] = tld_info["percentage"] / 100.0
    
    return distribution


def create_stratified_sample(
    data_dir: Path,
    output_file: Path,
    sample_size: int,
    tld_distribution: Dict[str, float],
    random_seed: int = 42,
    total_domains: int = 1_766_025_618
) -> Dict[str, int]:
    """
    Create a stratified sample of domains.
    
    Returns:
        Dictionary mapping TLD to actual count in the sample
    """
    print(f"\nCreating {output_file.name} with {sample_size:,} domains...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Calculate target counts per TLD
    target_counts = {}
    for tld, proportion in tld_distribution.items():
        target_counts[tld] = int(sample_size * proportion)
    
    # Adjust for rounding errors
    remaining = sample_size - sum(target_counts.values())
    if remaining > 0:
        # Add remaining to most common TLDs
        for tld in sorted(tld_distribution, key=tld_distribution.get, reverse=True)[:remaining]:
            target_counts[tld] += 1
    
    print(f"Target distribution: {len(target_counts)} TLDs")
    
    # Collect domains by TLD
    print("Collecting domains by TLD...")
    domains_by_tld = defaultdict(list)
    total_collected = 0
    current_file = None
    
    # Calculate total needed
    total_needed = sum(count * 2 for count in target_counts.values())
    
    # Create progress bar with total domains count for better visibility
    with tqdm(desc="Scanning domains", total=total_domains, unit=" domains") as pbar:
        for domain, file_name in iter_domain_files(data_dir):
            # Update description when file changes
            if file_name != current_file:
                current_file = file_name
                pbar.set_description(f"Collecting from {file_name}")
            
            try:
                extracted = tldextract.extract(domain)
                if extracted.suffix and extracted.suffix in target_counts:
                    tld = extracted.suffix
                    needed = target_counts[tld]
                    current = len(domains_by_tld[tld])
                    
                    # Collect 2x target to allow for random sampling
                    if current < needed * 2:
                        domains_by_tld[tld].append(domain)
                        total_collected += 1
                        pbar.update(1)
                        
                        # Update postfix with collection progress
                        if total_collected % 5000 == 0:
                            tlds_complete = sum(
                                1 for tld, count in target_counts.items()
                                if len(domains_by_tld.get(tld, [])) >= count
                            )
                            pbar.set_postfix({
                                'tlds_ready': f"{tlds_complete}/{len(target_counts)}",
                                'collected': f"{total_collected:,}"
                            })
            except Exception:
                continue
            
            # Check if we have enough for all TLDs
            have_enough = all(
                len(domains_by_tld[tld]) >= count
                for tld, count in target_counts.items()
            )
            if have_enough:
                pbar.set_description("Collection complete!")
                break
    
    # Sample from each TLD
    print("Sampling domains...")
    sampled_domains = []
    actual_counts = {}
    
    for tld, target_count in sorted(target_counts.items()):
        available = domains_by_tld.get(tld, [])
        if available:
            sample_count = min(len(available), target_count)
            if sample_count > 0:
                sampled = random.sample(available, sample_count)
                sampled_domains.extend(sampled)
                actual_counts[tld] = sample_count
        else:
            print(f"Warning: No domains found for TLD '{tld}'")
    
    # Shuffle the final sample
    random.shuffle(sampled_domains)
    
    # Write to file
    print(f"Writing {len(sampled_domains):,} domains to {output_file}...")
    with open(output_file, 'w') as f:
        for domain in sampled_domains:
            f.write(f"{domain}\n")
    
    return actual_counts


def save_sample_stats(
    output_dir: Path,
    sample_name: str,
    sample_size: int,
    actual_counts: Dict[str, int]
):
    """Save statistics about the created sample."""
    stats = {
        "sample_name": sample_name,
        "target_size": sample_size,
        "actual_size": sum(actual_counts.values()),
        "num_tlds": len(actual_counts),
        "tld_counts": actual_counts,
        "top_tlds": sorted(
            [{"tld": tld, "count": count} for tld, count in actual_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:20]
    }
    
    stats_file = output_dir / f"domains_{sample_name}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved sample statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified domain samples for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all sample sizes
  %(prog)s --data-dir data/raw/domains_project/data --output-dir data/processed/domains
  
  # Create specific sample sizes
  %(prog)s --data-dir data/raw/domains_project/data --output-dir data/processed/domains --samples tiny small
  
  # Create sample with custom TLD stats file
  %(prog)s --data-dir data/raw/domains_project/data --output-dir data/processed/domains \\
    --tld-stats-file data/processed/domains/tld_stats.json --samples medium
"""
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing domain files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save sample files"
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=list(SAMPLE_SIZES.keys()) + ["all"],
        default=["all"],
        help="Sample sizes to create (default: all)"
    )
    parser.add_argument(
        "--tld-stats-file",
        type=str,
        default=None,
        help="Path to TLD statistics file (default: output-dir/tld_stats.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load TLD distribution
    if args.tld_stats_file:
        stats_file = Path(args.tld_stats_file)
    else:
        stats_file = output_dir / "tld_stats.json"
    
    try:
        tld_distribution = load_tld_distribution(stats_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Determine which samples to create
    if "all" in args.samples:
        samples_to_create = list(SAMPLE_SIZES.keys())
    else:
        samples_to_create = args.samples
    
    # Create each sample
    for sample_name in samples_to_create:
        sample_size = SAMPLE_SIZES[sample_name]
        output_file = output_dir / f"domains_{sample_name}.txt"
        
        # Skip if already exists
        if output_file.exists():
            print(f"\nSkipping {sample_name} - file already exists: {output_file}")
            continue
        
        # Create stratified sample
        actual_counts = create_stratified_sample(
            data_dir,
            output_file,
            sample_size,
            tld_distribution,
            args.seed
        )
        
        # Save statistics
        save_sample_stats(output_dir, sample_name, sample_size, actual_counts)
        
        print(f"✅ Created {sample_name} sample: {sum(actual_counts.values()):,} domains")
    
    print(f"\n✅ All samples created successfully!")
    print(f"Output location: {output_dir}")


if __name__ == "__main__":
    main()