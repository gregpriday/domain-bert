#!/usr/bin/env python
"""
Build TLD vocabulary from domain dataset with true parallel processing.

This script analyzes domain data to create a TLD vocabulary file
that the DomainBERT tokenizer uses for TLD embeddings.
"""
import argparse
import sys
import json
from pathlib import Path
from collections import Counter
from typing import Iterator, Tuple, Dict, Optional, List
import tldextract
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from domainbert.tokenizer import DomainBertTokenizerFast


def process_file(file_path: str, min_count: int = 10) -> Tuple[Counter, int, int]:
    """
    Process a single file and return TLD counts.
    This runs in a separate process.
    """
    # Pre-initialize tldextract to avoid slow first call
    _ = tldextract.extract("example.com")
    
    tld_counts = Counter()
    total_domains = 0
    failed_extractions = 0
    
    if file_path.endswith('.xz'):
        # Use xz command to stream compressed file
        proc = subprocess.Popen(
            ['xzcat', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1024*1024  # 1MB buffer
        )
        
        for line in proc.stdout:
            domain = line.strip().lower()
            if domain and '.' in domain:
                total_domains += 1
                try:
                    extracted = tldextract.extract(domain)
                    if extracted.suffix:
                        # Only count single-level TLDs (no dots) or common 2-level ones
                        if '.' not in extracted.suffix or extracted.suffix.count('.') == 1:
                            tld_counts[extracted.suffix] += 1
                    else:
                        failed_extractions += 1
                except Exception:
                    failed_extractions += 1
    else:
        # Regular text file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                domain = line.strip().lower()
                if domain and '.' in domain:
                    total_domains += 1
                    try:
                        extracted = tldextract.extract(domain)
                        if extracted.suffix:
                            tld_counts[extracted.suffix] += 1
                        else:
                            failed_extractions += 1
                    except Exception:
                        failed_extractions += 1
    
    return tld_counts, total_domains, failed_extractions


def find_domain_files(data_dir: Path) -> List[Path]:
    """Find all domain files in directory and subdirectories."""
    patterns = ["**/*.txt", "**/*.xz"]
    files = []
    
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))
    
    # Exclude any files that might be documentation or stats
    files = [f for f in files if not any(
        skip in f.name for skip in ['README', 'LICENSE', 'stats', 'STATS']
    )]
    
    return files


def analyze_tlds_parallel(data_dir: Path, min_count: int = 10, num_processes: int = None) -> Tuple[list, Dict]:
    """
    Analyze TLD distribution using true parallel file processing.
    Each file is processed in a separate process.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Find all files
    files = find_domain_files(data_dir)
    if not files:
        raise ValueError(f"No domain files found in {data_dir}")
    
    print(f"Found {len(files)} domain files to process")
    print(f"Using {num_processes} processes for parallel processing")
    
    # Global counters
    total_tld_counts = Counter()
    total_domains = 0
    total_failed = 0
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(process_file, str(file_path), min_count): file_path 
            for file_path in files
        }
        
        # Process completed files with progress bar
        with tqdm(total=len(files), desc="Processing files", unit=" files") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    tld_counts, domains_count, failed_count = future.result()
                    
                    # Update global counters
                    total_tld_counts.update(tld_counts)
                    total_domains += domains_count
                    total_failed += failed_count
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'domains': f"{total_domains:,}",
                        'failed': f"{total_failed:,}",
                        'tlds': len(total_tld_counts)
                    })
                    
                except Exception as e:
                    print(f"\nError processing {file_path}: {e}")
    
    print(f"\n✓ Processed {total_domains:,} total domains")
    print(f"✓ Failed to extract TLD from {total_failed:,} entries")
    
    # Filter by minimum count
    vocab_tlds = [tld for tld, count in total_tld_counts.items() if count >= min_count]
    vocab_tlds.sort()
    
    print(f"Found {len(total_tld_counts)} unique TLDs")
    print(f"Found {len(vocab_tlds)} TLDs with at least {min_count} occurrences")
    
    # Compile statistics
    stats = {
        "total_domains": total_domains,
        "failed_extractions": total_failed,
        "total_unique_tlds": len(total_tld_counts),
        "vocab_size": len(vocab_tlds),
        "min_count": min_count,
        "top_tlds": [
            {
                "tld": tld,
                "count": count,
                "percentage": (count/total_domains)*100 if total_domains > 0 else 0
            }
            for tld, count in total_tld_counts.most_common(50)
        ],
        "excluded_tlds": [
            {"tld": tld, "count": count}
            for tld, count in total_tld_counts.items()
            if count < min_count
        ][:100]
    }
    
    return vocab_tlds, stats


def build_and_save_vocabulary(vocab_tlds: list, output_dir: Path, stats: Dict) -> Path:
    """Build the TLD vocabulary using the tokenizer and save it."""
    print("\nBuilding TLD vocabulary...")
    
    # Create tokenizer and manually set vocabulary
    tokenizer = DomainBertTokenizerFast(vocab_file=None)
    for tld in vocab_tlds:
        if tld not in tokenizer.tld_to_id:
            idx = len(tokenizer.tld_to_id)
            tokenizer.tld_to_id[tld] = idx
            tokenizer.id_to_tld[idx] = tld
    
    # Save vocabulary
    tokenizer.save_vocabulary(str(output_dir))
    vocab_file = output_dir / "tld_vocab.json"
    
    print(f"Saved TLD vocabulary to {vocab_file}")
    
    # Save statistics
    stats_file = output_dir / "tld_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved TLD statistics to {stats_file}")
    
    # Print summary of top TLDs
    print("\nTop 20 TLDs by frequency:")
    for i, tld_info in enumerate(stats["top_tlds"][:20], 1):
        print(f"{i:2d}. {tld_info['tld']:15s} {tld_info['count']:10,d} ({tld_info['percentage']:5.2f}%)")
    
    return vocab_file


def main():
    parser = argparse.ArgumentParser(
        description="Build TLD vocabulary from domain dataset (parallel version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build vocabulary from raw domain files
  %(prog)s --data-dir data/raw/domains_project/data --output-dir data/processed/domains
  
  # Build vocabulary with custom minimum count and more processes
  %(prog)s --data-dir data/raw/domains_project/data --output-dir data/processed/domains --min-count 100 --processes 16
  
  # Build vocabulary from processed samples
  %(prog)s --data-dir data/processed/domains --output-dir data/processed/domains --min-count 5 --processes 4
"""
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing domain files (.txt or .xz)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save vocabulary and statistics"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum occurrences for TLD to be included in vocabulary (default: 10)"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes for parallel processing (default: number of CPU cores)"
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
    
    # Analyze TLDs using parallel processing
    vocab_tlds, stats = analyze_tlds_parallel(data_dir, args.min_count, args.processes)
    
    if not vocab_tlds:
        print("Error: No TLDs found that meet the minimum count threshold")
        sys.exit(1)
    
    # Build and save vocabulary
    vocab_file = build_and_save_vocabulary(vocab_tlds, output_dir, stats)
    
    print(f"\n✅ TLD vocabulary built successfully!")
    print(f"Vocabulary size: {len(vocab_tlds)} TLDs")
    print(f"Output location: {output_dir}")


if __name__ == "__main__":
    main()