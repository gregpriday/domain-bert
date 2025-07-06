#!/usr/bin/env python3
"""
Ultra-fast data preparation - no shuffling, just decompress and concatenate.
Shuffling can be done during training with random file access.
"""

import os
import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def process_file(args):
    """Process a single file and append to output"""
    file_path, output_file, index = args
    
    domains_count = 0
    temp_file = f"{output_file}.part{index:04d}"
    
    try:
        # Prepare decompression command
        if str(file_path).endswith('.xz'):
            cmd = ['xzcat', str(file_path)]
        elif str(file_path).endswith('.gz'):
            cmd = ['zcat', str(file_path)]
        elif str(file_path).endswith('.txt'):
            # For text files, just copy valid domains
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                with open(temp_file, 'w', encoding='utf-8') as out_f:
                    for line in in_f:
                        domain = line.strip().lower()
                        if domain and '.' in domain and 3 < len(domain) < 254:
                            out_f.write(domain + '\n')
                            domains_count += 1
            return (temp_file, domains_count)
        else:
            return (None, 0)
        
        # Run decompression and filter
        with open(temp_file, 'w', encoding='utf-8', buffering=1024*1024) as out_f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1024*1024
            )
            
            for line in proc.stdout:
                domain = line.strip().lower()
                if domain and '.' in domain and 3 < len(domain) < 254:
                    out_f.write(domain + '\n')
                    domains_count += 1
            
            proc.wait()
        
        return (temp_file, domains_count)
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return (None, 0)


def main():
    # Setup paths
    data_dir = Path("/home/ubuntu/domain-bert/data/raw/domains_project/data")
    output_file = Path("/home/ubuntu/domain-bert/data/processed/all_domains_unshuffled.txt")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Finding domain files...")
    
    # Find all files
    files = []
    for pattern in ["*.xz", "*.gz", "*.txt"]:
        files.extend(data_dir.rglob(pattern))
    
    # Filter out small files and sort by size (process larger files first)
    files = [f for f in files if f.stat().st_size > 1024]
    files.sort(key=lambda f: f.stat().st_size, reverse=True)
    
    logger.info(f"Found {len(files)} domain files")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in files)
    logger.info(f"Total compressed size: {total_size / 1e9:.1f} GB")
    
    # Prepare arguments for multiprocessing
    args_list = [(f, output_file, i) for i, f in enumerate(files)]
    
    # Process files in parallel
    num_workers = cpu_count()
    logger.info(f"Processing files using {num_workers} CPU cores...")
    
    total_domains = 0
    temp_files = []
    
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for temp_file, count in pool.imap_unordered(process_file, args_list):
                if temp_file:
                    temp_files.append(temp_file)
                    total_domains += count
                pbar.update(1)
                pbar.set_postfix({'domains': f'{total_domains/1e6:.1f}M'})
    
    # Concatenate all temporary files
    logger.info(f"\nConcatenating {len(temp_files)} temporary files...")
    
    with open(output_file, 'wb') as out_f:
        for temp_file in tqdm(sorted(temp_files), desc="Merging"):
            with open(temp_file, 'rb') as in_f:
                # Use sendfile for fast copying
                while True:
                    chunk = in_f.read(1024*1024*10)  # 10MB chunks
                    if not chunk:
                        break
                    out_f.write(chunk)
            os.remove(temp_file)
    
    # Get final stats
    file_size = output_file.stat().st_size
    
    logger.info(f"\nComplete!")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Total domains: {total_domains:,}")
    logger.info(f"File size: {file_size / 1e9:.2f} GB")
    
    # Create shuffled version using shuf command (very fast)
    shuffled_file = output_file.parent / "all_domains_shuffled.txt"
    logger.info(f"\nCreating shuffled version using system shuf command...")
    
    start = time.time()
    result = subprocess.run(
        f"shuf {output_file} > {shuffled_file}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        elapsed = time.time() - start
        logger.info(f"Shuffling complete in {elapsed:.1f} seconds!")
        
        # Delete the unshuffled file to save space
        logger.info("Removing unshuffled file to save disk space...")
        output_file.unlink()
        logger.info(f"Deleted {output_file}")
        
        # Create sample files from shuffled data
        logger.info("Creating sample files...")
        
        with open(shuffled_file, 'r') as in_f:
            # 10M sample
            sample_10m = output_file.parent / "domains_sample_10m.txt"
            with open(sample_10m, 'w') as out_f:
                for i, line in enumerate(in_f):
                    if i >= 10_000_000:
                        break
                    out_f.write(line)
            
            # Reset to beginning
            in_f.seek(0)
            
            # 1M sample
            sample_1m = output_file.parent / "domains_sample_1m.txt"
            with open(sample_1m, 'w') as out_f:
                for i, line in enumerate(in_f):
                    if i >= 1_000_000:
                        break
                    out_f.write(line)
        
        logger.info(f"Created {sample_10m} and {sample_1m}")
        
        # Final summary
        final_size = shuffled_file.stat().st_size
        logger.info(f"\nFinal files:")
        logger.info(f"  - {shuffled_file} ({final_size/1e9:.1f} GB)")
        logger.info(f"  - {sample_10m} ({sample_10m.stat().st_size/1e6:.1f} MB)")
        logger.info(f"  - {sample_1m} ({sample_1m.stat().st_size/1e6:.1f} MB)")
    else:
        logger.error(f"Shuffling failed: {result.stderr}")
        logger.info("You can still use the unshuffled file for training")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")