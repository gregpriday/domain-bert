#!/usr/bin/env python3
"""
Download The Domains Project dataset
Contains 2.6+ billion domain names
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil


def check_dependencies():
    """Check if required tools are installed"""
    required_tools = ['git', 'git-lfs', 'xz']
    missing_tools = []
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"Error: Missing required tools: {', '.join(missing_tools)}")
        print("\nInstallation instructions:")
        print("Ubuntu/Debian: sudo apt-get install git git-lfs xz-utils")
        print("macOS: brew install git git-lfs xz")
        print("Then run: git lfs install")
        return False
    
    return True


def check_disk_space(target_dir: Path, required_gb: int = 200):
    """Check if there's enough disk space"""
    stat = os.statvfs(target_dir.parent)
    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    
    if available_gb < required_gb:
        print(f"Warning: Only {available_gb:.1f}GB available, {required_gb}GB recommended")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'
    
    print(f"Disk space OK: {available_gb:.1f}GB available")
    return True


def clone_repository(target_dir: Path):
    """Clone The Domains Project repository"""
    repo_url = "https://github.com/tb0hdan/domains.git"
    
    print(f"Cloning repository to {target_dir}...")
    
    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            check=True
        )
        
        # Change to repo directory
        os.chdir(target_dir)
        
        # Pull LFS files
        print("Downloading domain data files (this may take a while)...")
        subprocess.run(["git", "lfs", "pull"], check=True)
        
        print("Download complete!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
        return False


def extract_sample(data_dir: Path, sample_size: int = 1000000):
    """Extract a sample of domains for testing"""
    sample_file = data_dir / "domains_sample.txt"
    
    if sample_file.exists():
        print(f"Sample file already exists: {sample_file}")
        return sample_file
    
    print(f"Extracting sample of {sample_size:,} domains...")
    
    # Find first compressed file
    xz_files = sorted(data_dir.glob("domains-part-*.xz"))
    if not xz_files:
        print("No compressed domain files found!")
        return None
    
    first_file = xz_files[0]
    print(f"Using {first_file.name} for sample...")
    
    # Extract sample using xz and head
    try:
        cmd = f"xzcat '{first_file}' | head -n {sample_size} > '{sample_file}'"
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"Sample saved to: {sample_file}")
        return sample_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting sample: {e}")
        return None


def verify_download(data_dir: Path):
    """Verify the downloaded data"""
    xz_files = list(data_dir.glob("domains-part-*.xz"))
    
    if not xz_files:
        print("No domain data files found!")
        return False
    
    print(f"\nDownload verification:")
    print(f"Found {len(xz_files)} compressed domain files")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in xz_files)
    total_gb = total_size / (1024**3)
    print(f"Total compressed size: {total_gb:.1f}GB")
    
    # List files
    print("\nDomain files:")
    for f in sorted(xz_files)[:5]:
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name}: {size_mb:.1f}MB")
    
    if len(xz_files) > 5:
        print(f"  ... and {len(xz_files) - 5} more files")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download The Domains Project dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/domains_project",
        help="Directory to store the dataset (default: data/raw/domains_project)"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Extract only a sample instead of downloading full dataset"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000000,
        help="Number of domains to extract for sample"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if directory exists"
    )
    
    args = parser.parse_args()
    
    # Convert to Path
    data_dir = Path(args.data_dir).absolute()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check if already exists
    if data_dir.exists() and args.skip_download:
        print(f"Directory already exists: {data_dir}")
        print("Use --skip-download=false to re-download")
        
        # Verify existing download
        if verify_download(data_dir / "data"):
            # Extract sample if requested
            if args.sample_only:
                extract_sample(data_dir / "data", args.sample_size)
            return 0
        else:
            return 1
    
    # Create parent directory
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Check disk space
    if not args.sample_only:
        if not check_disk_space(data_dir, required_gb=200):
            return 1
    
    # Download
    if not args.skip_download and not data_dir.exists():
        if not clone_repository(data_dir):
            return 1
    
    # Verify download
    if not verify_download(data_dir / "data"):
        return 1
    
    # Extract sample
    extract_sample(data_dir / "data", args.sample_size)
    
    print("\n✅ Setup complete!")
    print(f"Domain data location: {data_dir / 'data'}")
    print("\nNext steps:")
    print("1. Run preprocessing: python scripts/data/preprocess_domains.py")
    print("2. Run pretraining: python scripts/train/domain_bert/pretrain_domain_bert.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())