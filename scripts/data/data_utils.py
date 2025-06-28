"""
Data utilities for DomainBERT pretraining
"""

import csv
from pathlib import Path
from typing import Iterator, List, Union
import pandas as pd


class MajesticMillionLoader:
    """Load domains from Majestic Million CSV"""
    
    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Majestic Million file not found: {csv_path}")
    
    def iter_domains(self) -> Iterator[str]:
        """Iterate through domains in Majestic Million"""
        # Try to detect the CSV structure first
        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read first line to check if it's a header
            first_line = f.readline().strip()
            
            # Reset file pointer
            f.seek(0)
            
            # Check if first line contains typical header keywords
            if any(keyword in first_line.lower() for keyword in ['domain', 'rank', 'url']):
                # Has header
                reader = csv.DictReader(f)
                
                # Find the domain column
                domain_col = None
                for col in reader.fieldnames:
                    if 'domain' in col.lower() or 'url' in col.lower():
                        domain_col = col
                        break
                
                if domain_col:
                    for row in reader:
                        domain = row[domain_col].strip()
                        # Clean up domain (remove protocol if present)
                        if '://' in domain:
                            domain = domain.split('://')[-1]
                        if '/' in domain:
                            domain = domain.split('/')[0]
                        if domain:
                            yield domain.lower()
            else:
                # No header, assume first column is domain
                f.seek(0)
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        domain = row[0].strip()
                        # Clean up domain
                        if '://' in domain:
                            domain = domain.split('://')[-1]
                        if '/' in domain:
                            domain = domain.split('/')[0]
                        if domain:
                            yield domain.lower()
    
    def to_text_file(self, output_path: Union[str, Path]):
        """Convert Majestic Million to text file format"""
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            for domain in self.iter_domains():
                f.write(f"{domain}\n")
        
        print(f"Converted Majestic Million to {output_path}")
        return output_path


class CombinedDomainDataset:
    """Combine multiple domain sources for training"""
    
    def __init__(self, sources: List[Union[str, Path]]):
        """
        Args:
            sources: List of file paths (can be .txt, .csv, .xz files)
        """
        self.sources = [Path(s) for s in sources]
        
    def iter_domains(self) -> Iterator[str]:
        """Iterate through all domains from all sources"""
        for source in self.sources:
            if source.suffix == '.csv':
                # Handle CSV files (like Majestic Million)
                loader = MajesticMillionLoader(source)
                yield from loader.iter_domains()
                
            elif source.suffix == '.xz':
                # Handle compressed files
                import subprocess
                proc = subprocess.Popen(
                    ['xzcat', str(source)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                for line in proc.stdout:
                    domain = line.strip().lower()
                    if domain:
                        yield domain
                        
            else:
                # Handle regular text files
                with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        domain = line.strip().lower()
                        if domain:
                            yield domain


def prepare_majestic_for_training(
    majestic_path: Union[str, Path] = "data/raw/majestic_million.csv",
    output_path: Union[str, Path] = "data/processed/domains/majestic_million.txt"
) -> Path:
    """
    Prepare Majestic Million for training by converting to text format
    
    Returns:
        Path to the prepared text file
    """
    majestic_path = Path(majestic_path)
    output_path = Path(output_path)
    
    if output_path.exists():
        print(f"Majestic Million already prepared at {output_path}")
        return output_path
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to text format
    loader = MajesticMillionLoader(majestic_path)
    loader.to_text_file(output_path)
    
    return output_path


if __name__ == "__main__":
    # Test Majestic Million loading
    majestic_file = Path("data/raw/majestic_million.csv")
    
    if majestic_file.exists():
        print("Testing Majestic Million loader...")
        loader = MajesticMillionLoader(majestic_file)
        
        # Print first 10 domains
        for i, domain in enumerate(loader.iter_domains()):
            print(f"{i+1}: {domain}")
            if i >= 9:
                break
        
        # Count total domains
        total = sum(1 for _ in loader.iter_domains())
        print(f"\nTotal domains in Majestic Million: {total:,}")