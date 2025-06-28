#!/usr/bin/env python3
"""
Extract TLDs from the Mozilla Public Suffix List and create a TLD vocabulary file.
"""
import json
import urllib.request
from pathlib import Path
import re


def download_public_suffix_list(url="https://publicsuffix.org/list/public_suffix_list.dat"):
    """Download the public suffix list."""
    print(f"Downloading public suffix list from {url}...")
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')
    return content


def extract_tlds(content, max_levels=2):
    """Extract TLDs from the public suffix list content.
    
    Args:
        content: The public suffix list content
        max_levels: Maximum number of dot-separated levels to include (default: 2)
                   1 = only TLDs like .com, .org
                   2 = TLDs + second-level like .co.uk, .ac.jp
    """
    tlds = set()
    
    for line in content.splitlines():
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('//'):
            continue
            
        # Remove wildcard and exception markers
        if line.startswith('*.'):
            line = line[2:]
        elif line.startswith('!'):
            continue
            
        # Only keep valid domain-like entries
        if line and not line.startswith('.') and re.match(r'^[a-z0-9.-]+$', line, re.IGNORECASE):
            # Count the number of dots (levels)
            num_dots = line.count('.')
            if num_dots < max_levels:
                tlds.add(line.lower())
    
    return sorted(tlds)


def create_tld_vocab(tlds, output_path):
    """Create TLD vocabulary file with special tokens."""
    # Create vocabulary with special tokens at the beginning
    tld_to_id = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    
    # Add all TLDs
    for idx, tld in enumerate(tlds, start=2):
        tld_to_id[tld] = idx
    
    vocab = {
        "tld_to_id": tld_to_id,
        "version": "1.0"
    }
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Created TLD vocabulary with {len(tld_to_id)} entries (including special tokens)")
    print(f"Saved to: {output_path}")
    
    return tld_to_id


def main():
    # Download public suffix list
    content = download_public_suffix_list()
    
    # Extract TLDs
    tlds = extract_tlds(content)
    print(f"Extracted {len(tlds)} unique TLDs")
    
    # Create vocabulary file
    output_path = Path(__file__).parent.parent.parent / "data" / "test_output" / "tld_vocab.json"
    tld_vocab = create_tld_vocab(tlds, output_path)
    
    # Print some statistics
    print(f"\nFirst 10 TLDs: {list(tld_vocab.keys())[2:12]}")
    print(f"Last 10 TLDs: {list(tld_vocab.keys())[-10:]}")


if __name__ == "__main__":
    main()