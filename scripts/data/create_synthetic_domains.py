#!/usr/bin/env python3
"""
Create synthetic domain data for testing DomainBERT
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Set


def generate_domain_names(num_domains: int) -> Set[str]:
    """Generate synthetic domain names."""
    
    # Common domain prefixes
    prefixes = [
        'www', 'mail', 'app', 'blog', 'shop', 'news', 'admin', 'api', 'secure',
        'my', 'web', 'online', 'service', 'cloud', 'data', 'info', 'portal',
        'store', 'site', 'home', 'mobile', 'dev', 'test', 'prod', 'staging'
    ]
    
    # Common words for domain names
    words = [
        'tech', 'global', 'digital', 'smart', 'net', 'pro', 'hub', 'zone',
        'world', 'center', 'point', 'space', 'link', 'connect', 'system',
        'group', 'team', 'corp', 'inc', 'company', 'service', 'solution',
        'market', 'trade', 'business', 'venture', 'startup', 'innovation',
        'creative', 'design', 'studio', 'lab', 'factory', 'workshop'
    ]
    
    # TLDs with approximate distribution
    tlds_weighted = [
        ('com', 70),  # 70% .com
        ('net', 10),  # 10% .net
        ('org', 8),   # 8% .org
        ('io', 3),
        ('co', 2),
        ('info', 2),
        ('biz', 1),
        ('tv', 1),
        ('cc', 1),
        ('me', 1),
        ('xyz', 1)
    ]
    
    # Expand TLDs based on weights
    tlds = []
    for tld, weight in tlds_weighted:
        tlds.extend([tld] * weight)
    
    domains = set()
    
    while len(domains) < num_domains:
        # Different patterns for domain generation
        pattern = random.randint(1, 6)
        
        if pattern == 1:
            # prefix-word.tld
            domain = f"{random.choice(prefixes)}-{random.choice(words)}.{random.choice(tlds)}"
        elif pattern == 2:
            # word-word.tld
            domain = f"{random.choice(words)}-{random.choice(words)}.{random.choice(tlds)}"
        elif pattern == 3:
            # wordnumber.tld
            domain = f"{random.choice(words)}{random.randint(1, 999)}.{random.choice(tlds)}"
        elif pattern == 4:
            # prefix.word.tld (subdomain)
            domain = f"{random.choice(prefixes)}.{random.choice(words)}.{random.choice(tlds)}"
        elif pattern == 5:
            # single word
            domain = f"{random.choice(words)}.{random.choice(tlds)}"
        else:
            # random string (short)
            length = random.randint(3, 8)
            name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=length))
            domain = f"{name}.{random.choice(tlds)}"
        
        domains.add(domain.lower())
    
    return domains


def save_domains_by_country(domains: List[str], output_dir: Path):
    """Save domains to country-specific files (simulating the real dataset structure)."""
    
    # Create a simplified country structure
    countries = {
        'united_states': 0.4,  # 40% of domains
        'germany': 0.1,
        'united_kingdom': 0.1,
        'canada': 0.05,
        'australia': 0.05,
        'france': 0.05,
        'japan': 0.05,
        'brazil': 0.05,
        'india': 0.05,
        'generic': 0.1  # Generic TLDs
    }
    
    # Distribute domains across countries
    domain_list = list(domains)
    random.shuffle(domain_list)
    
    start_idx = 0
    for country, percentage in countries.items():
        country_dir = output_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)
        
        num_country_domains = int(len(domains) * percentage)
        country_domains = domain_list[start_idx:start_idx + num_country_domains]
        start_idx += num_country_domains
        
        # Save to text file (uncompressed for simplicity)
        output_file = country_dir / f"domains-{country}.txt"
        with open(output_file, 'w') as f:
            for domain in sorted(country_domains):
                f.write(f"{domain}\n")
        
        print(f"Created {output_file} with {len(country_domains)} domains")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic domain data for testing")
    parser.add_argument(
        "--num-domains",
        type=int,
        default=1000000,
        help="Number of synthetic domains to generate (default: 1,000,000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/domains_project/data",
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_domains:,} synthetic domains...")
    domains = generate_domain_names(args.num_domains)
    
    print(f"Saving domains to {output_dir}...")
    save_domains_by_country(domains, output_dir)
    
    # Create a combined file for easy access
    all_domains_file = output_dir / "all_domains.txt"
    with open(all_domains_file, 'w') as f:
        for domain in sorted(domains):
            f.write(f"{domain}\n")
    
    print(f"\nCreated synthetic dataset with {len(domains):,} unique domains")
    print(f"Combined file: {all_domains_file}")
    

if __name__ == "__main__":
    main()