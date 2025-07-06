#!/usr/bin/env python3
"""Initialize and save the DomainBERT tokenizer"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from domainbert.tokenizer import DomainBertTokenizerFast

# Create tokenizer with TLD vocabulary
print("Creating DomainBERT tokenizer...")
tld_vocab_file = "/home/ubuntu/domain-bert/src/domainbert/data/tld_vocab.json"
tokenizer = DomainBertTokenizerFast(tld_vocab_file=tld_vocab_file)

# Save to models directory
output_dir = Path("/home/ubuntu/domain-bert/models/tokenizer")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving tokenizer to {output_dir}")
tokenizer.save_pretrained(str(output_dir))

print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print("Tokenizer saved successfully!")

# Test tokenization
test_domains = ["google.com", "github.io", "amazon.co.uk"]
print("\nTest tokenizations:")
for domain in test_domains:
    tokens = tokenizer.tokenize(domain)
    print(f"  {domain}: {tokens}")