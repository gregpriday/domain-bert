"""Lightweight domain parser for multi-worker support"""

import re
from typing import NamedTuple, Optional, Set
import json
from pathlib import Path


class DomainParts(NamedTuple):
    """Domain components"""
    subdomain: str
    domain: str
    suffix: str


class DomainParser:
    """Fast regex-based domain parser that supports multiprocessing"""
    
    def __init__(self, tld_set: Optional[Set[str]] = None):
        """Initialize parser with optional TLD set for validation"""
        if tld_set is None:
            # Load from our TLD vocabulary
            vocab_path = Path(__file__).parent / "data" / "tld_vocab.json"
            if vocab_path.exists():
                with open(vocab_path) as f:
                    data = json.load(f)
                    tld_dict = data.get("tld_to_id", {})
                    # Extract TLDs, excluding special tokens
                    self.tld_set = {
                        tld for tld in tld_dict.keys() 
                        if not (tld.startswith("<") and tld.endswith(">"))
                    }
            else:
                # Fallback to common TLDs
                self.tld_set = {
                    "com", "net", "org", "edu", "gov", "mil",
                    "co.uk", "ac.uk", "org.uk", "com.au", "co.jp",
                    "io", "app", "dev", "ai", "me", "info", "biz"
                }
        else:
            self.tld_set = tld_set
        
        # Build regex pattern for known TLDs (escape dots)
        escaped_tlds = [re.escape(tld) for tld in sorted(self.tld_set, key=len, reverse=True)]
        tld_pattern = "|".join(escaped_tlds)
        
        # Domain parsing regex - match from right to left for proper TLD handling
        # This ensures compound TLDs like co.uk are matched correctly
        self.domain_regex = re.compile(
            rf'^(.*?)([a-z0-9](?:[a-z0-9\-]{{0,61}}[a-z0-9])?)'     # Capture everything before domain as subdomain
            rf'\.({tld_pattern})$',                                  # Required TLD
            re.IGNORECASE
        )
        
        # Fallback regex for unknown TLDs
        self.fallback_regex = re.compile(
            r'^((?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)*)?'    # Optional subdomain(s)
            r'([a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?)'              # Required domain
            r'\.([a-z]{2,}(?:\.[a-z]{2,})*)$',                      # TLD (2+ letters, possibly compound)
            re.IGNORECASE
        )
    
    def extract(self, domain: str) -> DomainParts:
        """Extract domain parts from a full domain name"""
        domain = domain.lower().strip()
        
        # Try known TLD pattern first
        match = self.domain_regex.match(domain)
        if match:
            # Group 1 contains everything before the main domain (could be empty or subdomains)
            prefix = match.group(1)
            subdomain = prefix.rstrip('.') if prefix and prefix != '' else ''
            main_domain = match.group(2)
            tld = match.group(3)
            return DomainParts(subdomain, main_domain, tld)
        
        # Fallback for unknown TLDs
        match = self.fallback_regex.match(domain)
        if match:
            subdomain = match.group(1).rstrip('.') if match.group(1) else ''
            main_domain = match.group(2)
            tld = match.group(3)
            return DomainParts(subdomain, main_domain, tld)
        
        # If no match, treat entire string as domain with empty subdomain/suffix
        # This handles edge cases like localhost or IP addresses
        return DomainParts('', domain, '')
    
    def __reduce__(self):
        """Support pickling for multiprocessing"""
        return (self.__class__, (self.tld_set,))


# Singleton instance for convenience
_default_parser = None

def get_default_parser() -> DomainParser:
    """Get or create default parser instance"""
    global _default_parser
    if _default_parser is None:
        _default_parser = DomainParser()
    return _default_parser


def extract(domain: str) -> DomainParts:
    """Convenience function using default parser"""
    return get_default_parser().extract(domain)