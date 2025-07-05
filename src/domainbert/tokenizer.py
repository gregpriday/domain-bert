"""Fast domain tokenizer with structural awareness"""

import json
import os
import torch
from typing import Optional, Tuple, Union, Dict, List, Any
from collections import Counter

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast, BatchEncoding

from .domain_parser import DomainParser, extract as extract_domain


class DomainPreTokenizer:
    """Custom pre-tokenizer that preserves domain structure"""
    
    def __init__(self):
        self.parser = DomainParser()
    
    def pre_tokenize(self, pretok):
        """Split domain into structured components with labels"""
        text = pretok.text
        
        # Extract domain components
        extracted = self.parser.extract(text.lower())
        
        # Build splits with component labels
        splits = []
        offset = 0
        
        # Add subdomain if present
        if extracted.subdomain:
            subdomain_chars = extracted.subdomain
            for i, char in enumerate(subdomain_chars):
                if char == '.':
                    # Dot within subdomain is a separator
                    splits.append((char, (offset, offset + 1), (3, 0)))  # token_type=3 for separator
                else:
                    splits.append((char, (offset, offset + 1), (1, i)))  # token_type=1 for subdomain
                offset += 1
            
            # Add dot separator between subdomain and domain
            if offset < len(text) and text[offset] == '.':
                splits.append(('.', (offset, offset + 1), (3, 0)))  # token_type=3 for separator
                offset += 1
        
        # Add main domain
        domain_start = offset
        for i, char in enumerate(extracted.domain):
            splits.append((char, (offset, offset + 1), (0, i)))  # token_type=0 for main domain
            offset += 1
        
        # Add TLD if present
        if extracted.suffix and offset < len(text):
            # Add dot separator
            if text[offset] == '.':
                splits.append(('.', (offset, offset + 1), (3, 0)))
                offset += 1
            
            # Add entire TLD as a single token
            suffix_start = offset
            suffix_end = offset + len(extracted.suffix)
            # Check if TLD is in vocabulary, otherwise use [UNK_TLD]
            tld_token = extracted.suffix
            # Check if this TLD exists in the tokenizer's vocabulary
            # If not, it will be handled by the tokenizer's UNK mechanism
            splits.append((tld_token, (suffix_start, suffix_end), (2, 0)))  # token_type=2 for TLD
            offset = suffix_end
        
        return splits
    
    def pre_tokenize_str(self, text):
        """Interface for tokenizers library - returns list of (token, token_type) tuples"""
        class PreTok:
            def __init__(self, text):
                self.text = text
        
        pretok = PreTok(text)
        splits = self.pre_tokenize(pretok)
        
        # Convert to expected format: list of (token, token_type) tuples
        result = []
        for token, span, (token_type, _) in splits:
            result.append((token, token_type))
        
        return result


class DomainBertTokenizerFast(PreTrainedTokenizerFast):
    """Fast domain tokenizer with structural token typing"""
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load tokenizer from pretrained model"""
        import os
        from pathlib import Path
        
        # Check if tld_vocab.json exists in the directory
        path = Path(pretrained_model_name_or_path)
        tld_vocab_file = path / "tld_vocab.json"
        if tld_vocab_file.exists():
            kwargs['tld_vocab_file'] = str(tld_vocab_file)
        
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
    def __init__(
        self,
        vocab_file=None,
        tld_vocab_file=None,
        max_length=64,
        model_max_length=64,
        clean_up_tokenization_spaces=True,
        **kwargs
    ):
        # Initialize domain parser first
        self.parser = DomainParser()
        
        # First load TLD vocabulary so it can be included in base tokenizer
        self.tld_to_id = {"[PAD_TLD]": 0, "[UNK_TLD]": 1}
        self.id_to_tld = {0: "[PAD_TLD]", 1: "[UNK_TLD]"}
        
        if tld_vocab_file:
            with open(tld_vocab_file, 'r') as f:
                tld_vocab = json.load(f)
                # Handle both formats: nested dict with 'tld_to_id' key or flat dict
                if isinstance(tld_vocab, dict):
                    if 'tld_to_id' in tld_vocab:
                        self.tld_to_id = tld_vocab['tld_to_id']
                    else:
                        # Flat dictionary format
                        self.tld_to_id = tld_vocab
                    
                    # Convert string keys to int values if needed
                    self.tld_to_id = {k: int(v) if isinstance(v, str) else v for k, v in self.tld_to_id.items()}
                    self.id_to_tld = {v: k for k, v in self.tld_to_id.items()}
        
        # Build the base tokenizer (now includes TLDs)
        if vocab_file:
            tokenizer = Tokenizer.from_file(vocab_file)
        else:
            # Create tokenizer with characters and TLDs
            tokenizer = self._create_base_tokenizer()
        
        # Set up post-processing with special tokens
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 2),
                ("[SEP]", 3),
            ],
        )
        
        # Initialize parent class
        # Remove any duplicate special token arguments from kwargs
        for key in ['pad_token', 'mask_token', 'cls_token', 'sep_token', 'unk_token']:
            kwargs.pop(key, None)
            
        super().__init__(
            tokenizer_object=tokenizer,
            model_max_length=model_max_length,
            pad_token="[PAD]",
            mask_token="[MASK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            unk_token="[UNK]",
            **kwargs
        )
        
        # Domain pre-tokenizer for structure extraction
        self.domain_pretokenizer = DomainPreTokenizer()
        
        # Store max length
        self._max_length = max_length
    
    def _create_base_tokenizer(self):
        """Create the base tokenizer with characters and TLDs"""
        # Character vocabulary
        vocab = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[UNK]": 4,
            "[UNK_TLD]": 5  # Unknown TLD token
        }
        
        # Add only valid domain name characters
        # Letters a-z (domains are case-insensitive, we use lowercase)
        for c in 'abcdefghijklmnopqrstuvwxyz':
            vocab[c] = len(vocab)
        
        # Digits 0-9
        for c in '0123456789':
            vocab[c] = len(vocab)
        
        # Hyphen and period (valid in domains)
        vocab['-'] = len(vocab)
        vocab['.'] = len(vocab)
        
        # Add TLDs from vocabulary if available
        if hasattr(self, 'tld_to_id') and self.tld_to_id:
            for tld, tld_id in self.tld_to_id.items():
                if tld not in ["[PAD_TLD]", "[UNK_TLD]", "<PAD>", "<UNK>"]:
                    vocab[tld] = len(vocab)
        
        # Create WordLevel model for exact token matching
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        
        # Use simple whitespace pre-tokenizer (we'll handle domain splitting in encode)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # No special decoder needed for WordLevel
        tokenizer.decoder = None
        
        return tokenizer
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Override tokenization to handle domain structure with TLDs as single tokens"""
        # Parse domain structure
        extracted = self.parser.extract(text.lower())
        tokens = []
        
        # Add subdomain tokens (character by character)
        if extracted.subdomain:
            for char in extracted.subdomain:
                if char in self.vocab:
                    tokens.append(char)
                else:
                    tokens.append('[UNK]')
            tokens.append('.')  # separator
        
        # Add main domain tokens (character by character)
        for char in extracted.domain:
            if char in self.vocab:
                tokens.append(char)
            else:
                tokens.append('[UNK]')
        
        # Add TLD as a single token
        if extracted.suffix:
            tokens.append('.')  # separator
            # Check if TLD is in vocabulary
            if extracted.suffix in self.vocab:
                tokens.append(extracted.suffix)
            else:
                # For unknown TLDs, use [UNK_TLD]
                tokens.append('[UNK_TLD]')
        
        return tokens
    
    def _encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding_strategy=None,
        truncation_strategy=None,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs
    ):
        """Override to use our custom tokenization"""
        # First tokenize using our custom method
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # Add special tokens if requested
        if add_special_tokens:
            input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        
        # Create encoding
        encoding = {"input_ids": input_ids}
        
        # Add attention mask
        if return_attention_mask is not False:
            encoding["attention_mask"] = [1] * len(input_ids)
        
        # Handle padding/truncation
        if max_length is not None:
            if len(input_ids) > max_length:
                # Truncate
                for key in encoding:
                    encoding[key] = encoding[key][:max_length]
            elif padding_strategy == "max_length":
                # Pad
                pad_length = max_length - len(input_ids)
                encoding["input_ids"] = encoding["input_ids"] + [self.pad_token_id] * pad_length
                if "attention_mask" in encoding:
                    encoding["attention_mask"] = encoding["attention_mask"] + [0] * pad_length
        
        return encoding
    
    def __call__(
        self,
        text,
        text_pair=None,
        text_target=None,
        text_pair_target=None,
        add_special_tokens: bool = True,
        padding=True,
        truncation=None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """Override __call__ to add structural information"""
        # Get base encoding first
        encoding = super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length or self._max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=True,  # Always return for override
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs
        )
        
        # Add structural information
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        # Extract TLD IDs and structural token types
        tld_ids = []
        structural_token_types = []
        
        for domain in texts:
            if isinstance(domain, str):
                # Extract components
                extracted = self.parser.extract(domain.lower())
                
                # Get TLD ID
                tld = extracted.suffix or 'unknown'
                # Try to get unknown TLD ID - check both possible keys
                unk_tld_id = self.tld_to_id.get('[UNK_TLD]', self.tld_to_id.get('<UNK>', 1))
                tld_id = self.tld_to_id.get(tld, unk_tld_id)
                tld_ids.append(tld_id)
                
                # Create structural token types
                token_types = self._create_token_type_ids(domain, extracted)
                structural_token_types.append(token_types)
        
        # Override token_type_ids with structural information
        if return_token_type_ids is not False and structural_token_types:
            if isinstance(text, str):
                # Single encoding
                if return_tensors == "pt":
                    max_len = encoding['input_ids'].shape[-1]
                else:
                    max_len = len(encoding['input_ids'])
                types = structural_token_types[0]
                
                # Pad or truncate to match encoded length
                if len(types) < max_len:
                    types = types + [0] * (max_len - len(types))
                else:
                    types = types[:max_len]
                
                if return_tensors == "pt":
                    encoding['token_type_ids'] = torch.tensor([types])
                else:
                    encoding['token_type_ids'] = types
            else:
                # Batch encoding
                if return_tensors == "pt":
                    batch_size = encoding['input_ids'].shape[0]
                    max_len = encoding['input_ids'].shape[1]
                else:
                    batch_size = len(encoding['input_ids'])
                    max_len = max(len(ids) for ids in encoding['input_ids'])
                
                # Create properly padded token type IDs
                final_token_types = []
                for i in range(batch_size):
                    types = structural_token_types[i] if i < len(structural_token_types) else [0] * max_len
                    
                    # Pad or truncate to match encoded length
                    if len(types) < max_len:
                        types = types + [0] * (max_len - len(types))
                    else:
                        types = types[:max_len]
                    
                    final_token_types.append(types)
                
                if return_tensors == "pt":
                    encoding['token_type_ids'] = torch.tensor(final_token_types)
                else:
                    encoding['token_type_ids'] = final_token_types
        
        # Add TLD IDs
        if tld_ids:
            if return_tensors == "pt":
                encoding['tld_ids'] = torch.tensor(tld_ids)
            else:
                encoding['tld_ids'] = tld_ids
        
        return encoding
    
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens: bool = True,
        padding_strategy = None,
        truncation_strategy = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """Override to handle structural token types"""
        # Call parent's implementation with proper parameters
        encoding = super()._batch_encode_plus(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length or self._max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs
        )
        
        return encoding
    
    def _create_token_type_ids(self, domain: str, extracted) -> List[int]:
        """Create structural token type IDs for a domain"""
        types = []
        
        # CLS token
        types.append(0)
        
        # Process subdomain
        if extracted.subdomain:
            types.extend([1] * len(extracted.subdomain))  # Type 1 for subdomain
            types.append(3)  # Type 3 for dot separator
        
        # Process main domain
        types.extend([0] * len(extracted.domain))  # Type 0 for main domain
        
        # Process TLD
        if extracted.suffix:
            types.append(3)  # Type 3 for dot separator
            types.extend([2] * len(extracted.suffix))  # Type 2 for TLD
        
        # SEP token
        types.append(0)
        
        return types
    
    def build_tld_vocabulary(self, domains: List[str], min_count: int = 10):
        """Build TLD vocabulary from domain list"""
        tld_counter = Counter()
        
        for domain in domains:
            extracted = self.parser.extract(domain.lower())
            tld = extracted.suffix or 'unknown'
            tld_counter[tld] += 1
        
        # Add frequent TLDs to vocabulary
        for tld, count in tld_counter.most_common():
            if count >= min_count and tld not in self.tld_to_id:
                idx = len(self.tld_to_id)
                self.tld_to_id[tld] = idx
                self.id_to_tld[idx] = tld
        
        print(f"Built TLD vocabulary with {len(self.tld_to_id)} TLDs")
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        """Save both tokenizer and TLD vocabulary"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save TLD vocabulary (this is the main vocabulary we care about)
        tld_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "tld_vocab.json"
        )
        
        with open(tld_vocab_file, 'w') as f:
            json.dump(self.tld_to_id, f, indent=2)
        
        # For compatibility, return as tuple
        return (tld_vocab_file,)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save the tokenizer configuration and vocabulary"""
        # Save the base tokenizer
        files = super().save_pretrained(save_directory, **kwargs)
        
        # Also save TLD vocabulary
        self.save_vocabulary(save_directory)
        
        return files