"""Fast domain tokenizer with structural awareness"""

import json
import os
import tldextract
import torch
from typing import Optional, Tuple, Union, Dict, List, Any
from collections import Counter

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast, BatchEncoding


class DomainPreTokenizer:
    """Custom pre-tokenizer that preserves domain structure"""
    
    def __init__(self):
        self.extractor = tldextract.extract
    
    def pre_tokenize(self, pretok):
        """Split domain into structured components with labels"""
        text = pretok.text
        
        # Extract domain components
        extracted = self.extractor(text.lower())
        
        # Build splits with component labels
        splits = []
        offset = 0
        
        # Add subdomain if present
        if extracted.subdomain:
            for i, char in enumerate(extracted.subdomain):
                splits.append((char, (offset, offset + 1), (1, i)))  # token_type=1 for subdomain
                offset += 1
            
            # Add dot separator
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
            
            # Add TLD characters
            for i, char in enumerate(extracted.suffix):
                if offset < len(text):
                    splits.append((char, (offset, offset + 1), (2, i)))  # token_type=2 for TLD
                    offset += 1
        
        return splits
    
    def pre_tokenize_str(self, text):
        """Interface for tokenizers library"""
        class PreTok:
            def __init__(self, text):
                self.text = text
        
        pretok = PreTok(text)
        return self.pre_tokenize(pretok)


class DomainBertTokenizerFast(PreTrainedTokenizerFast):
    """Fast domain tokenizer with structural token typing"""
    
    def __init__(
        self,
        vocab_file=None,
        tld_vocab_file=None,
        max_length=64,
        model_max_length=64,
        clean_up_tokenization_spaces=True,
        **kwargs
    ):
        # Build the base tokenizer
        if vocab_file:
            tokenizer = Tokenizer.from_file(vocab_file)
        else:
            # Create character-level BPE tokenizer
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
        
        # TLD vocabulary
        self.tld_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_tld = {0: "<PAD>", 1: "<UNK>"}
        
        if tld_vocab_file:
            with open(tld_vocab_file, 'r') as f:
                tld_vocab = json.load(f)
                self.tld_to_id = tld_vocab.get('tld_to_id', self.tld_to_id)
                self.id_to_tld = {v: k for k, v in self.tld_to_id.items()}
        
        # Domain pre-tokenizer for structure extraction
        self.domain_pretokenizer = DomainPreTokenizer()
        
        # Store max length
        self._max_length = max_length
    
    def _create_base_tokenizer(self):
        """Create the base character-level tokenizer"""
        # Character vocabulary
        vocab = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[UNK]": 4
        }
        
        # Add printable ASCII characters
        for i in range(32, 127):
            vocab[chr(i)] = len(vocab)
        
        # Create BPE model
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
        
        # Use character-level pre-tokenization
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Set decoder
        tokenizer.decoder = decoders.BPEDecoder()
        
        return tokenizer
    
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
                extracted = tldextract.extract(domain.lower())
                
                # Get TLD ID
                tld = extracted.suffix or 'unknown'
                tld_id = self.tld_to_id.get(tld, self.tld_to_id['<UNK>'])
                tld_ids.append(tld_id)
                
                # Create structural token types
                token_types = self._create_token_type_ids(domain, extracted)
                structural_token_types.append(token_types)
        
        # Override token_type_ids with structural information
        if return_token_type_ids is not False and structural_token_types:
            if isinstance(text, str):
                # Single encoding
                max_len = len(encoding['input_ids'])
                types = structural_token_types[0]
                
                # Pad or truncate to match encoded length
                if len(types) < max_len:
                    types = types + [0] * (max_len - len(types))
                else:
                    types = types[:max_len]
                
                encoding['token_type_ids'] = types
            else:
                # Batch encoding
                batch_size = len(encoding['input_ids'])
                max_len = len(encoding['input_ids'][0]) if return_tensors else max(len(ids) for ids in encoding['input_ids'])
                
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
            extracted = tldextract.extract(domain.lower())
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
            json.dump({
                'tld_to_id': self.tld_to_id,
                'version': '1.0'
            }, f, indent=2)
        
        # For compatibility, return as tuple
        return (tld_vocab_file,)