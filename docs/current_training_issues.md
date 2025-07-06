# DomainBERT Training Issues - July 6, 2025

## Current Status
- **Training Step**: 35,000 / 862,318 (4.1% complete)
- **Training Loss**: 0.037 (appears good)
- **Test Accuracy**: 3.2% (barely above 2.3% random baseline)
- **Perplexity**: 663M (should be <20)

## Critical Issue
Despite low training loss, the model is NOT learning meaningful character patterns. It can only predict obvious tokens like '.' (dots) with 100% accuracy, but fails completely on character reconstruction.

## Root Cause Analysis

### 1. **Vocabulary Mixing Problem**
- Total vocabulary: 555 tokens (43 characters + 512 TLDs)
- Model doesn't distinguish between character tokens and TLD tokens
- When predicting masked characters, it considers all 555 tokens as valid options
- Example: When masking 'g' in "google.com", the model might predict "com" (a TLD token)

### 2. **Current Tokenization**
```
Domain: google.com
Tokens: ['g', 'o', 'o', 'g', 'l', 'e', '.', 'com']
         ↑---- character tokens ----↑  ↑   ↑TLD token
```

### 3. **Masking Strategy**
- 30% of non-TLD tokens masked (very aggressive)
- Model mostly predicts tokens: '4' ([UNK]), '-', '2', '1'
- Suggests model learned to predict common tokens rather than contextual reconstruction

### 4. **Architecture Issue**
```python
# Current: Single MLM head predicting from all 555 tokens
self.mlm_predictions = nn.Linear(hidden_size, vocab_size=555)

# Problem: Character positions shouldn't predict TLD tokens
# TLD positions shouldn't predict character tokens
```

## Test Results Summary
```
Input: g[MASK]ogle.com
Expected: 'o'
Model predicts: '-' (100%), '4' (25%), '2' (17%)

Input: [MASK].com  
Expected: any valid domain name character
Model predicts: '4' ([UNK]) consistently
```

## Proposed Solutions

### Option 1: Separate Prediction Heads
- Character MLM head: Predicts only from 43 character tokens
- TLD MLM head: Predicts only from 512 TLD tokens
- Use position information to determine which head to use

### Option 2: Masked Softmax
- During loss calculation, mask out invalid tokens based on position
- Character positions: only compute loss over character vocabulary
- TLD positions: only compute loss over TLD vocabulary

### Option 3: Reduce Masking Rate
- Current 30% is too aggressive for short sequences
- Try 15% to give model more context

## Key Metrics to Track
- Character-level accuracy (currently 3.2%)
- Perplexity on held-out data (currently 663M)
- Distribution of predicted tokens (currently skewed to '4', '-', '2')

## Training Configuration
- Batch size: 1024
- Learning rate: 1e-4
- Gradient accumulation: 2
- Effective batch size: 2048
- FP16 training on A100 GPU
- 30% masking rate (likely too high)

## Next Steps
1. Implement vocabulary constraints in model
2. Reduce masking rate to 15-20%
3. Monitor character prediction accuracy specifically
4. Consider curriculum learning (start with easier examples)