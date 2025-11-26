# WikiText-103 Experiment Setup

## Overview
Test Kolosis models on WikiText-103 to validate cognitive mechanisms at proper scale.

## Dataset Stats
- **Tokens**: 103M
- **Vocab**: 267K words
- **Avg sequence length**: 200+ tokens
- **Complexity**: Real Wikipedia text
- **Size**: ~500MB

## Why This Matters
- **Long contexts**: Temporal attention can shine (200+ tokens vs 32)
- **Complex entities**: Concept classification useful
- **Many examples**: Pattern memory can learn
- **Large data**: Prevents overfitting

## Setup Instructions

### 1. Download WikiText-103
```bash
cd /home/imsarthakshrma/Projects/RIIK/data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```

### 2. Install Dependencies (if needed)
```bash
./venv/bin/pip install datasets transformers
```

## Experiments to Run

### Experiment 1: Baseline GPT
**Script**: `experiments/wikitext/train_baseline_gpt.py`

**Config**:
```python
vocab_size: 267735  # WikiText-103 vocab
n_embd: 256
n_head: 8
n_layer: 6
block_size: 256  # Longer context
batch_size: 32
epochs: 10
lr: 0.0003
```

**Expected time**: ~2-4 hours on GPU

**Command**:
```bash
./venv/bin/python experiments/wikitext/train_baseline_gpt.py
```

---

### Experiment 2: Hierarchical-Only
**Script**: `experiments/wikitext/train_hierarchical.py`

**Config**: Same as baseline

**Expected**: Should beat baseline (proven on TinyStories)

**Command**:
```bash
./venv/bin/python experiments/wikitext/train_hierarchical.py
```

---

### Experiment 3: Kolosis V2 Minimal
**Script**: `experiments/wikitext/train_kolosis_v2_minimal.py`

**Config**: Same as baseline

**Hypothesis**: Will beat hierarchical-only at this scale

**Command**:
```bash
./venv/bin/python experiments/wikitext/train_kolosis_v2_minimal.py
```

---

## Monitoring

All scripts will:
- Save checkpoints every epoch
- Log to `experiments/wikitext_results/`
- Track: train loss, val loss, perplexity
- Save final model weights

## Expected Results

| Model | Perplexity | vs Baseline | Training Time |
|-------|------------|-------------|---------------|
| Baseline GPT | ~40-50 | - | 2-4h |
| Hierarchical | ~35-45 | -10-15% | 2-4h |
| Kolosis V2 Minimal | ~30-40 | -20-25% | 3-5h |

**Key Metric**: Perplexity (lower is better)

## Analysis After Training

Run comparison script:
```bash
./venv/bin/python experiments/wikitext/compare_results.py
```

This will generate:
- Performance comparison table
- Loss curves visualization
- Perplexity comparison
- Statistical significance tests

## Notes

- **GPU Required**: WikiText-103 is too large for CPU training
- **Memory**: May need to reduce batch_size if OOM
- **Time**: Each experiment takes 2-5 hours
- **Checkpoints**: Can resume if interrupted

## Quick Test (Optional)

Test on WikiText-2 (smaller) first:
```bash
# Download WikiText-2
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip

# Run quick test (~10 minutes)
./venv/bin/python experiments/wikitext/train_baseline_gpt.py --dataset wikitext-2 --epochs 3
```

## What to Watch For

1. **Perplexity trends**: Should decrease steadily
2. **Hierarchical vs Baseline**: Hierarchical should win
3. **V2 Minimal vs Hierarchical**: This is the key test!
4. **Fusion weights**: Should stabilize around 0.4-0.6

## Success Criteria

✅ Hierarchical beats baseline (validates TinyStories findings)
✅ V2 Minimal beats hierarchical (validates parallel streams at scale)
✅ Perplexity improvements >10% (meaningful gains)
✅ Training stable (no divergence)
