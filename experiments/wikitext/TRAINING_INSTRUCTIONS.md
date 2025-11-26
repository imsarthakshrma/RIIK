# WikiText-103 Training Instructions

## Setup

```bash
cd /home/imsarthakshrma/Projects/RIIK

# Install required packages
./venv/bin/pip install datasets transformers tokenizers tqdm

# Verify GPU
nvidia-smi
```

---

## Training Commands

### 1. Baseline GPT (2-4 hours)
```bash
./venv/bin/python experiments/wikitext/train_baseline_gpt.py
```

### 2. Hierarchical Embeddings (2-4 hours)
```bash
./venv/bin/python experiments/wikitext/train_hierarchical.py
```

### 3. Kolosis V2 Minimal (3-5 hours)
```bash
# Script ready, run when previous two complete
./venv/bin/python experiments/wikitext/train_kolosis_v2_minimal_wikitext.py
```

---

## What to Expect

### Progress Output
```
Epoch 1/10
  Batch 100/5000 | Loss: 4.2341
  Batch 200/5000 | Loss: 3.9876
  ...
Evaluating: 100%|████████| 500/500
Results:
  Train Loss: 3.8234
  Val Loss: 3.7123
  Perplexity: 40.98
  ✅ Saved best model
```

### Checkpoints
Models saved to `experiments/wikitext_results/`:
- `baseline_gpt_best.pt` - Best baseline model
- `hierarchical_best.pt` - Best hierarchical model
- `*_epoch_N.pt` - Checkpoint every epoch

### Results
JSON files with training curves:
- `baseline_gpt_results.json`
- `hierarchical_results.json`

---

## Monitoring

### Watch Progress
```bash
# In another terminal
watch -n 1 nvidia-smi

# Or tail logs
tail -f nohup.out
```

### Run in Background
```bash
nohup ./venv/bin/python experiments/wikitext/train_baseline_gpt.py > baseline.log 2>&1 &
```

---

## Expected Results

Based on TinyStories findings:

| Model | Expected Perplexity | vs Baseline |
|-------|---------------------|-------------|
| Baseline GPT | 40-50 | - |
| Hierarchical | 35-45 | -10-15% |
| Kolosis V2 Minimal | 30-40 | -20-30% |

**Lower perplexity = better**

---

## Troubleshooting

### Out of Memory
Reduce batch size in the script:
```python
config['batch_size'] = 16  # or 8
```

### Slow Download
Dataset auto-downloads from HuggingFace. If slow:
```bash
# Pre-download
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1')"
```

### Resume Training
If interrupted, modify script to load checkpoint:
```python
checkpoint = torch.load('experiments/wikitext_results/baseline_gpt_epoch_5.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## After Training

Compare results:
```bash
./venv/bin/python experiments/wikitext/compare_results.py
```

This will generate:
- Performance comparison table
- Loss curves visualization
- Statistical significance tests

---

## Key Predictions

### Temporal Attention Will Shine ✨

With 256 token context:
- Fast scale (8 tokens): ✅ Fully utilized
- Medium scale (106 tokens): ✅ Fully utilized
- Slow scale (2064 tokens): ⚠️ Partially utilized

**Expected**: Temporal attention shows +15-20% improvement

### Hierarchical Will Beat Baseline

Proven on TinyStories (+12%), should hold at scale.

### V2 Minimal Will Be Best

Combines hierarchical + semantic streams with direct supervision.

---

## Status

**Scripts Ready**:
- ✅ `train_baseline_gpt.py` (complete with tokenization)
- ✅ `train_hierarchical.py` (complete with tokenization)
- ⏳ `train_kolosis_v2_minimal_wikitext.py` (creating next)

**Ready to train manually!**
