# WikiText-103 Training Guide

## Quick Start

### 1. Setup Environment
```bash
cd /home/imsarthakshrma/Projects/RIIK

# Install required packages
./venv/bin/pip install datasets transformers tokenizers
```

### 2. Download Data (Optional - scripts will auto-download)
```bash
# WikiText-103 will be downloaded automatically by HuggingFace datasets
# Or manually:
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```

### 3. Run Experiments

**Note**: Each experiment takes 2-4 hours on GPU. Run them sequentially or in parallel if you have multiple GPUs.

#### Experiment 1: Baseline GPT
```bash
./venv/bin/python experiments/wikitext/train_baseline_gpt.py
```

#### Experiment 2: Hierarchical (2-layer)
```bash
./venv/bin/python experiments/wikitext/train_hierarchical.py
```

#### Experiment 3: Kolosis V2 Minimal
```bash
./venv/bin/python experiments/wikitext/train_kolosis_v2_minimal.py
```

### 4. Compare Results
```bash
./venv/bin/python experiments/wikitext/compare_results.py
```

---

## What to Expect

### Training Progress
Each script will print:
- Epoch progress (1-10)
- Batch progress (every 100 batches)
- Train loss, val loss, perplexity
- Checkpoints saved every epoch

### Expected Results

| Model | Perplexity | Training Time |
|-------|------------|---------------|
| Baseline GPT | 40-50 | 2-4h |
| Hierarchical (2-layer) | 35-45 | 2-4h |
| Kolosis V2 Minimal | 30-40 | 3-5h |

**Lower perplexity = better**

### Files Created
```
experiments/wikitext_results/
├── baseline_gpt_results.json
├── baseline_gpt_best.pt
├── hierarchical_results.json
├── hierarchical_best.pt
├── kolosis_v2_minimal_results.json
├── kolosis_v2_minimal_best.pt
└── comparison_report.md
```

---

## Monitoring

### Check Progress
```bash
# Watch training logs
tail -f experiments/wikitext_results/training.log

# Check GPU usage
nvidia-smi -l 1
```

### Resume Training
If training is interrupted, checkpoints are saved every epoch. You can resume by modifying the script to load from checkpoint.

---

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size in the script:
```python
config['batch_size'] = 16  # or 8
```

### Slow Training
- Ensure you're using GPU (check `device` output)
- Reduce model size if needed:
  ```python
  config['n_layer'] = 4  # instead of 6
  config['n_embd'] = 128  # instead of 256
  ```

### Dataset Download Issues
If automatic download fails:
```bash
# Manual download
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```

---

## Important Notes

⚠️ **GPU Required**: WikiText-103 is too large for CPU training

⚠️ **Time Commitment**: Each experiment takes 2-4 hours

⚠️ **Disk Space**: ~2GB for dataset + checkpoints

✅ **Automatic Checkpointing**: Training can be interrupted and resumed

✅ **Progress Logging**: All output saved to log files

---

## After Training

### Analyze Results
The comparison script will generate:
1. Performance table (perplexity comparison)
2. Loss curves (training progress visualization)
3. Statistical tests (significance of improvements)
4. Recommendations (which model to use)

### Expected Findings
Based on TinyStories results:
- Hierarchical should beat baseline (+10-15%)
- V2 Minimal should beat hierarchical (+20-30%)
- Confirms cognitive mechanisms work at scale

---

## Quick Test (Optional)

Test on WikiText-2 (smaller dataset, ~10 minutes):
```bash
./venv/bin/python experiments/wikitext/train_baseline_gpt.py --dataset wikitext-2 --epochs 3
```

This validates your setup before running the full experiments.

---

## Status

**Scripts Created**:
- ✅ `experiments/wikitext/train_baseline_gpt.py` (template - needs tokenization)
- ⏳ `experiments/wikitext/train_hierarchical.py` (to create)
- ⏳ `experiments/wikitext/train_kolosis_v2_minimal.py` (to create)
- ⏳ `experiments/wikitext/compare_results.py` (to create)

**Next**: I'll create the remaining scripts with proper tokenization.
