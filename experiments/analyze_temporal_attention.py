"""
Analyze Multi-Scale Temporal Attention
Investigate how fast, medium, and slow decay contribute.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from neural_networks.kolosis.temporal_attention import MultiScaleMultiHeadAttention
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import download_tinystories, TinyStoriesDataset

def analyze_learned_decays(model, device='cuda'):
    """Analyze what decay rates the model learned"""
    print("\n" + "="*60)
    print("LEARNED DECAY RATES")
    print("="*60)
    
    # Extract decay parameters from first attention head
    first_head = model.blocks[0].sa.heads[0]
    
    # Compute actual decay rates
    gamma_fast = torch.sigmoid(first_head.gamma_fast_logit) * 0.3 + 0.6
    gamma_medium = torch.sigmoid(first_head.gamma_medium_logit) * 0.15 + 0.85
    gamma_slow = torch.sigmoid(first_head.gamma_slow_logit) * 0.05 + 0.95
    
    # Compute alpha (mixing weights)
    alpha = F.softmax(first_head.alpha_logits, dim=0)
    
    print(f"\nDecay Rates:")
    print(f"  Fast (γ_f):   {gamma_fast.item():.4f} (weight: {alpha[0].item():.4f})")
    print(f"  Medium (γ_m): {gamma_medium.item():.4f} (weight: {alpha[1].item():.4f})")
    print(f"  Slow (γ_s):   {gamma_slow.item():.4f} (weight: {alpha[2].item():.4f})")
    
    # Compute effective memory span
    def memory_span(gamma, threshold=0.1):
        """How many steps until decay drops below threshold"""
        return int(np.log(threshold) / np.log(gamma.item()))
    
    print(f"\nEffective Memory Span (10% threshold):")
    print(f"  Fast:   {memory_span(gamma_fast)} tokens")
    print(f"  Medium: {memory_span(gamma_medium)} tokens")
    print(f"  Slow:   {memory_span(gamma_slow)} tokens")
    
    return {
        'gamma_fast': gamma_fast.item(),
        'gamma_medium': gamma_medium.item(),
        'gamma_slow': gamma_slow.item(),
        'alpha_fast': alpha[0].item(),
        'alpha_medium': alpha[1].item(),
        'alpha_slow': alpha[2].item()
    }

def visualize_temporal_bias(model, seq_len=32, device='cuda'):
    """Visualize the temporal bias matrix"""
    print("\n" + "="*60)
    print("TEMPORAL BIAS VISUALIZATION")
    print("="*60)
    
    first_head = model.blocks[0].sa.heads[0]
    
    # Compute temporal bias
    temporal_bias = first_head.compute_temporal_bias(seq_len, device)
    temporal_bias_np = temporal_bias.detach().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Heatmap
    im = axes[0].imshow(temporal_bias_np, cmap='viridis', aspect='auto')
    axes[0].set_title('Temporal Bias Matrix')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im, ax=axes[0])
    
    # Decay curves
    for i in range(seq_len):
        axes[1].plot(temporal_bias_np[i, :i+1], alpha=0.3, color='blue')
    
    # Plot average decay
    avg_decay = np.mean([temporal_bias_np[i, :i+1] for i in range(1, seq_len)], axis=0)
    axes[1].plot(avg_decay, color='red', linewidth=2, label='Average Decay')
    
    axes[1].set_title('Temporal Decay Curves')
    axes[1].set_xlabel('Distance (tokens)')
    axes[1].set_ylabel('Bias Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('experiments/temporal_analysis', exist_ok=True)
    plt.savefig('experiments/temporal_analysis/temporal_bias.png', dpi=150)
    print("✅ Saved visualization to experiments/temporal_analysis/temporal_bias.png")
    
    plt.close()

def test_temporal_contribution(config, train_loader, val_loader, device='cuda'):
    """Test if temporal attention actually helps"""
    print("\n" + "="*60)
    print("TEMPORAL ATTENTION CONTRIBUTION TEST")
    print("="*60)
    
    from experiments.hybrid_models import GPT_TemporalAttention
    from neural_networks.autograd.transformer_torch import GPT as BaselineGPT
    
    # Train baseline
    print("\n### Baseline GPT (no temporal) ###")
    baseline = BaselineGPT(**config).to(device)
    optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.003)
    
    for epoch in range(100):
        baseline.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = baseline(x, y)
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            baseline.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, loss = baseline(x, y)
                    val_loss += loss.item()
            print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}")
    
    baseline.eval()
    baseline_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = baseline(x, y)
            baseline_val_loss += loss.item()
    baseline_val_loss /= len(val_loader)
    
    # Train temporal
    print("\n### GPT + Temporal Attention ###")
    temporal = GPT_TemporalAttention(**config).to(device)
    optimizer = torch.optim.AdamW(temporal.parameters(), lr=0.003)
    
    for epoch in range(100):
        temporal.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = temporal(x, y)
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            temporal.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, loss = temporal(x, y)
                    val_loss += loss.item()
            print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}")
    
    temporal.eval()
    temporal_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = temporal(x, y)
            temporal_val_loss += loss.item()
    temporal_val_loss /= len(val_loader)
    
    # Analyze learned parameters
    decay_params = analyze_learned_decays(temporal, device)
    
    # Visualize
    visualize_temporal_bias(temporal, seq_len=32, device=device)
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    improvement = (baseline_val_loss - temporal_val_loss) / baseline_val_loss * 100
    
    print(f"\nBaseline GPT:      {baseline_val_loss:.4f}")
    print(f"Temporal Attention: {temporal_val_loss:.4f}")
    print(f"Improvement:        {improvement:+.1f}%")
    
    print(f"\nDecay Contributions:")
    print(f"  Fast (γ={decay_params['gamma_fast']:.3f}):   {decay_params['alpha_fast']*100:.1f}%")
    print(f"  Medium (γ={decay_params['gamma_medium']:.3f}): {decay_params['alpha_medium']*100:.1f}%")
    print(f"  Slow (γ={decay_params['gamma_slow']:.3f}):   {decay_params['alpha_slow']*100:.1f}%")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if improvement > 5:
        print("✅ Temporal attention provides significant benefit!")
    elif improvement > 0:
        print("⚠️  Temporal attention helps slightly")
    else:
        print("❌ Temporal attention doesn't help at this scale")
    
    # Check if all scales are used
    if decay_params['alpha_fast'] > 0.2 and decay_params['alpha_medium'] > 0.2 and decay_params['alpha_slow'] > 0.2:
        print("✅ All three temporal scales are being used")
    else:
        print("⚠️  Some temporal scales are underutilized:")
        if decay_params['alpha_fast'] < 0.2:
            print("   - Fast decay is ignored")
        if decay_params['alpha_medium'] < 0.2:
            print("   - Medium decay is ignored")
        if decay_params['alpha_slow'] < 0.2:
            print("   - Slow decay is ignored")
    
    return {
        'baseline_loss': baseline_val_loss,
        'temporal_loss': temporal_val_loss,
        'improvement': improvement,
        'decay_params': decay_params
    }

def main():
    config = {
        'vocab_size': None,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 2,
        'block_size': 32,
        'dropout': 0.1
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    json_path, _ = download_tinystories()
    
    with open(json_path, 'r') as f:
        import json
        data = json.load(f)
        text = ' '.join(data['stories'])
    
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    config['vocab_size'] = tokenizer.vocab_size
    
    dataset = TinyStoriesDataset(json_path, tokenizer, block_size=config['block_size'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    print("="*60)
    print("MULTI-SCALE TEMPORAL ATTENTION ANALYSIS")
    print("="*60)
    
    results = test_temporal_contribution(config, train_loader, val_loader, device)
    
    # Save results
    os.makedirs('experiments/temporal_analysis', exist_ok=True)
    with open('experiments/temporal_analysis/results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print("\n✅ Analysis complete! Results saved to experiments/temporal_analysis/")

if __name__ == "__main__":
    main()
