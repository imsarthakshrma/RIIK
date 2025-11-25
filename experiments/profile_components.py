"""
Profile Kolosis components to identify bottlenecks.
"""
import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.kolosis import KolosisTransformer
from neural_networks.autograd.transformer_torch import GPT as BaselineGPT

def profile_model(model, name, batch_size=16, seq_len=32, vocab_size=100, iterations=100):
    """Profile a model's forward and backward pass"""
    print(f"\n{'='*60}")
    print(f"Profiling: {name}")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Random data
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(10):
        logits, loss = model(x, y)
        loss.backward()
        model.zero_grad()
    
    # Profile forward pass
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(iterations):
        logits, loss = model(x, y)
    torch.cuda.synchronize() if device == 'cuda' else None
    forward_time = (time.time() - start) / iterations
    
    # Profile backward pass
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(iterations):
        logits, loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize() if device == 'cuda' else None
    total_time = (time.time() - start) / iterations
    backward_time = total_time - forward_time
    
    print(f"Forward pass:  {forward_time*1000:.2f}ms")
    print(f"Backward pass: {backward_time*1000:.2f}ms")
    print(f"Total:         {total_time*1000:.2f}ms")
    
    return {
        'forward': forward_time,
        'backward': backward_time,
        'total': total_time
    }

def main():
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 2,
        'block_size': 32,
        'dropout': 0.1
    }
    
    # Profile Baseline GPT
    baseline = BaselineGPT(**config)
    baseline_stats = profile_model(baseline, "Baseline GPT")
    
    # Profile Kolosis
    kolosis = KolosisTransformer(**config)
    kolosis_stats = profile_model(kolosis, "Kolosis (Full)")
    
    # Comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    overhead = (kolosis_stats['total'] - baseline_stats['total']) / baseline_stats['total'] * 100
    print(f"\nKolosis overhead: {overhead:.1f}%")
    print(f"Baseline: {baseline_stats['total']*1000:.2f}ms per step")
    print(f"Kolosis:  {kolosis_stats['total']*1000:.2f}ms per step")
    
    # Breakdown
    print(f"\nBreakdown:")
    print(f"  Forward overhead:  {(kolosis_stats['forward']/baseline_stats['forward']-1)*100:.1f}%")
    print(f"  Backward overhead: {(kolosis_stats['backward']/baseline_stats['backward']-1)*100:.1f}%")

if __name__ == "__main__":
    main()
