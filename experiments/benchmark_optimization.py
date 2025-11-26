"""
Benchmark optimized vs original Kolosis.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from neural_networks.kolosis import KolosisTransformer
from neural_networks.kolosis.optimized_kolosis import OptimizedKolosisTransformer
from neural_networks.autograd.transformer_torch import GPT as BaselineGPT

def benchmark_model(model, name, iterations=100):
    """Benchmark a model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    batch_size = 16
    seq_len = 32
    vocab_size = 100
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(10):
        logits, loss = model(x, y)
        loss.backward()
        model.zero_grad()
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(iterations):
        logits, loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = (time.time() - start) / iterations
    
    return elapsed * 1000  # Convert to ms

def main():
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 2,
        'block_size': 32,
        'dropout': 0.1
    }
    
    print("="*60)
    print("KOLOSIS OPTIMIZATION BENCHMARK")
    print("="*60)
    
    # Benchmark baseline
    print("\n1. Baseline GPT")
    baseline = BaselineGPT(**config)
    baseline_time = benchmark_model(baseline, "Baseline")
    print(f"   Time per step: {baseline_time:.2f}ms")
    
    # Benchmark original Kolosis
    print("\n2. Kolosis (Original)")
    kolosis_orig = KolosisTransformer(**config)
    kolosis_orig_time = benchmark_model(kolosis_orig, "Original Kolosis")
    print(f"   Time per step: {kolosis_orig_time:.2f}ms")
    print(f"   Overhead vs baseline: {(kolosis_orig_time/baseline_time-1)*100:.1f}%")
    
    # Benchmark optimized Kolosis
    print("\n3. Kolosis (Optimized)")
    kolosis_opt = OptimizedKolosisTransformer(**config)
    kolosis_opt_time = benchmark_model(kolosis_opt, "Optimized Kolosis")
    print(f"   Time per step: {kolosis_opt_time:.2f}ms")
    print(f"   Overhead vs baseline: {(kolosis_opt_time/baseline_time-1)*100:.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    speedup = kolosis_orig_time / kolosis_opt_time
    overhead_reduction = (kolosis_orig_time - kolosis_opt_time) / kolosis_orig_time * 100
    
    print(f"\nBaseline GPT:         {baseline_time:.2f}ms")
    print(f"Kolosis (Original):   {kolosis_orig_time:.2f}ms (+{(kolosis_orig_time/baseline_time-1)*100:.1f}%)")
    print(f"Kolosis (Optimized):  {kolosis_opt_time:.2f}ms (+{(kolosis_opt_time/baseline_time-1)*100:.1f}%)")
    print(f"\nSpeedup: {speedup:.2f}x faster")
    print(f"Overhead reduction: {overhead_reduction:.1f}%")
    
    # Check if we beat baseline
    if kolosis_opt_time < baseline_time:
        print(f"\n🎉 OPTIMIZED KOLOSIS IS FASTER THAN BASELINE! ({(baseline_time/kolosis_opt_time-1)*100:.1f}% faster)")
    elif kolosis_opt_time < baseline_time * 1.1:
        print(f"\n✅ OPTIMIZED KOLOSIS MATCHES BASELINE SPEED (within 10%)")
    else:
        print(f"\n⚠️  Still {(kolosis_opt_time/baseline_time-1)*100:.1f}% slower than baseline")

if __name__ == "__main__":
    main()
