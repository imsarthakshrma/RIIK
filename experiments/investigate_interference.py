"""
Investigate why full Kolosis underperforms individual components.
Tests three hypotheses:
1. Competing gradients between mechanisms
2. Hyperparameter mismatch (different optimal learning rates)
3. Information bottlenecks in early layers
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from neural_networks.kolosis import KolosisTransformer
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import TinyStoriesDataset
from experiments.hybrid_models import GPT_HierarchicalEmbedding, GPT_TemporalAttention

def test_gradient_conflicts(model, train_loader, device='cuda'):
    """
    Test Hypothesis 1: Competing gradients between mechanisms
    Measure gradient magnitudes and directions for different components
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 1: Gradient Conflicts")
    print("="*60)
    
    model = model.to(device)
    model.train()
    
    # Get one batch
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    
    # Forward and backward
    logits, loss = model(x, y)
    loss.backward()
    
    # Analyze gradients by component
    gradient_stats = {}
    
    # Embeddings gradients
    if hasattr(model, 'embeddings'):
        emb_grads = []
        for name, param in model.embeddings.named_parameters():
            if param.grad is not None:
                emb_grads.append(param.grad.norm().item())
        gradient_stats['embeddings'] = {
            'mean': np.mean(emb_grads) if emb_grads else 0,
            'std': np.std(emb_grads) if emb_grads else 0,
            'max': np.max(emb_grads) if emb_grads else 0
        }
    
    # Attention gradients
    attn_grads = []
    for name, param in model.named_parameters():
        if 'sa' in name or 'heads' in name:
            if param.grad is not None:
                attn_grads.append(param.grad.norm().item())
    gradient_stats['attention'] = {
        'mean': np.mean(attn_grads) if attn_grads else 0,
        'std': np.std(attn_grads) if attn_grads else 0,
        'max': np.max(attn_grads) if attn_grads else 0
    }
    
    # FFN gradients
    ffn_grads = []
    for name, param in model.named_parameters():
        if 'ffwd' in name:
            if param.grad is not None:
                ffn_grads.append(param.grad.norm().item())
    gradient_stats['feedforward'] = {
        'mean': np.mean(ffn_grads) if ffn_grads else 0,
        'std': np.std(ffn_grads) if ffn_grads else 0,
        'max': np.max(ffn_grads) if ffn_grads else 0
    }
    
    print("\nGradient Magnitudes by Component:")
    for component, stats in gradient_stats.items():
        print(f"  {component:15s}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, max={stats['max']:.6f}")
    
    # Check for gradient imbalance
    means = [stats['mean'] for stats in gradient_stats.values() if stats['mean'] > 0]
    if means:
        ratio = max(means) / min(means)
        print(f"\nGradient imbalance ratio: {ratio:.2f}x")
        if ratio > 10:
            print("⚠️  SEVERE gradient imbalance detected!")
        elif ratio > 5:
            print("⚠️  Moderate gradient imbalance detected")
        else:
            print("✅ Gradients relatively balanced")
    
    model.zero_grad()
    return gradient_stats

def test_learning_rate_sensitivity(model_class, train_loader, val_loader, config, device='cuda'):
    """
    Test Hypothesis 2: Hyperparameter mismatch
    Try different learning rates and see which works best
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 2: Learning Rate Sensitivity")
    print("="*60)
    
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
    results = {}
    
    for lr in learning_rates:
        print(f"\nTesting lr={lr}")
        model = model_class(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Train for 20 epochs
        losses = []
        for epoch in range(20):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        
        final_loss = np.mean(losses[-10:])  # Average of last 10 steps
        results[lr] = final_loss
        print(f"  Final loss: {final_loss:.4f}")
    
    # Find optimal
    optimal_lr = min(results, key=results.get)
    print(f"\nOptimal learning rate: {optimal_lr}")
    print(f"Loss range: {min(results.values()):.4f} - {max(results.values()):.4f}")
    
    return results

def test_information_flow(model, train_loader, device='cuda'):
    """
    Test Hypothesis 3: Information bottlenecks
    Measure activation magnitudes through layers
    """
    print("\n" + "="*60)
    print("HYPOTHESIS 3: Information Bottlenecks")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    # Hook to capture activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    if hasattr(model, 'embeddings'):
        model.embeddings.register_forward_hook(get_activation('embeddings'))
    
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            block.register_forward_hook(get_activation(f'block_{i}'))
    
    # Forward pass
    x, y = next(iter(train_loader))
    x = x.to(device)
    with torch.no_grad():
        _ = model(x)
    
    # Analyze activation magnitudes
    print("\nActivation Statistics by Layer:")
    for name, act in activations.items():
        mean_act = act.abs().mean().item()
        std_act = act.std().item()
        max_act = act.abs().max().item()
        print(f"  {name:15s}: mean={mean_act:.4f}, std={std_act:.4f}, max={max_act:.4f}")
    
    # Check for bottlenecks (very small activations)
    bottlenecks = []
    for name, act in activations.items():
        mean_act = act.abs().mean().item()
        if mean_act < 0.1:
            bottlenecks.append((name, mean_act))
    
    if bottlenecks:
        print(f"\n⚠️  Potential bottlenecks detected:")
        for name, val in bottlenecks:
            print(f"    {name}: {val:.6f}")
    else:
        print("\n✅ No obvious bottlenecks detected")
    
    return activations

def main():
    # Setup
    config = {
        'vocab_size': 29,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 2,
        'block_size': 32,
        'dropout': 0.1
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    from neural_networks.data import download_tinystories
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
    
    print("\n" + "="*60)
    print("COMPONENT INTERFERENCE INVESTIGATION")
    print("="*60)
    
    # Test full Kolosis
    print("\n### Testing Full Kolosis ###")
    kolosis = KolosisTransformer(**config)
    
    grad_stats = test_gradient_conflicts(kolosis, train_loader, device)
    info_flow = test_information_flow(kolosis, train_loader, device)
    
    # Test learning rate sensitivity
    print("\n### Testing Hierarchical Embeddings (for comparison) ###")
    hier_lr_results = test_learning_rate_sensitivity(
        GPT_HierarchicalEmbedding, train_loader, val_loader, config, device
    )
    
    print("\n### Testing Full Kolosis LR Sensitivity ###")
    kolosis_lr_results = test_learning_rate_sensitivity(
        KolosisTransformer, train_loader, val_loader, config, device
    )
    
    # Summary
    print("\n" + "="*60)
    print("INVESTIGATION SUMMARY")
    print("="*60)
    
    print("\n1. Gradient Conflicts:")
    print(f"   - Gradient imbalance detected: Check ratios above")
    
    print("\n2. Learning Rate Sensitivity:")
    print(f"   - Hierarchical optimal LR: {min(hier_lr_results, key=hier_lr_results.get)}")
    print(f"   - Kolosis optimal LR: {min(kolosis_lr_results, key=kolosis_lr_results.get)}")
    
    print("\n3. Information Bottlenecks:")
    print(f"   - Check activation statistics above")

if __name__ == "__main__":
    main()
