"""
Train Kolosis V2 Minimal on TinyStories
Quick validation that simplification works.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import time
import json

from neural_networks.kolosis.kolosis_v2_minimal import KolosisV2Minimal
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import download_tinystories, TinyStoriesDataset

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    """Train model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    results = {'train_loss': [], 'val_loss': [], 'fusion_weights': []}
    best_val_loss = float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        results['train_loss'].append(avg_train)
        results['val_loss'].append(avg_val)
        results['fusion_weights'].append(model.get_fusion_weight())
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
        
        if epoch % 10 == 0:
            fusion_w = model.get_fusion_weight()
            print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Fusion: {fusion_w:.3f}")
    
    results['best_val_loss'] = best_val_loss
    results['final_val_loss'] = results['val_loss'][-1]
    
    return results

def main():
    # Configuration
    config = {
        'vocab_size': None,
        'n_embd': 64,
        'block_size': 32,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    print(f"Vocab size: {config['vocab_size']}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = KolosisV2Minimal(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\n" + "="*60)
    print("TRAINING KOLOSIS V2 MINIMAL")
    print("="*60)
    
    start_time = time.time()
    results = train_model(model, train_loader, val_loader, epochs=100, lr=0.003, device=device)
    total_time = time.time() - start_time
    
    results['total_time'] = total_time
    results['parameters'] = n_params
    
    # Save results
    os.makedirs('experiments/kolosis_v2_minimal_results', exist_ok=True)
    with open('experiments/kolosis_v2_minimal_results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Parameters: {n_params:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Final val loss: {results['final_val_loss']:.4f}")
    print(f"Final fusion weight: {results['fusion_weights'][-1]:.3f}")
    
    # Compare to baselines
    print("\n" + "="*60)
    print("COMPARISON TO BASELINES")
    print("="*60)
    
    baselines = {
        'Baseline GPT': 2.84,
        'Hierarchical-Only': 2.50,
        'Kolosis V1': 2.78,
        'Kolosis V2 Full': 2.56,
        'Kolosis V2 Minimal': results['final_val_loss']
    }
    
    for name, loss in baselines.items():
        improvement = (2.84 - loss) / 2.84 * 100
        marker = "🏆" if loss == min(baselines.values()) else ""
        print(f"{name:25s}: {loss:.4f} ({improvement:+.1f}%) {marker}")
    
    print(f"\nResults saved to experiments/kolosis_v2_minimal_results/results.json")

if __name__ == "__main__":
    main()
