"""
3-Phase Training Pipeline for Kolosis V2
Phase 1: Pre-train streams independently
Phase 2: Train fusion gates
Phase 3: End-to-end fine-tuning
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json

from neural_networks.kolosis.kolosis_v2 import KolosisV2
from neural_networks.autograd.transformer_torch import GPT as BaselineGPT
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import TinyStoriesDataset, download_tinystories

def train_phase1_streams(model, train_loader, val_loader, epochs=20, lr=0.001, device='cuda'):
    """
    Phase 1: Pre-train each stream independently
    Each stream learns to predict on its own
    """
    print("\n" + "="*60)
    print("PHASE 1: Pre-training Streams Independently")
    print("="*60)
    
    model = model.to(device)
    
    # Separate optimizers for each stream
    stream_optimizers = {
        'symbol': torch.optim.AdamW(model.symbol_stream.parameters(), lr=lr),
        'concept': torch.optim.AdamW(model.concept_stream.parameters(), lr=lr),
        'temporal': torch.optim.AdamW(model.temporal_stream.parameters(), lr=lr),
        'semantic': torch.optim.AdamW(model.semantic_stream.parameters(), lr=lr),
    }
    
    results = {'train_loss': [], 'val_loss': [], 'stream_losses': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            _, loss, stream_info = model(x, y, return_stream_outputs=True)
            
            # Backward for each stream independently
            for opt in stream_optimizers.values():
                opt.zero_grad()
            
            loss.backward()
            
            for opt in stream_optimizers.values():
                opt.step()
            
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
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
    
    print(f"Phase 1 complete! Final val loss: {results['val_loss'][-1]:.4f}")
    return results

def train_phase2_fusion(model, train_loader, val_loader, epochs=30, lr=0.0003, device='cuda'):
    """
    Phase 2: Train fusion gates
    Freeze streams, learn optimal combination
    """
    print("\n" + "="*60)
    print("PHASE 2: Training Fusion Gates")
    print("="*60)
    
    model = model.to(device)
    
    # Freeze all streams
    for param in model.symbol_stream.parameters():
        param.requires_grad = False
    for param in model.concept_stream.parameters():
        param.requires_grad = False
    for param in model.temporal_stream.parameters():
        param.requires_grad = False
    for param in model.semantic_stream.parameters():
        param.requires_grad = False
    
    # Only train fusion gate and final head
    optimizer = torch.optim.AdamW([
        {'params': model.fusion_gate.parameters()},
        {'params': model.final_head.parameters()}
    ], lr=lr)
    
    results = {'train_loss': [], 'val_loss': [], 'gate_weights': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        epoch_gates = []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, loss, stream_info = model(x, y, return_stream_outputs=True)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            epoch_gates.append(stream_info['gate_weights'].mean(dim=[0,1]).detach().cpu())
        
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
        avg_gates = torch.stack(epoch_gates).mean(dim=0)
        
        results['train_loss'].append(avg_train)
        results['val_loss'].append(avg_val)
        results['gate_weights'].append(avg_gates.tolist())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Gates: {avg_gates.numpy()}")
    
    print(f"Phase 2 complete! Final val loss: {results['val_loss'][-1]:.4f}")
    return results

def train_phase3_finetune(model, train_loader, val_loader, epochs=50, lr=0.0001, device='cuda'):
    """
    Phase 3: End-to-end fine-tuning
    Unfreeze everything, joint optimization
    """
    print("\n" + "="*60)
    print("PHASE 3: End-to-End Fine-tuning")
    print("="*60)
    
    model = model.to(device)
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    results = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, loss = model(x, y)
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
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
    
    print(f"Phase 3 complete! Best val loss: {best_val_loss:.4f}")
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
    model = KolosisV2(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # 3-Phase Training
    start_time = time.time()
    
    phase1_results = train_phase1_streams(model, train_loader, val_loader, epochs=20, lr=0.001, device=device)
    phase2_results = train_phase2_fusion(model, train_loader, val_loader, epochs=30, lr=0.0003, device=device)
    phase3_results = train_phase3_finetune(model, train_loader, val_loader, epochs=50, lr=0.0001, device=device)
    
    total_time = time.time() - start_time
    
    # Save results
    results = {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'phase3': phase3_results,
        'total_time': total_time,
        'final_val_loss': phase3_results['val_loss'][-1]
    }
    
    os.makedirs('experiments/kolosis_v2_results', exist_ok=True)
    with open('experiments/kolosis_v2_results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Final val loss: {results['final_val_loss']:.4f}")
    print(f"Results saved to experiments/kolosis_v2_results/training_results.json")

if __name__ == "__main__":
    main()
