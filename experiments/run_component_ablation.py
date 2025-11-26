"""
Run component-level ablation studies.
Tests each Kolosis component individually.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import download_tinystories, TinyStoriesDataset
from torch.utils.data import DataLoader
from experiments.ablation_study import AblationStudy
from experiments.hybrid_models import GPT_TemporalAttention, GPT_HierarchicalEmbedding

def main():
    # Configuration
    config = {
        'vocab_size': None,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 2,
        'block_size': 32,
        'dropout': 0.1,
        'epochs': 100,
        'lr': 0.0003,
        'batch_size': 16,
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    json_path, toon_path = download_tinystories()
    
    with open(json_path, 'r') as f:
        import json
        data = json.load(f)
        text = ' '.join(data['stories'])
    
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    config['vocab_size'] = tokenizer.vocab_size
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create datasets
    dataset = TinyStoriesDataset(json_path, tokenizer, block_size=config['block_size'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Run component ablation studies
    study = AblationStudy(output_dir='experiments/component_ablation_results')
    
    # Experiment 1: GPT + Temporal Attention Only
    print("\n" + "="*60)
    print("EXPERIMENT 1: GPT + Temporal Attention Only")
    print("="*60)
    
    model1 = GPT_TemporalAttention(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout']
    )
    
    results1 = study.train_model(
        model1, train_loader, val_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        model_name='GPT + Temporal Attention',
        device=device
    )
    study.results['gpt_temporal'] = results1
    
    # Experiment 2: GPT + Hierarchical Embeddings Only
    print("\n" + "="*60)
    print("EXPERIMENT 2: GPT + Hierarchical Embeddings Only")
    print("="*60)
    
    model2 = GPT_HierarchicalEmbedding(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout']
    )
    
    results2 = study.train_model(
        model2, train_loader, val_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        model_name='GPT + Hierarchical Embeddings',
        device=device
    )
    study.results['gpt_hierarchical'] = results2
    
    # Save results
    study.save_results()
    
    # Generate component comparison report
    print("\n" + "="*60)
    print("COMPONENT ABLATION SUMMARY")
    print("="*60)
    
    if 'gpt_temporal' in study.results and 'gpt_hierarchical' in study.results:
        temporal = study.results['gpt_temporal']
        hierarchical = study.results['gpt_hierarchical']
        
        print(f"\nGPT + Temporal Attention:")
        print(f"  Final val loss: {temporal['final_val_loss']:.4f}")
        print(f"  Training time: {temporal['total_time']:.1f}s")
        
        print(f"\nGPT + Hierarchical Embeddings:")
        print(f"  Final val loss: {hierarchical['final_val_loss']:.4f}")
        print(f"  Training time: {hierarchical['total_time']:.1f}s")

if __name__ == "__main__":
    main()
