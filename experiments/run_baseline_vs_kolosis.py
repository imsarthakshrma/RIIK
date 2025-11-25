"""
Run Baseline GPT vs Kolosis comparison experiment.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import download_tinystories, TinyStoriesDataset
from torch.utils.data import DataLoader
from experiments.ablation_study import AblationStudy

def main():
    # Configuration
    config = {
        'vocab_size': None,  # Will be set after tokenizer training
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
    
    # Load or create dataset
    print("\n=== Loading Dataset ===")
    json_path, toon_path = download_tinystories()
    
    # Create tokenizer
    print("\n=== Creating Tokenizer ===")
    with open(json_path, 'r') as f:
        import json
        data = json.load(f)
        text = ' '.join(data['stories'])
    
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    config['vocab_size'] = tokenizer.vocab_size
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample text: {text[:100]}...")
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    dataset = TinyStoriesDataset(json_path, tokenizer, block_size=config['block_size'])
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Run ablation study
    print("\n=== Starting Ablation Study ===")
    study = AblationStudy()
    
    # Experiment 1: Baseline GPT
    print("\n" + "="*60)
    print("EXPERIMENT 1: Baseline GPT")
    print("="*60)
    baseline_results = study.run_baseline_gpt(train_loader, val_loader, config, device)
    
    # Experiment 2: Full Kolosis
    print("\n" + "="*60)
    print("EXPERIMENT 2: Kolosis (Full)")
    print("="*60)
    kolosis_results = study.run_kolosis_full(train_loader, val_loader, config, device)
    
    # Save and report
    study.save_results()
    report = study.generate_report()
    
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(report)

if __name__ == "__main__":
    main()
