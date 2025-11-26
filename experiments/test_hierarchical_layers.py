"""
Test 2-layer vs 3-layer hierarchical embeddings.
Does Symbol+Concept match Symbol+Concept+Law?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.data import download_tinystories, TinyStoriesDataset

class HierarchicalEmbedding2Layer(nn.Module):
    """2-layer: Symbol + Concept only"""
    
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.symbol_emb = nn.Embedding(vocab_size, n_embd)
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, idx):
        B, T = idx.shape
        symbol = self.symbol_emb(idx)
        concept = self.concept_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        
        return symbol + self.alpha * concept + pos

class HierarchicalEmbedding3Layer(nn.Module):
    """3-layer: Symbol + Concept + Law"""
    
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.symbol_emb = nn.Embedding(vocab_size, n_embd)
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.law_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, idx):
        B, T = idx.shape
        symbol = self.symbol_emb(idx)
        concept = self.concept_emb(idx)
        law = self.law_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        
        return symbol + self.alpha * concept + self.beta * law + pos

class GPTWithHierarchical(nn.Module):
    """GPT with hierarchical embeddings"""
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, embedding_type='3layer', dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # Hierarchical embeddings
        if embedding_type == '2layer':
            self.embeddings = HierarchicalEmbedding2Layer(vocab_size, n_embd, block_size)
        else:
            self.embeddings = HierarchicalEmbedding3Layer(vocab_size, n_embd, block_size)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            
        return logits, loss

def train_model(model, train_loader, val_loader, epochs=100, lr=0.003, device='cuda'):
    """Train model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Val: {avg_val:.4f}")
    
    return best_val_loss

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
    print("HIERARCHICAL EMBEDDING LAYER ABLATION")
    print("="*60)
    print(f"Vocab size: {config['vocab_size']}")
    
    # Test 2-layer
    print("\n### Testing 2-Layer (Symbol + Concept) ###")
    model_2layer = GPTWithHierarchical(**config, embedding_type='2layer')
    params_2layer = sum(p.numel() for p in model_2layer.parameters())
    print(f"Parameters: {params_2layer:,}")
    
    start = time.time()
    loss_2layer = train_model(model_2layer, train_loader, val_loader, epochs=100, device=device)
    time_2layer = time.time() - start
    
    print(f"Best val loss: {loss_2layer:.4f}")
    print(f"Training time: {time_2layer:.1f}s")
    
    # Get learned weight
    alpha_2layer = torch.sigmoid(model_2layer.embeddings.alpha).item()
    print(f"Learned α (concept weight): {alpha_2layer:.4f}")
    
    # Test 3-layer
    print("\n### Testing 3-Layer (Symbol + Concept + Law) ###")
    model_3layer = GPTWithHierarchical(**config, embedding_type='3layer')
    params_3layer = sum(p.numel() for p in model_3layer.parameters())
    print(f"Parameters: {params_3layer:,}")
    
    start = time.time()
    loss_3layer = train_model(model_3layer, train_loader, val_loader, epochs=100, device=device)
    time_3layer = time.time() - start
    
    print(f"Best val loss: {loss_3layer:.4f}")
    print(f"Training time: {time_3layer:.1f}s")
    
    # Get learned weights
    alpha_3layer = torch.sigmoid(model_3layer.embeddings.alpha).item()
    beta_3layer = torch.sigmoid(model_3layer.embeddings.beta).item()
    print(f"Learned α (concept weight): {alpha_3layer:.4f}")
    print(f"Learned β (law weight): {beta_3layer:.4f}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\n2-Layer (Symbol + Concept):")
    print(f"  Parameters: {params_2layer:,}")
    print(f"  Best val loss: {loss_2layer:.4f}")
    print(f"  Training time: {time_2layer:.1f}s")
    print(f"  Concept weight: {alpha_2layer:.4f}")
    
    print(f"\n3-Layer (Symbol + Concept + Law):")
    print(f"  Parameters: {params_3layer:,}")
    print(f"  Best val loss: {loss_3layer:.4f}")
    print(f"  Training time: {time_3layer:.1f}s")
    print(f"  Concept weight: {alpha_3layer:.4f}")
    print(f"  Law weight: {beta_3layer:.4f}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    param_diff = params_3layer - params_2layer
    loss_diff = (loss_2layer - loss_3layer) / loss_2layer * 100
    
    print(f"\nParameter difference: {param_diff:,} (+{param_diff/params_2layer*100:.1f}%)")
    print(f"Performance difference: {loss_diff:+.1f}%")
    
    if abs(loss_diff) < 5:
        print("\n✅ CONCLUSION: 2-layer and 3-layer perform similarly!")
        print("   → Law layer may be redundant")
        print("   → Symbol + Concept is minimal sufficient architecture")
    elif loss_3layer < loss_2layer:
        print("\n⚠️  CONCLUSION: 3-layer outperforms 2-layer")
        print(f"   → Law layer adds value ({loss_diff:.1f}% improvement)")
        print("   → Keep all three layers")
    else:
        print("\n⚠️  CONCLUSION: 2-layer outperforms 3-layer")
        print(f"   → Law layer hurts performance ({-loss_diff:.1f}% worse)")
        print("   → Remove Law layer")
    
    # Check if Law weight is low
    if beta_3layer < 0.2:
        print(f"\n💡 INSIGHT: Law weight is low ({beta_3layer:.4f})")
        print("   → Model doesn't use Law layer much")
        print("   → Supports removing it")

if __name__ == "__main__":
    main()
