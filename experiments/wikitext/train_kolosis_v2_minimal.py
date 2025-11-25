"""
WikiText-103 Training: Kolosis V2 Minimal
Combines hierarchical embeddings + semantic stream with direct supervision.
Optimized for longer contexts (256 tokens) where temporal patterns emerge.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

class HierarchicalEmbedding2Layer(nn.Module):
    """2-layer hierarchical: Symbol + Concept"""
    
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
        return symbol + torch.sigmoid(self.alpha) * concept + pos

class KolosisV2Minimal(nn.Module):
    """
    Kolosis V2 Minimal for WikiText-103
    - Hierarchical embeddings (Symbol + Concept)
    - Dual streams (Concept + Semantic)
    - Direct supervision
    - Learned fusion
    """
    
    def __init__(self, vocab_size, n_embd=256, block_size=256, n_layer=6, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Hierarchical embeddings
        self.symbol_emb = nn.Embedding(vocab_size, n_embd)
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Relationship encoder for semantic stream
        self.relation_encoder = nn.Linear(n_embd * 2, n_embd)
        
        # Concept stream
        self.concept_stream = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=8,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layer)
        ])
        
        # Semantic stream
        self.semantic_stream = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=8,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layer)
        ])
        
        # Layer norms
        self.concept_norm = nn.LayerNorm(n_embd)
        self.semantic_norm = nn.LayerNorm(n_embd)
        
        # Prediction heads (direct supervision)
        self.concept_head = nn.Linear(n_embd, vocab_size)
        self.semantic_head = nn.Linear(n_embd, vocab_size)
        
        # Fusion
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.ensemble_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_hierarchical_embedding(self, idx):
        """Create hierarchical embeddings"""
        B, T = idx.shape
        symbol = self.symbol_emb(idx)
        concept = self.concept_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        return symbol + torch.sigmoid(self.alpha) * concept + pos
    
    def create_semantic_embedding(self, idx):
        """Create relationship-aware embeddings"""
        B, T = idx.shape
        base_emb = self.create_hierarchical_embedding(idx)
        
        if T > 1:
            pairs = torch.cat([base_emb[:, :-1], base_emb[:, 1:]], dim=-1)
            relation_features = self.relation_encoder(pairs)
            relation_features = F.pad(relation_features, (0, 0, 0, 1))
            return base_emb + relation_features
        else:
            return base_emb
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Create embeddings
        concept_emb = self.create_hierarchical_embedding(idx)
        semantic_emb = self.create_semantic_embedding(idx)
        
        # Process through streams
        concept_feat = self.concept_stream(concept_emb)
        concept_feat = self.concept_norm(concept_feat)
        concept_logits = self.concept_head(concept_feat)
        
        semantic_feat = self.semantic_stream(semantic_emb)
        semantic_feat = self.semantic_norm(semantic_feat)
        semantic_logits = self.semantic_head(semantic_feat)
        
        # Fusion
        fusion_weight = torch.sigmoid(self.fusion_weight)
        fused_feat = fusion_weight * concept_feat + (1 - fusion_weight) * semantic_feat
        ensemble_logits = self.ensemble_head(fused_feat)
        
        # Loss
        loss = None
        if targets is not None:
            concept_loss = F.cross_entropy(concept_logits.view(B*T, self.vocab_size), targets.view(B*T))
            semantic_loss = F.cross_entropy(semantic_logits.view(B*T, self.vocab_size), targets.view(B*T))
            ensemble_loss = F.cross_entropy(ensemble_logits.view(B*T, self.vocab_size), targets.view(B*T))
            
            loss = 0.5 * ensemble_loss + 0.25 * concept_loss + 0.25 * semantic_loss
        
        return ensemble_logits, loss

class WikiTextDataset(Dataset):
    """WikiText-103 dataset"""
    
    def __init__(self, texts, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        
        print(f"Tokenizing {len(texts)} documents...")
        for text in tqdm(texts):
            if len(text.strip()) == 0:
                continue
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            for i in range(0, len(tokens) - block_size, block_size // 2):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) == block_size + 1:
                    self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def main():
    print("="*60)
    print("WIKITEXT-103: KOLOSIS V2 MINIMAL")
    print("="*60)
    
    config = {
        'vocab_size': 50257,
        'n_embd': 256,
        'block_size': 256,
        'n_layer': 6,
        'dropout': 0.1,
        'batch_size': 32,
        'epochs': 10,
        'lr': 0.0003
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("\nLoading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    train_dataset = WikiTextDataset(dataset['train']['text'], tokenizer, config['block_size'])
    val_dataset = WikiTextDataset(dataset['validation']['text'], tokenizer, config['block_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True)
    
    # Create model
    print(f"\nCreating Kolosis V2 Minimal...")
    model = KolosisV2Minimal(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    results = {'config': config, 'train_losses': [], 'val_losses': [], 'perplexities': [], 'fusion_weights': []}
    best_val_loss = float('inf')
    os.makedirs('experiments/wikitext_results', exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1)
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        fusion_weight = torch.sigmoid(model.fusion_weight).item()
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        results['fusion_weights'].append(fusion_weight)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Concept weight (α): {torch.sigmoid(model.alpha).item():.4f}")
        print(f"  Fusion weight: {fusion_weight:.4f} (concept: {fusion_weight:.2f}, semantic: {1-fusion_weight:.2f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'experiments/wikitext_results/kolosis_v2_minimal_best.pt')
            print("  ✅ Saved best model")
    
    with open('experiments/wikitext_results/kolosis_v2_minimal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    print(f"Final fusion weight: {results['fusion_weights'][-1]:.4f}")

if __name__ == "__main__":
    main()
