"""
Hybrid models for component-level ablation studies.
Each model adds one Kolosis component to baseline GPT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.autograd.transformer_torch import GPT as BaselineGPT
from neural_networks.kolosis.temporal_attention import MultiScaleMultiHeadAttention
from neural_networks.kolosis.hierarchical_embedding import HierarchicalEmbedding

class GPT_TemporalAttention(nn.Module):
    """GPT with only Multi-Scale Temporal Attention"""
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        
        # Standard embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Blocks with temporal attention
        self.blocks = nn.Sequential(*[
            TemporalBlock(n_embd, n_head, block_size, dropout) 
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
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


class TemporalBlock(nn.Module):
    """Transformer block with temporal attention"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiScaleMultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT_HierarchicalEmbedding(nn.Module):
    """GPT with only Hierarchical Embeddings"""
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        
        # Hierarchical embeddings
        self.embeddings = HierarchicalEmbedding(vocab_size, n_embd, block_size)
        
        # Standard transformer blocks
        self.blocks = nn.Sequential(*[
            StandardBlock(n_embd, n_head, block_size, dropout) 
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
        B, T = idx.shape
        
        x = self.embeddings(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


class StandardBlock(nn.Module):
    """Standard transformer block"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        attn_out, _ = self.sa(x, x, x, need_weights=False)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x


if __name__ == "__main__":
    # Test hybrid models
    vocab_size = 100
    n_embd = 64
    n_head = 4
    n_layer = 2
    block_size = 32
    
    print("Testing Hybrid Models")
    
    # Test temporal attention model
    print("\n1. GPT + Temporal Attention")
    model1 = GPT_TemporalAttention(vocab_size, n_embd, n_head, n_layer, block_size)
    x = torch.randint(0, vocab_size, (2, 16))
    y = torch.randint(0, vocab_size, (2, 16))
    logits, loss = model1(x, y)
    print(f"   Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test hierarchical embedding model
    print("\n2. GPT + Hierarchical Embeddings")
    model2 = GPT_HierarchicalEmbedding(vocab_size, n_embd, n_head, n_layer, block_size)
    logits, loss = model2(x, y)
    print(f"   Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    print("\nHybrid models ready for ablation studies!")
