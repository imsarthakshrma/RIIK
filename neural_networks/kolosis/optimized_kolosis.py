"""
Optimized Kolosis Transformer - Full Model
Integrates all optimized components for maximum performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimized_components import OptimizedTemporalAttention, OptimizedHierarchicalEmbedding

class OptimizedKolosisBlock(nn.Module):
    """Optimized Kolosis block with streamlined components"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        
        # Optimized multi-head temporal attention
        self.heads = nn.ModuleList([
            OptimizedTemporalAttention(head_size, n_embd, block_size, dropout) 
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Multi-head attention
        attn_out = torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1)
        attn_out = self.dropout(self.proj(attn_out))
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ffwd(self.ln2(x))
        
        return x


class OptimizedKolosisTransformer(nn.Module):
    """
    Optimized Kolosis Transformer with performance improvements.
    Maintains accuracy while reducing training time.
    """
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Optimized hierarchical embeddings
        self.embeddings = OptimizedHierarchicalEmbedding(vocab_size, n_embd, block_size)
        
        # Optimized transformer blocks
        self.blocks = nn.Sequential(*[
            OptimizedKolosisBlock(n_embd, n_head, block_size, dropout) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
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
        
        # Hierarchical embeddings
        x = self.embeddings(idx)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Output
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
        """Generate new tokens"""
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


if __name__ == "__main__":
    print("Testing Optimized Kolosis Transformer")
    
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 2,
        'block_size': 32,
        'dropout': 0.1
    }
    
    model = OptimizedKolosisTransformer(**config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    
    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    context = torch.randint(0, 100, (1, 5))
    generated = model.generate(context, max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")
    
    print("\nOptimized Kolosis ready!")
