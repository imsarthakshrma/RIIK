"""
Optimized Kolosis components for faster training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedTemporalAttention(nn.Module):
    """
    Optimized Multi-Scale Temporal Attention with caching.
    """
    
    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.block_size = block_size
        
        # Learnable parameters (same as before)
        self.gamma_fast_logit = nn.Parameter(torch.tensor(0.0))
        self.gamma_medium_logit = nn.Parameter(torch.tensor(2.0))
        self.gamma_slow_logit = nn.Parameter(torch.tensor(4.0))
        self.alpha_logits = nn.Parameter(torch.zeros(3))
        
        # Causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # OPTIMIZATION: Cache temporal bias (recompute only when parameters change)
        self.register_buffer('_cached_temporal_bias', None)
        self.register_buffer('_cached_seq_len', torch.tensor(0))
        
    def compute_temporal_bias(self, seq_len, device):
        """Compute temporal bias with caching"""
        # Check if we can use cached bias
        if (self._cached_temporal_bias is not None and 
            self._cached_seq_len == seq_len and
            not self.training):  # Only cache during inference
            return self._cached_temporal_bias[:seq_len, :seq_len]
        
        # Compute decay rates
        gamma_fast = torch.sigmoid(self.gamma_fast_logit) * 0.3 + 0.6
        gamma_medium = torch.sigmoid(self.gamma_medium_logit) * 0.15 + 0.85
        gamma_slow = torch.sigmoid(self.gamma_slow_logit) * 0.05 + 0.95
        alpha = F.softmax(self.alpha_logits, dim=0)
        
        # OPTIMIZATION: Vectorized computation
        delta_t = torch.arange(seq_len, device=device).unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)
        delta_t = delta_t.clamp(min=0).float()
        
        # Compute all scales at once
        decays = torch.stack([
            alpha[0] * torch.pow(gamma_fast, delta_t),
            alpha[1] * torch.pow(gamma_medium, delta_t),
            alpha[2] * torch.pow(gamma_slow, delta_t)
        ], dim=0).sum(dim=0)
        
        temporal_bias = torch.log(decays + 1e-8)
        
        # Cache for inference
        if not self.training and seq_len <= self.block_size:
            self._cached_temporal_bias = temporal_bias
            self._cached_seq_len = torch.tensor(seq_len)
        
        return temporal_bias
        
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Content-based scores
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # Add temporal bias
        temporal_bias = self.compute_temporal_bias(T, x.device)
        wei = wei + temporal_bias.unsqueeze(0)
        
        # Causal mask and softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        return out


class OptimizedHierarchicalEmbedding(nn.Module):
    """
    Optimized Hierarchical Embedding with fused operations.
    """
    
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        
        # OPTIMIZATION: Single embedding lookup instead of 3 separate ones
        # We'll use embedding groups within one table
        self.combined_emb = nn.Embedding(vocab_size, n_embd * 3)
        self.position_emb = nn.Embedding(block_size, n_embd)
        
        # Learnable mixing weights
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.beta_logit = nn.Parameter(torch.tensor(-1.0))
        
        # OPTIMIZATION: Simpler concept extractor (1 layer instead of 2)
        self.concept_extractor = nn.Linear(n_embd, n_embd)
        
    def forward(self, idx):
        B, T = idx.shape
        
        # OPTIMIZATION: Single embedding lookup, then split
        combined = self.combined_emb(idx)  # (B, T, 3*C)
        n_embd = combined.shape[-1] // 3
        symbol, concept, law = combined.split(n_embd, dim=-1)
        
        # Position embeddings
        pos = self.position_emb(torch.arange(T, device=idx.device))
        
        # Concept refinement (simplified)
        concept_refined = F.relu(self.concept_extractor(concept))
        
        # Mixing weights
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        
        # OPTIMIZATION: Fused combination
        x = symbol + alpha * concept_refined + beta * law + pos
        
        return x


if __name__ == "__main__":
    print("Optimized Kolosis Components")
    print("="*60)
    
    # Test optimized temporal attention
    print("\n1. Optimized Temporal Attention")
    attn = OptimizedTemporalAttention(16, 64, 32)
    x = torch.randn(2, 16, 64)
    
    # Warmup
    for _ in range(10):
        _ = attn(x)
    
    # Time it
    import time
    start = time.time()
    for _ in range(100):
        _ = attn(x)
    elapsed = (time.time() - start) / 100
    print(f"   Average time: {elapsed*1000:.2f}ms")
    
    # Test optimized hierarchical embedding
    print("\n2. Optimized Hierarchical Embedding")
    emb = OptimizedHierarchicalEmbedding(100, 64, 32)
    idx = torch.randint(0, 100, (2, 16))
    
    # Warmup
    for _ in range(10):
        _ = emb(idx)
    
    # Time it
    start = time.time()
    for _ in range(100):
        _ = emb(idx)
    elapsed = (time.time() - start) / 100
    print(f"   Average time: {elapsed*1000:.2f}ms")
    
    print("\nOptimizations ready!")
