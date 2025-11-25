"""
Multi-Scale Temporal Attention for KOLOSIS
Implements attention with learnable temporal decay rates across 3 scales.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleTemporalAttention(nn.Module):
    """
    Single head of self-attention with multi-scale temporal bias.
    
    Traditional: Attention(Q,K,V) = softmax(QK^T / √d_k) V
    Kolosis: Attention(Q,K,V) = softmax(QK^T / √d_k + T(Δt)) V
    
    Where T_ij(Δt) = log(Σ α_s · γ_s^Δt_ij) for s ∈ {fast, medium, slow}
    """
    
    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.block_size = block_size
        
        # Learnable decay rates for 3 temporal scales
        # Use sigmoid to constrain to (0, 1)
        self.gamma_fast_logit = nn.Parameter(torch.tensor(0.0))  # ~0.7 after sigmoid
        self.gamma_medium_logit = nn.Parameter(torch.tensor(2.0))  # ~0.9 after sigmoid
        self.gamma_slow_logit = nn.Parameter(torch.tensor(4.0))  # ~0.98 after sigmoid
        
        # Learnable mixing weights (use softmax for normalization)
        self.alpha_logits = nn.Parameter(torch.zeros(3))
        
        # Register causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def compute_temporal_bias(self, seq_len, device):
        """
        Compute temporal bias matrix T(Δt)
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
            
        Returns:
            temporal_bias: (seq_len, seq_len) matrix
        """
        # Δt_ij = i - j (temporal distance between tokens)
        i_indices = torch.arange(seq_len, device=device).unsqueeze(1)
        j_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        delta_t = (i_indices - j_indices).float()
        delta_t = delta_t.clamp(min=0)  # Only consider past tokens
        
        # Convert logits to decay rates using sigmoid
        gamma_fast = torch.sigmoid(self.gamma_fast_logit) * 0.3 + 0.6  # Range: [0.6, 0.9]
        gamma_medium = torch.sigmoid(self.gamma_medium_logit) * 0.15 + 0.85  # Range: [0.85, 1.0]
        gamma_slow = torch.sigmoid(self.gamma_slow_logit) * 0.05 + 0.95  # Range: [0.95, 1.0]
        
        # Compute mixing weights using softmax
        alpha = F.softmax(self.alpha_logits, dim=0)
        
        # Compute decay for each scale: α_s · γ_s^Δt
        fast_decay = alpha[0] * torch.pow(gamma_fast, delta_t)
        medium_decay = alpha[1] * torch.pow(gamma_medium, delta_t)
        slow_decay = alpha[2] * torch.pow(gamma_slow, delta_t)
        
        # Sum and take log for numerical stability
        # T(Δt) = log(Σ α_s · γ_s^Δt_ij)
        temporal_bias = torch.log(fast_decay + medium_decay + slow_decay + 1e-8)
        
        return temporal_bias
        
    def forward(self, x):
        """
        Forward pass with temporal bias
        
        Args:
            x: (B, T, C) input tensor
            
        Returns:
            out: (B, T, head_size) output tensor
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # Compute content-based attention scores
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)
        
        # Add temporal bias
        temporal_bias = self.compute_temporal_bias(T, x.device)  # (T, T)
        wei = wei + temporal_bias.unsqueeze(0)  # Broadcast across batch
        
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted aggregation
        out = wei @ v  # (B, T, head_size)
        return out
    
    def get_temporal_stats(self):
        """Get current temporal scale parameters for analysis"""
        gamma_fast = torch.sigmoid(self.gamma_fast_logit) * 0.3 + 0.6
        gamma_medium = torch.sigmoid(self.gamma_medium_logit) * 0.15 + 0.85
        gamma_slow = torch.sigmoid(self.gamma_slow_logit) * 0.05 + 0.95
        alpha = F.softmax(self.alpha_logits, dim=0)
        
        return {
            'gamma_fast': gamma_fast.item(),
            'gamma_medium': gamma_medium.item(),
            'gamma_slow': gamma_slow.item(),
            'alpha_fast': alpha[0].item(),
            'alpha_medium': alpha[1].item(),
            'alpha_slow': alpha[2].item(),
        }


class MultiScaleMultiHeadAttention(nn.Module):
    """Multiple heads of multi-scale temporal attention in parallel"""
    
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([
            MultiScaleTemporalAttention(head_size, n_embd, block_size, dropout) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
    def get_all_temporal_stats(self):
        """Get temporal stats from all heads"""
        return [head.get_temporal_stats() for head in self.heads]


if __name__ == "__main__":
    # Test multi-scale temporal attention
    batch_size = 2
    seq_len = 8
    n_embd = 32
    head_size = 8
    block_size = 16
    
    # Create attention module
    attn = MultiScaleTemporalAttention(head_size, n_embd, block_size)
    
    # Random input
    x = torch.randn(batch_size, seq_len, n_embd)
    
    # Forward pass
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Check temporal stats
    stats = attn.get_temporal_stats()
    print(f"\nTemporal Scale Parameters:")
    print(f"  Fast decay (γ_fast): {stats['gamma_fast']:.4f}, weight: {stats['alpha_fast']:.4f}")
    print(f"  Medium decay (γ_medium): {stats['gamma_medium']:.4f}, weight: {stats['alpha_medium']:.4f}")
    print(f"  Slow decay (γ_slow): {stats['gamma_slow']:.4f}, weight: {stats['alpha_slow']:.4f}")
