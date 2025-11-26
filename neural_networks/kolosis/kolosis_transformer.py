"""
Full Kolosis Transformer Model
Integrates all cognitive-inspired components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_attention import MultiScaleMultiHeadAttention
from .hierarchical_embedding import HierarchicalEmbedding, ConceptClassifier
from .pattern_memory import PatternMemory

class KolosisBlock(nn.Module):
    """Kolosis Transformer block with cognitive mechanisms"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        
        # Multi-scale temporal attention
        self.sa = MultiScaleMultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        
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
        
        # Concept classifier for dual processing
        self.concept_classifier = ConceptClassifier(n_embd)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C) input
            
        Returns:
            x: (B, T, C) output
        """
        # Self-attention with residual
        attn_out = self.sa(self.ln1(x))
        x = x + attn_out
        
        # Concept classification and dual processing
        context = x.mean(dim=1, keepdim=True).expand_as(x)  # Simple context
        intrinsic_prob = self.concept_classifier(x, context)
        x_processed = self.concept_classifier.get_dual_processing(x, intrinsic_prob)
        
        # Feed-forward with residual
        ff_out = self.ffwd(self.ln2(x_processed))
        x = x + ff_out
        
        return x


class KolosisTransformer(nn.Module):
    """
    Full Kolosis Transformer with all cognitive-inspired mechanisms.
    """
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Hierarchical embeddings
        self.embeddings = HierarchicalEmbedding(vocab_size, n_embd, block_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            KolosisBlock(n_embd, n_head, block_size, dropout) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Language model head
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Pattern memory (optional, for advanced reasoning)
        self.pattern_memory = PatternMemory(n_embd, max_patterns=100)
        self.use_pattern_memory = False  # Can be enabled for experiments
        
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
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices (optional)
            
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar (if targets provided)
        """
        B, T = idx.shape
        
        # Hierarchical embeddings
        x = self.embeddings(idx)  # (B, T, C)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            # Update pattern memory if enabled
            if self.use_pattern_memory and self.training:
                context = x.mean(dim=1)  # (B, C)
                # Note: Would need attention weights for full pattern extraction
                self.pattern_memory.update_patterns(loss.item())
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens"""
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    
    def get_cognitive_stats(self):
        """Get statistics about cognitive mechanisms"""
        stats = {
            'hierarchy_weights': self.embeddings.get_hierarchy_weights(),
            'temporal_stats': [block.sa.get_all_temporal_stats() for block in self.blocks],
            'pattern_memory': self.pattern_memory.get_pattern_stats() if self.use_pattern_memory else None,
        }
        return stats


if __name__ == "__main__":
    # Test Kolosis transformer
    vocab_size = 100
    n_embd = 64
    n_head = 4
    n_layer = 2
    block_size = 32
    batch_size = 2
    seq_len = 16
    
    print("=== Kolosis Transformer Test ===")
    
    # Create model
    model = KolosisTransformer(vocab_size, n_embd, n_head, n_layer, block_size)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Random input
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(idx, targets)
    print(f"\nInput shape: {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n=== Generation Test ===")
    context = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(context, max_new_tokens=10, temperature=0.8, top_k=10)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    # Get cognitive stats
    print("\n=== Cognitive Mechanisms Stats ===")
    stats = model.get_cognitive_stats()
    print(f"Hierarchy weights: {stats['hierarchy_weights']}")
    print(f"Temporal stats (first head, first layer):")
    for key, val in stats['temporal_stats'][0][0].items():
        print(f"  {key}: {val:.4f}")
