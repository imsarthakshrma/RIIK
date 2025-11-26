"""
Kolosis V2 Minimal: Streamlined architecture based on learnings.
Only the components that matter: Hierarchical + Semantic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class KolosisV2Minimal(nn.Module):
    """
    Minimal Kolosis V2 with only proven components:
    - Hierarchical embeddings (proven winner: +12%)
    - Semantic stream (47% gate weight in V2)
    
    Removed:
    - Symbol stream (13% gate weight - redundant with hierarchical)
    - Temporal stream (12% gate weight - needs longer contexts)
    """
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Hierarchical embeddings (Symbol/Concept/Law)
        self.symbol_emb = nn.Embedding(vocab_size, n_embd)
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.law_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Learnable hierarchy weights
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Concept weight
        self.beta = nn.Parameter(torch.tensor(0.3))   # Law weight
        
        # Concept abstraction stream
        self.concept_stream = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=4,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        # Semantic/relationship stream
        self.semantic_stream = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=4,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        # Relationship encoder (for semantic stream)
        self.relation_encoder = nn.Linear(n_embd * 2, n_embd)
        
        # Layer norms
        self.concept_norm = nn.LayerNorm(n_embd)
        self.semantic_norm = nn.LayerNorm(n_embd)
        
        # Dual prediction heads (direct supervision)
        self.concept_head = nn.Linear(n_embd, vocab_size)
        self.semantic_head = nn.Linear(n_embd, vocab_size)
        
        # Simple fusion (learned weighted average)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # Final ensemble head
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
        law = self.law_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        
        # Hierarchical combination
        return symbol + self.alpha * concept + self.beta * law + pos
    
    def create_semantic_embedding(self, idx):
        """Create relationship-aware embeddings"""
        B, T = idx.shape
        
        # Base hierarchical embedding
        base_emb = self.create_hierarchical_embedding(idx)
        
        # Add pairwise relationship encoding
        if T > 1:
            pairs = torch.cat([base_emb[:, :-1], base_emb[:, 1:]], dim=-1)
            relation_features = self.relation_encoder(pairs)
            relation_features = F.pad(relation_features, (0, 0, 0, 1))
            return base_emb + relation_features
        else:
            return base_emb
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices (optional)
            
        Returns:
            logits: (B, T, vocab_size) predictions
            loss: Scalar loss (if targets provided)
        """
        B, T = idx.shape
        
        # Create embeddings for both streams
        concept_emb = self.create_hierarchical_embedding(idx)
        semantic_emb = self.create_semantic_embedding(idx)
        
        # Process through concept stream
        concept_feat = concept_emb
        for layer in self.concept_stream:
            concept_feat = layer(concept_feat)
        concept_feat = self.concept_norm(concept_feat)
        concept_logits = self.concept_head(concept_feat)
        
        # Process through semantic stream
        semantic_feat = semantic_emb
        for layer in self.semantic_stream:
            semantic_feat = layer(semantic_feat)
        semantic_feat = self.semantic_norm(semantic_feat)
        semantic_logits = self.semantic_head(semantic_feat)
        
        # Fusion: learned weighted average
        fusion_weight = torch.sigmoid(self.fusion_weight)
        fused_feat = fusion_weight * concept_feat + (1 - fusion_weight) * semantic_feat
        ensemble_logits = self.ensemble_head(fused_feat)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Multi-head supervision
            concept_loss = F.cross_entropy(
                concept_logits.view(B*T, self.vocab_size),
                targets.view(B*T)
            )
            semantic_loss = F.cross_entropy(
                semantic_logits.view(B*T, self.vocab_size),
                targets.view(B*T)
            )
            ensemble_loss = F.cross_entropy(
                ensemble_logits.view(B*T, self.vocab_size),
                targets.view(B*T)
            )
            
            # Combined loss: 50% ensemble + 25% concept + 25% semantic
            loss = 0.5 * ensemble_loss + 0.25 * concept_loss + 0.25 * semantic_loss
        
        return ensemble_logits, loss
    
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
    
    def get_fusion_weight(self):
        """Get current fusion weight"""
        return torch.sigmoid(self.fusion_weight).item()


if __name__ == "__main__":
    print("Testing Kolosis V2 Minimal")
    
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'block_size': 32,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    model = KolosisV2Minimal(**config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    
    logits, loss = model(x, y)
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Fusion weight: {model.get_fusion_weight():.4f}")
    
    # Test generation
    context = torch.randint(0, 100, (1, 5))
    generated = model.generate(context, max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")
    
    print("\nKolosis V2 Minimal ready!")
    print(f"Parameter reduction: {n_params} vs ~454K (V2 Full) = {(1 - n_params/454000)*100:.1f}% smaller")
