"""
Kolosis V2: Parallel Stream Architecture
Each cognitive mechanism has its own direct supervision path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelStream(nn.Module):
    """Base class for cognitive stream with direct supervision"""
    
    def __init__(self, n_embd, vocab_size, n_layer=2, dropout=0.1):
        super().__init__()
        self.n_embd = n_embd
        
        # Stream-specific processing
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=4,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(n_embd)
        
        # Direct supervision head
        self.prediction_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, C) input features
            mask: Optional attention mask
            
        Returns:
            features: (B, T, C) processed features
            logits: (B, T, vocab_size) predictions
        """
        # Process through layers
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        features = self.norm(x)
        logits = self.prediction_head(features)
        
        return features, logits


class SymbolStream(ParallelStream):
    """Fast token processing stream (baseline GPT-like)"""
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__(n_embd, vocab_size, n_layer, dropout)
        
        # Simple token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
    def embed(self, idx):
        """Create embeddings for this stream"""
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        return tok_emb + pos_emb


class ConceptStream(ParallelStream):
    """Hierarchical abstraction stream"""
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__(n_embd, vocab_size, n_layer, dropout)
        
        # Hierarchical embeddings (Symbol/Concept/Law)
        self.symbol_emb = nn.Embedding(vocab_size, n_embd)
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.law_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Learnable mixing weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))
        
    def embed(self, idx):
        """Create hierarchical embeddings"""
        B, T = idx.shape
        
        symbol = self.symbol_emb(idx)
        concept = self.concept_emb(idx)
        law = self.law_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        
        # Hierarchical combination
        return symbol + self.alpha * concept + self.beta * law + pos


class TemporalStream(ParallelStream):
    """Multi-scale temporal memory stream"""
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__(n_embd, vocab_size, n_layer, dropout)
        
        # Standard embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Temporal decay parameters
        self.gamma_fast = nn.Parameter(torch.tensor(0.7))
        self.gamma_medium = nn.Parameter(torch.tensor(0.9))
        self.gamma_slow = nn.Parameter(torch.tensor(0.99))
        
        # Temporal bias projection
        self.temporal_proj = nn.Linear(3, 1)
        
    def embed(self, idx):
        """Create embeddings with temporal awareness"""
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        
        # Add temporal bias encoding
        temporal_encoding = self._compute_temporal_encoding(T, idx.device)
        
        return tok_emb + pos_emb + temporal_encoding.unsqueeze(0)
    
    def _compute_temporal_encoding(self, seq_len, device):
        """Compute multi-scale temporal encoding"""
        positions = torch.arange(seq_len, device=device).float()
        
        # Three temporal scales
        fast = torch.pow(self.gamma_fast, positions)
        medium = torch.pow(self.gamma_medium, positions)
        slow = torch.pow(self.gamma_slow, positions)
        
        # Stack and project
        temporal_features = torch.stack([fast, medium, slow], dim=-1)  # (T, 3)
        temporal_encoding = self.temporal_proj(temporal_features)  # (T, 1)
        
        return temporal_encoding.expand(-1, self.n_embd)


class SemanticStream(ParallelStream):
    """Relationship-aware processing stream"""
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__(n_embd, vocab_size, n_layer, dropout)
        
        # Standard embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Relationship encoding
        self.relation_encoder = nn.Linear(n_embd * 2, n_embd)
        
    def embed(self, idx):
        """Create relationship-aware embeddings"""
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        
        # Add pairwise relationship encoding
        # Simple version: encode token pairs
        if T > 1:
            pairs = torch.cat([tok_emb[:, :-1], tok_emb[:, 1:]], dim=-1)
            relation_features = self.relation_encoder(pairs)
            
            # Pad to match sequence length
            relation_features = F.pad(relation_features, (0, 0, 0, 1))
            
            return tok_emb + pos_emb + relation_features
        else:
            return tok_emb + pos_emb


class FusionGate(nn.Module):
    """Learn to combine outputs from multiple streams"""
    
    def __init__(self, n_streams, n_embd):
        super().__init__()
        self.n_streams = n_streams
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(n_streams * n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_streams),
            nn.Softmax(dim=-1)
        )
        
        # Entropy regularization weight
        self.entropy_weight = 0.01
        
    def forward(self, stream_features):
        """
        Args:
            stream_features: List of (B, T, C) tensors
            
        Returns:
            fused_features: (B, T, C) combined features
            gate_weights: (B, T, n_streams) fusion weights
        """
        B, T, C = stream_features[0].shape
        
        # Concatenate all stream features
        combined = torch.cat(stream_features, dim=-1)  # (B, T, n_streams * C)
        
        # Compute gate weights
        gate_weights = self.gate_network(combined)  # (B, T, n_streams)
        
        # Weighted combination
        fused = torch.zeros(B, T, C, device=stream_features[0].device)
        for i, features in enumerate(stream_features):
            fused += gate_weights[:, :, i:i+1] * features
        
        return fused, gate_weights
    
    def compute_entropy_loss(self, gate_weights):
        """Encourage diverse gate usage (prevent collapse)"""
        # Average gate weights across batch and time
        avg_weights = gate_weights.mean(dim=[0, 1])  # (n_streams,)
        
        # Entropy: -Σ p log p
        entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum()
        
        # Maximize entropy (minimize negative entropy)
        return -self.entropy_weight * entropy


class KolosisV2(nn.Module):
    """
    Kolosis V2: Parallel Stream Architecture
    Each cognitive mechanism has direct supervision.
    """
    
    def __init__(self, vocab_size, n_embd, block_size, n_layer=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Four parallel streams
        self.symbol_stream = SymbolStream(vocab_size, n_embd, block_size, n_layer, dropout)
        self.concept_stream = ConceptStream(vocab_size, n_embd, block_size, n_layer, dropout)
        self.temporal_stream = TemporalStream(vocab_size, n_embd, block_size, n_layer, dropout)
        self.semantic_stream = SemanticStream(vocab_size, n_embd, block_size, n_layer, dropout)
        
        # Fusion gate
        self.fusion_gate = FusionGate(n_streams=4, n_embd=n_embd)
        
        # Final prediction head (for fused output)
        self.final_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, return_stream_outputs=False):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices (optional)
            return_stream_outputs: Whether to return individual stream outputs
            
        Returns:
            logits: (B, T, vocab_size) final predictions
            loss: Scalar loss (if targets provided)
            stream_info: Dict with stream outputs (if return_stream_outputs=True)
        """
        B, T = idx.shape
        
        # Process through each stream
        symbol_emb = self.symbol_stream.embed(idx)
        concept_emb = self.concept_stream.embed(idx)
        temporal_emb = self.temporal_stream.embed(idx)
        semantic_emb = self.semantic_stream.embed(idx)
        
        symbol_feat, symbol_logits = self.symbol_stream(symbol_emb)
        concept_feat, concept_logits = self.concept_stream(concept_emb)
        temporal_feat, temporal_logits = self.temporal_stream(temporal_emb)
        semantic_feat, semantic_logits = self.semantic_stream(semantic_emb)
        
        # Fuse streams
        fused_feat, gate_weights = self.fusion_gate([
            symbol_feat, concept_feat, temporal_feat, semantic_feat
        ])
        
        # Final prediction
        fusion_logits = self.final_head(fused_feat)
        
        # Compute loss if targets provided
        loss = None
        stream_losses = None
        if targets is not None:
            # Multi-head supervision: each stream + fusion
            stream_logits_list = [symbol_logits, concept_logits, temporal_logits, semantic_logits]
            
            stream_losses = []
            for stream_logits in stream_logits_list:
                stream_loss = F.cross_entropy(
                    stream_logits.view(B*T, self.vocab_size),
                    targets.view(B*T)
                )
                stream_losses.append(stream_loss)
            
            # Fusion loss
            fusion_loss = F.cross_entropy(
                fusion_logits.view(B*T, self.vocab_size),
                targets.view(B*T)
            )
            
            # Entropy regularization
            entropy_loss = self.fusion_gate.compute_entropy_loss(gate_weights)
            
            # Combined loss: 50% fusion + 50% average of streams + entropy
            total_loss = (
                0.5 * fusion_loss +
                0.5 * sum(stream_losses) / len(stream_losses) +
                entropy_loss
            )
            
            loss = total_loss
        
        if return_stream_outputs:
            stream_info = {
                'symbol_logits': symbol_logits,
                'concept_logits': concept_logits,
                'temporal_logits': temporal_logits,
                'semantic_logits': semantic_logits,
                'gate_weights': gate_weights,
                'stream_losses': stream_losses
            }
            return fusion_logits, loss, stream_info
        
        return fusion_logits, loss
    
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
    print("Testing Kolosis V2")
    
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'block_size': 32,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    model = KolosisV2(**config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    
    logits, loss, stream_info = model(x, y, return_stream_outputs=True)
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"\nGate weights (avg): {stream_info['gate_weights'].mean(dim=[0,1])}")
    print(f"Stream losses: {[l.item() for l in stream_info['stream_losses']]}")
    
    print("\nKolosis V2 ready!")
