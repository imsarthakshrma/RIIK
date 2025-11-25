"""
Hierarchical Concept Embeddings for KOLOSIS
Three-layer knowledge hierarchy: Symbol → Concept → Law
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalEmbedding(nn.Module):
    """
    Three-layer embedding hierarchy inspired by AI-Newton.
    
    Traditional: x = E_token(idx) + E_pos(pos)
    Kolosis: x = E_symbol(idx) + α·E_concept(idx) + β·E_law(idx) + E_pos(pos)
    
    Layers:
    1. Symbol: Raw tokens (character/word level)
    2. Concept: Mid-level abstractions (noun phrases, verb patterns)
    3. Law: High-level patterns (grammar rules, conversational structures)
    """
    
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        
        # Three embedding layers
        self.symbol_emb = nn.Embedding(vocab_size, n_embd)
        self.concept_emb = nn.Embedding(vocab_size, n_embd)
        self.law_emb = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(block_size, n_embd)
        
        # Learnable mixing weights (constrained to [0, 1] via sigmoid)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # Concept weight
        self.beta_logit = nn.Parameter(torch.tensor(-1.0))  # Law weight
        
        # Concept extraction network (learns to identify concepts from token sequences)
        self.concept_extractor = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd)
        )
        
    def forward(self, idx):
        """
        Args:
            idx: (B, T) token indices
            
        Returns:
            x: (B, T, C) hierarchical embeddings
        """
        B, T = idx.shape
        
        # Get base embeddings
        symbol = self.symbol_emb(idx)  # (B, T, C)
        concept = self.concept_emb(idx)  # (B, T, C)
        law = self.law_emb(idx)  # (B, T, C)
        pos = self.position_emb(torch.arange(T, device=idx.device))  # (T, C)
        
        # Apply concept extraction (refine concept embeddings based on context)
        concept_refined = self.concept_extractor(concept)
        
        # Compute mixing weights
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        
        # Hierarchical combination
        x = symbol + alpha * concept_refined + beta * law + pos
        
        return x
    
    def get_hierarchy_weights(self):
        """Get current mixing weights for analysis"""
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        
        return {
            'symbol_weight': 1.0,  # Always 1.0 (base)
            'concept_weight': alpha.item(),
            'law_weight': beta.item(),
        }


class ConceptClassifier(nn.Module):
    """
    Classify tokens as Intrinsic (static) or Dynamical (evolving).
    
    Intrinsic: Proper nouns, object identities (cached, cold memory)
    Dynamical: Verbs, states, references (fresh computation, hot memory)
    """
    
    def __init__(self, n_embd):
        super().__init__()
        
        # Classification network
        self.classifier = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd),  # Token + context
            nn.ReLU(),
            nn.Linear(n_embd, 1),
            nn.Sigmoid()  # P(intrinsic)
        )
        
        # Context summarizer (simple average pooling for now)
        # In production, could use attention-based summarization
        
    def forward(self, token_emb, context_emb):
        """
        Args:
            token_emb: (B, T, C) token embeddings
            context_emb: (B, T, C) context embeddings
            
        Returns:
            intrinsic_prob: (B, T, 1) probability of being intrinsic
        """
        # Concatenate token and context
        combined = torch.cat([token_emb, context_emb], dim=-1)  # (B, T, 2C)
        
        # Classify
        intrinsic_prob = self.classifier(combined)  # (B, T, 1)
        
        return intrinsic_prob
    
    def get_dual_processing(self, token_emb, intrinsic_prob):
        """
        Apply dual processing paths based on classification.
        
        Args:
            token_emb: (B, T, C) token embeddings
            intrinsic_prob: (B, T, 1) intrinsic probabilities
            
        Returns:
            processed: (B, T, C) processed embeddings
        """
        # Intrinsic path: minimal processing (cached)
        intrinsic_path = token_emb
        
        # Dynamical path: full processing (fresh computation)
        # In practice, this would involve more complex transformations
        dynamical_path = token_emb * 1.1  # Slight amplification
        
        # Weighted combination
        processed = intrinsic_prob * intrinsic_path + (1 - intrinsic_prob) * dynamical_path
        
        return processed


if __name__ == "__main__":
    # Test hierarchical embeddings
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    n_embd = 32
    block_size = 16
    
    # Create embedding module
    emb = HierarchicalEmbedding(vocab_size, n_embd, block_size)
    
    # Random input
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    x = emb(idx)
    print(f"Input shape: {idx.shape}")
    print(f"Output shape: {x.shape}")
    
    # Check hierarchy weights
    weights = emb.get_hierarchy_weights()
    print(f"\nHierarchy Weights:")
    print(f"  Symbol: {weights['symbol_weight']:.4f}")
    print(f"  Concept: {weights['concept_weight']:.4f}")
    print(f"  Law: {weights['law_weight']:.4f}")
    
    # Test concept classifier
    print("\n--- Concept Classifier ---")
    classifier = ConceptClassifier(n_embd)
    
    context = torch.randn(batch_size, seq_len, n_embd)
    intrinsic_prob = classifier(x, context)
    print(f"Intrinsic probabilities shape: {intrinsic_prob.shape}")
    print(f"Sample probabilities: {intrinsic_prob[0, :, 0].detach()}")
    
    # Test dual processing
    processed = classifier.get_dual_processing(x, intrinsic_prob)
    print(f"Processed embeddings shape: {processed.shape}")
