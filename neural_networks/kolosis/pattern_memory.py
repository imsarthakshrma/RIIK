"""
Pattern Memory System for KOLOSIS
Implements cross-sequence learning and plausible reasoning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class AttentionPattern:
    """Single attention pattern with trigger condition and bias"""
    
    def __init__(self, trigger, bias, confidence=1.0):
        self.trigger = trigger  # Context embedding that triggers this pattern
        self.bias = bias  # Attention bias to apply
        self.confidence = confidence  # Confidence score
        self.success_count = 0
        self.failure_count = 0
        
    def update_confidence(self, success: bool, learning_rate: float = 0.1):
        """Update confidence based on success/failure"""
        if success:
            self.confidence = min(1.0, self.confidence * (1 + learning_rate))
            self.success_count += 1
        else:
            self.confidence = max(0.1, self.confidence * (1 - learning_rate))
            self.failure_count += 1


class PatternMemory(nn.Module):
    """
    Pattern memory system for plausible reasoning.
    Learns and applies successful attention patterns across sequences.
    """
    
    def __init__(self, n_embd, max_patterns=100, similarity_threshold=0.7):
        super().__init__()
        self.n_embd = n_embd
        self.max_patterns = max_patterns
        self.similarity_threshold = similarity_threshold
        self.patterns = deque(maxlen=max_patterns)
        
        # Pattern matching network
        self.pattern_matcher = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, 1),
            nn.Sigmoid()
        )
        
        # Bias generator
        self.bias_generator = nn.Linear(n_embd, 1)
        
    def match_patterns(self, context):
        """
        Match current context against stored patterns.
        
        Args:
            context: (B, C) context embedding
            
        Returns:
            match_scores: List of (pattern_idx, score) tuples
        """
        if len(self.patterns) == 0:
            return []
        
        matches = []
        for idx, pattern in enumerate(self.patterns):
            # Compute similarity
            trigger_tensor = pattern.trigger.to(context.device)
            
            # Expand context to match batch size of trigger if needed
            if context.dim() == 1:
                context = context.unsqueeze(0)
            
            # Concatenate and compute match score
            combined = torch.cat([context, trigger_tensor.expand(context.shape[0], -1)], dim=-1)
            score = self.pattern_matcher(combined).mean().item()
            
            if score > self.similarity_threshold:
                matches.append((idx, score * pattern.confidence))
                
        return matches
    
    def apply_reasoning_bias(self, attention_scores, context, lambda_weight=0.1):
        """
        Apply reasoning bias based on matched patterns.
        
        Args:
            attention_scores: (B, T, T) attention scores
            context: (B, C) context embedding
            lambda_weight: Weight for reasoning bias
            
        Returns:
            biased_scores: (B, T, T) attention scores with reasoning bias
        """
        matches = self.match_patterns(context)
        
        if len(matches) == 0:
            return attention_scores
        
        # Aggregate bias from matched patterns
        reasoning_bias = torch.zeros_like(attention_scores)
        
        for pattern_idx, match_score in matches:
            pattern = self.patterns[pattern_idx]
            # Apply pattern bias weighted by match score
            reasoning_bias += match_score * pattern.bias.to(attention_scores.device)
        
        # Add weighted reasoning bias
        biased_scores = attention_scores + lambda_weight * reasoning_bias
        
        return biased_scores
    
    def extract_pattern(self, context, attention_weights, loss):
        """
        Extract a new pattern from successful attention.
        
        Args:
            context: (B, C) context embedding
            attention_weights: (B, T, T) attention weights
            loss: Scalar loss value
        """
        # Only extract patterns from successful predictions (low loss)
        if loss > 1.0:  # Threshold for "success"
            return
        
        # Average context across batch
        avg_context = context.mean(dim=0) if context.dim() > 1 else context
        
        # Average attention weights across batch
        avg_attention = attention_weights.mean(dim=0) if attention_weights.dim() > 2 else attention_weights
        
        # Create new pattern
        new_pattern = AttentionPattern(
            trigger=avg_context.detach().cpu(),
            bias=avg_attention.detach().cpu(),
            confidence=1.0
        )
        
        # Add to memory
        self.patterns.append(new_pattern)
    
    def update_patterns(self, loss, threshold=1.0):
        """Update confidence of all patterns based on loss"""
        success = loss < threshold
        
        for pattern in self.patterns:
            pattern.update_confidence(success)
    
    def get_pattern_stats(self):
        """Get statistics about stored patterns"""
        if len(self.patterns) == 0:
            return {
                'num_patterns': 0,
                'avg_confidence': 0.0,
                'total_successes': 0,
                'total_failures': 0
            }
        
        return {
            'num_patterns': len(self.patterns),
            'avg_confidence': sum(p.confidence for p in self.patterns) / len(self.patterns),
            'total_successes': sum(p.success_count for p in self.patterns),
            'total_failures': sum(p.failure_count for p in self.patterns)
        }


if __name__ == "__main__":
    # Test pattern memory
    n_embd = 32
    batch_size = 2
    seq_len = 8
    
    # Create pattern memory
    memory = PatternMemory(n_embd, max_patterns=10)
    
    # Simulate context and attention
    context = torch.randn(batch_size, n_embd)
    attention_scores = torch.randn(batch_size, seq_len, seq_len)
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    print("Pattern Memory System Test")
    print(f"Initial patterns: {memory.get_pattern_stats()['num_patterns']}")
    
    # Extract a pattern from successful prediction
    memory.extract_pattern(context, attention_weights, loss=0.5)
    print(f"After extraction: {memory.get_pattern_stats()['num_patterns']}")
    
    # Try matching
    new_context = context + torch.randn_like(context) * 0.1  # Similar context
    matches = memory.match_patterns(new_context[0])
    print(f"Matches found: {len(matches)}")
    
    # Apply reasoning bias
    biased_scores = memory.apply_reasoning_bias(attention_scores, new_context)
    print(f"Biased scores shape: {biased_scores.shape}")
    
    # Update patterns
    memory.update_patterns(loss=0.3)
    stats = memory.get_pattern_stats()
    print(f"\nPattern Stats:")
    print(f"  Num patterns: {stats['num_patterns']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.4f}")
