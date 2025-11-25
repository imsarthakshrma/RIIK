"""
Unit tests for Kolosis components
"""
import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.kolosis import (
    MultiScaleTemporalAttention,
    HierarchicalEmbedding,
    ConceptClassifier,
    PatternMemory,
    KolosisTransformer
)

class TestKolosisComponents(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.vocab_size = 100
        self.n_embd = 32
        self.block_size = 16
        self.n_head = 2
        
    def test_temporal_attention(self):
        """Test Multi-Scale Temporal Attention"""
        head_size = self.n_embd // self.n_head
        attn = MultiScaleTemporalAttention(head_size, self.n_embd, self.block_size)
        
        x = torch.randn(self.batch_size, self.seq_len, self.n_embd)
        out = attn(x)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, head_size))
        
        # Check temporal stats
        stats = attn.get_temporal_stats()
        self.assertIn('gamma_fast', stats)
        self.assertIn('gamma_medium', stats)
        self.assertIn('gamma_slow', stats)
        
        # Verify decay rates are in correct ranges
        self.assertGreater(stats['gamma_fast'], 0.6)
        self.assertLess(stats['gamma_fast'], 0.9)
        self.assertGreater(stats['gamma_slow'], 0.95)
        
    def test_hierarchical_embedding(self):
        """Test Hierarchical Embeddings"""
        emb = HierarchicalEmbedding(self.vocab_size, self.n_embd, self.block_size)
        
        idx = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        x = emb(idx)
        
        self.assertEqual(x.shape, (self.batch_size, self.seq_len, self.n_embd))
        
        # Check hierarchy weights
        weights = emb.get_hierarchy_weights()
        self.assertEqual(weights['symbol_weight'], 1.0)
        self.assertGreater(weights['concept_weight'], 0)
        self.assertLess(weights['concept_weight'], 1)
        
    def test_concept_classifier(self):
        """Test Intrinsic/Dynamical Classification"""
        classifier = ConceptClassifier(self.n_embd)
        
        token_emb = torch.randn(self.batch_size, self.seq_len, self.n_embd)
        context_emb = torch.randn(self.batch_size, self.seq_len, self.n_embd)
        
        intrinsic_prob = classifier(token_emb, context_emb)
        
        self.assertEqual(intrinsic_prob.shape, (self.batch_size, self.seq_len, 1))
        
        # Probabilities should be in [0, 1]
        self.assertTrue(torch.all(intrinsic_prob >= 0))
        self.assertTrue(torch.all(intrinsic_prob <= 1))
        
        # Test dual processing
        processed = classifier.get_dual_processing(token_emb, intrinsic_prob)
        self.assertEqual(processed.shape, token_emb.shape)
        
    def test_pattern_memory(self):
        """Test Pattern Memory System"""
        memory = PatternMemory(self.n_embd, max_patterns=10)
        
        context = torch.randn(self.batch_size, self.n_embd)
        attention_weights = torch.randn(self.batch_size, self.seq_len, self.seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Extract pattern
        memory.extract_pattern(context, attention_weights, loss=0.5)
        stats = memory.get_pattern_stats()
        self.assertEqual(stats['num_patterns'], 1)
        
        # Test pattern matching
        matches = memory.match_patterns(context[0])
        self.assertIsInstance(matches, list)
        
        # Test reasoning bias
        attention_scores = torch.randn(self.batch_size, self.seq_len, self.seq_len)
        biased = memory.apply_reasoning_bias(attention_scores, context)
        self.assertEqual(biased.shape, attention_scores.shape)
        
    def test_kolosis_transformer(self):
        """Test full Kolosis Transformer"""
        n_layer = 2
        model = KolosisTransformer(
            self.vocab_size, self.n_embd, self.n_head, n_layer, self.block_size
        )
        
        idx = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        logits, loss = model(idx, targets)
        
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
        
        # Test generation
        context = torch.randint(0, self.vocab_size, (1, 5))
        generated = model.generate(context, max_new_tokens=10)
        self.assertEqual(generated.shape[0], 1)
        self.assertEqual(generated.shape[1], 15)  # 5 + 10
        
        # Test cognitive stats
        stats = model.get_cognitive_stats()
        self.assertIn('hierarchy_weights', stats)
        self.assertIn('temporal_stats', stats)
        
    def test_kolosis_learning(self):
        """Test that Kolosis can learn a simple pattern"""
        n_layer = 1
        model = KolosisTransformer(
            self.vocab_size, self.n_embd, self.n_head, n_layer, self.block_size
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Simple pattern: always predict next token as current + 1 (mod vocab_size)
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        
        initial_loss = None
        final_loss = None
        
        # Train for a few steps
        for i in range(20):
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                initial_loss = loss.item()
            if i == 19:
                final_loss = loss.item()
        
        # Loss should decrease
        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()
