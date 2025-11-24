import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.autograd.nn import Embedding
from neural_networks.autograd.engine import Value

class TestNLP(unittest.TestCase):
    
    def test_tokenizer(self):
        text = "hello world"
        tokenizer = CharacterTokenizer()
        tokenizer.train(text)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        self.assertEqual(text, decoded)
        self.assertEqual(len(encoded), len(text))
        self.assertGreater(tokenizer.vocab_size, 0)
        
    def test_embedding(self):
        vocab_size = 10
        embed_dim = 4
        embed = Embedding(vocab_size, embed_dim)
        
        # Test single index
        idx = 5
        vec = embed(idx)
        self.assertEqual(len(vec), embed_dim)
        self.assertIsInstance(vec[0], Value)
        
        # Test batch of indices
        idxs = [1, 2, 3]
        vecs = embed(idxs)
        self.assertEqual(len(vecs), 3)
        self.assertEqual(len(vecs[0]), embed_dim)
        
    def test_embedding_learning(self):
        # Simple task: learn to embed index 0 to [1, 1] and index 1 to [-1, -1]
        embed = Embedding(2, 2)
        
        # Target vectors
        targets = {
            0: [1.0, 1.0],
            1: [-1.0, -1.0]
        }
        
        # Training loop
        for k in range(50):
            loss = Value(0)
            for idx, target in targets.items():
                vec = embed(idx)
                # MSE loss
                for v, t in zip(vec, target):
                    loss += (v - t)**2
            
            embed.zero_grad()
            loss.backward()
            
            for p in embed.parameters():
                p.data += -0.1 * p.grad
                
        # Check if learned
        vec0 = embed(0)
        vec1 = embed(1)
        
        self.assertAlmostEqual(vec0[0].data, 1.0, delta=0.1)
        self.assertAlmostEqual(vec1[0].data, -1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
