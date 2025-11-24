import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.autograd.transformer import GPT
from neural_networks.autograd.engine import Value

class TestTransformer(unittest.TestCase):
    
    def test_gpt_forward(self):
        vocab_size = 10
        n_embd = 8
        n_head = 2
        n_layer = 1
        block_size = 5
        
        model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
        
        # Input sequence
        idx = [1, 2, 3]
        
        logits, loss = model(idx)
        
        # Check shape
        self.assertEqual(len(logits), 3) # T
        self.assertEqual(len(logits[0]), vocab_size) # Vocab
        
    def test_gpt_overfitting(self):
        # Can we learn a simple sequence?
        # "0 1 0 1" -> next is 0
        
        vocab_size = 2
        n_embd = 8
        n_head = 2
        n_layer = 1
        block_size = 4
        
        model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
        
        # Data: 0 1 0 1 -> 0
        x = [0, 1, 0, 1]
        y = [1, 0, 1, 0] # targets
        
        # Training loop
        for k in range(100):
            logits, loss = model(x, y)
            
            model.zero_grad()
            loss.backward()
            
            for p in model.parameters():
                p.data += -0.5 * p.grad
                
        # Check loss
        self.assertLess(loss.data, 1.0)
        
        # Check generation
        context = [0, 1]
        generated = model.generate(context, max_new_tokens=2)
        # Should be [0, 1, 0, 1]
        self.assertEqual(generated, [0, 1, 0, 1])

if __name__ == '__main__':
    unittest.main()
