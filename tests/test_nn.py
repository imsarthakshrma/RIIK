import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.autograd.nn import MLP
from neural_networks.autograd.engine import Value

class TestNN(unittest.TestCase):
    
    def test_mlp_forward(self):
        # 2 inputs, hidden layer of 4, output of 1
        model = MLP(2, [4, 1])
        
        x = [Value(2.0), Value(3.0)]
        y = model(x)
        
        self.assertIsInstance(y, Value)
        
    def test_mlp_learning(self):
        # Simple binary classification
        model = MLP(2, [4, 4, 1])
        
        # Dataset
        xs = [
            [2.0, 3.0],
            [3.0, -1.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ]
        ys = [1.0, -1.0, -1.0, 1.0] # desired targets
        
        # Training loop
        for k in range(100):
            # forward
            ypred = [model(x) for x in xs]
            loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
            
            # backward
            model.zero_grad()
            loss.backward()
            
            # update
            for p in model.parameters():
                p.data += -0.05 * p.grad
                
        # Check if loss decreased
        self.assertLess(loss.data, 1.0)
        
        # Check predictions
        ypred = [model(x) for x in xs]
        for ygt, yout in zip(ys, ypred):
            # Signs should match
            self.assertEqual(1 if yout.data > 0 else -1, 1 if ygt > 0 else -1)

if __name__ == '__main__':
    unittest.main()
