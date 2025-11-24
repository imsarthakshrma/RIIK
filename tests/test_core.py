import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.core.network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Simple architecture for testing
        self.architecture = [
            (2, 4, 'relu'),
            (4, 1, 'sigmoid')
        ]
        self.model = NeuralNetwork(self.architecture, task='classification')
        
    def test_initialization(self):
        """Test if model initializes correctly"""
        self.assertEqual(len(self.model.layers), 2)
        self.assertEqual(self.model.layers[0].input_size, 2)
        self.assertEqual(self.model.layers[0].output_size, 4)
        self.assertEqual(self.model.layers[1].output_size, 1)
        
    def test_forward_shape(self):
        """Test if forward pass returns correct shape"""
        X = np.random.randn(5, 2)
        output = self.model.forward(X, training=False)
        self.assertEqual(output.shape, (5, 1))
        
    def test_xor_learning(self):
        """Test if model can learn XOR problem"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Train for enough epochs to learn
        self.model.fit(X, y, epochs=1000, batch_size=4, verbose=False, early_stopping=False)
        
        # Check predictions
        predictions = self.model.predict(X)
        binary_preds = (predictions > 0.5).astype(int)
        
        # Should match targets exactly for XOR
        np.testing.assert_array_equal(binary_preds, y)
        
    def test_regression_shape(self):
        """Test regression task shape"""
        arch = [(2, 4, 'relu'), (4, 1, 'linear')]
        model = NeuralNetwork(arch, task='regression')
        X = np.random.randn(5, 2)
        output = model.predict(X)
        self.assertEqual(output.shape, (5, 1))

if __name__ == '__main__':
    unittest.main()
