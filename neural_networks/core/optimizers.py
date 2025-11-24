import numpy as np
from typing import Tuple, List, Optional, Dict, Any

class Optimizer:
    """Advanced optimizers for real-world training"""
    
    class SGD:
        def __init__(self, learning_rate=0.01, momentum=0.9):
            self.lr = learning_rate
            self.momentum = momentum
            self.velocity = {}
        
        def update(self, layer_id, weights, biases, dW, db):
            if layer_id not in self.velocity:
                self.velocity[layer_id] = {
                    'weights': np.zeros_like(weights),
                    'biases': np.zeros_like(biases)
                }
            
            # Momentum update
            self.velocity[layer_id]['weights'] = (self.momentum * self.velocity[layer_id]['weights'] + 
                                                 self.lr * dW)
            self.velocity[layer_id]['biases'] = (self.momentum * self.velocity[layer_id]['biases'] + 
                                                self.lr * db)
            
            return (weights - self.velocity[layer_id]['weights'], 
                   biases - self.velocity[layer_id]['biases'])
    
    class Adam:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.lr = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = {}  # First moment
            self.v = {}  # Second moment
            self.t = 0   # Time step
        
        def update(self, layer_id, weights, biases, dW, db):
            self.t += 1
            
            if layer_id not in self.m:
                self.m[layer_id] = {
                    'weights': np.zeros_like(weights),
                    'biases': np.zeros_like(biases)
                }
                self.v[layer_id] = {
                    'weights': np.zeros_like(weights),
                    'biases': np.zeros_like(biases)
                }
            
            # Update biased first moment estimate
            self.m[layer_id]['weights'] = self.beta1 * self.m[layer_id]['weights'] + (1 - self.beta1) * dW
            self.m[layer_id]['biases'] = self.beta1 * self.m[layer_id]['biases'] + (1 - self.beta1) * db
            
            # Update biased second raw moment estimate
            self.v[layer_id]['weights'] = self.beta2 * self.v[layer_id]['weights'] + (1 - self.beta2) * (dW ** 2)
            self.v[layer_id]['biases'] = self.beta2 * self.v[layer_id]['biases'] + (1 - self.beta2) * (db ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat_w = self.m[layer_id]['weights'] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m[layer_id]['biases'] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v[layer_id]['weights'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v[layer_id]['biases'] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            new_weights = weights - self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            new_biases = biases - self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
            
            return new_weights, new_biases