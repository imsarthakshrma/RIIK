import numpy as np
from typing import Tuple
from .activation import Activation



class Layer:
    """Optimized layer implementation for production use"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu', 
                 dropout_rate: float = 0.0, batch_norm: bool = False):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.training = True
        
        # Initialize weights with proper scaling
        if activation in ['relu', 'leaky_relu']:
            # He initialization
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif activation == 'sigmoid':
            # Xavier initialization
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            # General Xavier initialization
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        
        self.biases = np.zeros((1, output_size))
        
        # Batch normalization parameters
        if batch_norm:
            self.gamma = np.ones((1, output_size))
            self.beta = np.zeros((1, output_size))
            self.running_mean = np.zeros((1, output_size))
            self.running_var = np.ones((1, output_size))
        
        # Cache for backpropagation
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['input'] = x
        
        # Linear transformation
        z = np.dot(x, self.weights) + self.biases
        self.cache['z'] = z
        
        # Batch normalization
        if self.batch_norm:
            if self.training:
                mean = np.mean(z, axis=0, keepdims=True)
                var = np.var(z, axis=0, keepdims=True)
                
                # Update running statistics
                self.running_mean = 0.9 * self.running_mean + 0.1 * mean
                self.running_var = 0.9 * self.running_var + 0.1 * var
                
                z_norm = (z - mean) / np.sqrt(var + 1e-8)
            else:
                z_norm = (z - self.running_mean) / np.sqrt(self.running_var + 1e-8)
            
            z = self.gamma * z_norm + self.beta
            self.cache['z_norm'] = z_norm
            self.cache['mean'] = mean if self.training else self.running_mean
            self.cache['var'] = var if self.training else self.running_var
        
        # Activation
        if self.activation == 'relu':
            a = Activation.relu(z)
        elif self.activation == 'leaky_relu':
            a = Activation.leaky_relu(z)
        elif self.activation == 'sigmoid':
            a = Activation.sigmoid(z)
        elif self.activation == 'tanh':
            a = Activation.tanh(z)
        elif self.activation == 'softmax':
            a = Activation.softmax(z)
        else:  # linear
            a = z
        
        self.cache['a'] = a
        
        # Dropout
        if self.training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
            a = a * dropout_mask
            self.cache['dropout_mask'] = dropout_mask
        
        return a
    
    def backward(self, da: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.cache['input'].shape[0]
        
        # Dropout backward
        if self.training and self.dropout_rate > 0:
            da = da * self.cache['dropout_mask']
        
        # Activation backward
        if self.activation == 'relu':
            dz = da * Activation.relu_derivative(self.cache['z'])
        elif self.activation == 'leaky_relu':
            dz = da * Activation.leaky_relu_derivative(self.cache['z'])
        elif self.activation == 'sigmoid':
            dz = da * Activation.sigmoid_derivative(self.cache['z'])
        elif self.activation == 'tanh':
            dz = da * Activation.tanh_derivative(self.cache['z'])
        else:  # linear or softmax
            dz = da
        
        # Batch normalization backward
        dgamma = dbeta = None
        if self.batch_norm:
            dgamma = np.sum(dz * self.cache['z_norm'], axis=0, keepdims=True)
            dbeta = np.sum(dz, axis=0, keepdims=True)
            
            dz_norm = dz * self.gamma
            dvar = np.sum(dz_norm * (self.cache['z'] - self.cache['mean']) * -0.5 * 
                         np.power(self.cache['var'] + 1e-8, -1.5), axis=0, keepdims=True)
            dmean = np.sum(dz_norm * -1 / np.sqrt(self.cache['var'] + 1e-8), axis=0, keepdims=True) + \
                   dvar * np.mean(-2 * (self.cache['z'] - self.cache['mean']), axis=0, keepdims=True)
            
            dz = dz_norm / np.sqrt(self.cache['var'] + 1e-8) + \
                dvar * 2 * (self.cache['z'] - self.cache['mean']) / m + \
                dmean / m
        
        # Linear backward
        dW = np.dot(self.cache['input'].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        da_prev = np.dot(dz, self.weights.T)
        
        return da_prev, dW, db, dgamma, dbeta
    
    def set_training(self, training: bool):
        self.training = training