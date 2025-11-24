import random
import time
import pickle
import os
from ..autograd.engine import Value

class Trainer:
    def __init__(self, model, tokenizer, learning_rate=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.history = {'loss': []}

    def train(self, text_data, epochs=100, batch_size=4, block_size=8, save_path=None):
        """
        Train the model on text data.
        """
        # Encode data
        data = self.tokenizer.encode(text_data)
        n = len(data)
        
        print(f"Training on {n} tokens for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Sample a batch
            ix = [random.randint(0, n - block_size - 1) for _ in range(batch_size)]
            x = [data[i:i+block_size] for i in ix]
            y = [data[i+1:i+block_size+1] for i in ix]
            
            # Forward pass
            total_loss = Value(0)
            for i in range(batch_size):
                _, loss = self.model(x[i], y[i])
                total_loss += loss
            
            mean_loss = total_loss / Value(batch_size)
            
            # Backward pass
            self.model.zero_grad()
            mean_loss.backward()
            
            # Update weights (SGD)
            for p in self.model.parameters():
                p.data -= self.lr * p.grad
            
            # Track loss
            current_loss = mean_loss.data
            self.history['loss'].append(current_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {current_loss:.4f}")
                
        print(f"Training completed in {time.time() - start_time:.2f}s")
        
        if save_path:
            self.save_checkpoint(save_path)
            
    def save_checkpoint(self, path):
        """Save model checkpoint (weights only for now since Value objects aren't easily picklable with graph)"""
        # We'll just save the parameters data
        params = [p.data for p in self.model.parameters()]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'params': params,
                'vocab_size': len(self.tokenizer.chars),
                'chars': self.tokenizer.chars,
                'history': self.history
            }, f)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
            
        # Restore tokenizer
        self.tokenizer.chars = checkpoint['chars']
        self.tokenizer.vocab_size = len(self.tokenizer.chars)
        self.tokenizer.stoi = {ch:i for i,ch in enumerate(self.tokenizer.chars)}
        self.tokenizer.itos = {i:ch for i,ch in enumerate(self.tokenizer.chars)}
        
        # Restore weights
        params_data = checkpoint['params']
        model_params = self.model.parameters()
        
        if len(params_data) != len(model_params):
            print(f"Warning: Parameter count mismatch. Saved: {len(params_data)}, Model: {len(model_params)}")
            return
            
        for p, data in zip(model_params, params_data):
            p.data = data
            
        self.history = checkpoint.get('history', {'loss': []})
        print(f"Model loaded from {path}")
