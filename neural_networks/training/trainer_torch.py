"""
PyTorch-accelerated Trainer for KOLOSIS.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import pickle

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

class Trainer:
    def __init__(self, model, tokenizer, device='cuda', learning_rate=3e-4):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train(self, text_data, epochs=100, batch_size=32, block_size=64, val_split=0.1):
        """Train the model on text data"""
        # Encode data
        data = self.tokenizer.encode(text_data)
        n = len(data)
        split_idx = int(n * (1 - val_split))
        
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Create datasets
        train_dataset = TextDataset(train_data, block_size)
        val_dataset = TextDataset(val_data, block_size) if len(val_data) > block_size else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training on {len(train_data)} tokens for {epochs} epochs...")
        print(f"Batch size: {batch_size}, Block size: {block_size}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                self.optimizer.zero_grad()
                logits, loss = self.model(xb, yb)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_dataset:
                self.model.eval()
                with torch.no_grad():
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)
                    val_loss = 0
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        _, loss = self.model(xb, yb)
                        val_loss += loss.item()
                    avg_val_loss = val_loss / len(val_loader)
                    self.history['val_loss'].append(avg_val_loss)
                    
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}")
                    
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f}s ({elapsed/epochs:.2f}s/epoch)")
        
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'tokenizer_chars': self.tokenizer.chars,
        }, path)
        print(f"Model saved to {path}")
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        # Restore tokenizer
        self.tokenizer.chars = checkpoint['tokenizer_chars']
        self.tokenizer.vocab_size = len(self.tokenizer.chars)
        self.tokenizer.stoi = {ch:i for i,ch in enumerate(self.tokenizer.chars)}
        self.tokenizer.itos = {i:ch for i,ch in enumerate(self.tokenizer.chars)}
        
        print(f"Model loaded from {path}")
