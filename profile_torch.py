import time
import torch
from neural_networks.autograd.transformer_torch import GPT

def profile_torch():
    print("Profiling KOLOSIS (PyTorch Accelerated)...")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    vocab_size = 65
    n_embd = 16
    n_head = 2
    n_layer = 1
    block_size = 8
    
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    
    # Dummy input (batch size 4)
    x = torch.randint(0, vocab_size, (4, block_size), device=device)
    y = torch.randint(0, vocab_size, (4, block_size), device=device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        logits, loss = model(x, y)
        
    # Benchmark
    print("Running benchmark (10 iterations)...")
    start = time.time()
    for _ in range(10):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
    end = time.time()
    avg_time = (end - start) / 10
    print(f"Average time per step: {avg_time:.4f}s")
    print(f"Speedup vs scalar: {0.9072 / avg_time:.1f}x")

if __name__ == "__main__":
    profile_torch()
