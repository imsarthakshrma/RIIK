import time
from neural_networks.autograd.transformer import GPT
from neural_networks.nlp.tokenizer import CharacterTokenizer

def profile():
    print("Profiling KOLOSIS (Scalar Autograd)...")
    
    # Setup
    vocab_size = 65
    n_embd = 16
    n_head = 2
    n_layer = 1
    block_size = 8
    
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
    
    # Dummy input
    x = [1, 2, 3, 4, 5, 6, 7, 8] # (T=8)
    y = [2, 3, 4, 5, 6, 7, 8, 9]
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        model(x, y)
        
    # Benchmark
    print("Running benchmark (10 iterations)...")
    start = time.time()
    for _ in range(10):
        logits, loss = model(x, y)
        model.zero_grad()
        loss.backward()
        
    end = time.time()
    avg_time = (end - start) / 10
    print(f"Average time per step: {avg_time:.4f}s")

if __name__ == "__main__":
    profile()
