import torch
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.autograd.transformer_torch import GPT
from neural_networks.training.trainer_torch import Trainer

def main():
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load larger dataset
    data_path = 'data/science.txt'
    with open(data_path, 'r') as f:
        text = f.read()
        
    # For demo, let's use a larger text
    text = text * 20  # Repeat to have more data
    print(f"Loaded {len(text)} characters of scientific knowledge.")
    
    # Tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Model (larger since we have speed now!)
    n_embd = 64
    n_head = 4
    n_layer = 4
    block_size = 32
    vocab_size = tokenizer.vocab_size
    
    print(f"Initializing GPT model (n_embd={n_embd}, n_head={n_head}, n_layer={n_layer})...")
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    
    # Train
    trainer = Trainer(model, tokenizer, device=device, learning_rate=3e-4)
    trainer.train(text, epochs=500, batch_size=16, block_size=block_size)
    trainer.save_checkpoint('models/science_model_torch.pt')
    
    # Generate
    print("\nGenerating scientific facts...")
    model.eval()
    
    context_str = "The "
    context = tokenizer.encode(context_str)
    context_tensor = torch.tensor([context], dtype=torch.long, device=device)
    
    generated_idx = model.generate(context_tensor, max_new_tokens=100, temperature=0.8, top_k=10)
    generated_text = tokenizer.decode(generated_idx[0].tolist())
    
    print(f"\nInput: {context_str}")
    print(f"Output: {generated_text}")

if __name__ == "__main__":
    main()
