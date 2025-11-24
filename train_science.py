import os
from neural_networks.nlp.tokenizer import CharacterTokenizer
from neural_networks.autograd.transformer import GPT
from neural_networks.training.trainer import Trainer

def main():
    # 1. Load Data
    data_path = 'data/science.txt'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    with open(data_path, 'r') as f:
        text = f.read()
        
    print(f"Loaded {len(text)} characters of scientific knowledge.")
    
    # 2. Tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Chars: {''.join(tokenizer.chars)}")
    
    # 3. Model
    # Small model for scalar autograd speed
    n_embd = 8
    n_head = 2
    n_layer = 1
    block_size = 8
    vocab_size = tokenizer.vocab_size
    
    print(f"Initializing GPT model (n_embd={n_embd}, n_head={n_head}, n_layer={n_layer})...")
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
    
    # 4. Train
    trainer = Trainer(model, tokenizer, learning_rate=0.1)
    trainer.train(text, epochs=50, batch_size=4, block_size=block_size, save_path='models/science_model.pkl')
    
    # 5. Generate
    print("\nGenerating scientific facts...")
    context_str = "The "
    context = tokenizer.encode(context_str)
    
    generated_idx = model.generate(context, max_new_tokens=50)
    generated_text = tokenizer.decode(generated_idx)
    
    print(f"\nInput: {context_str}")
    print(f"Output: {generated_text}")

if __name__ == "__main__":
    main()
