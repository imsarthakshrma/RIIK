class CharacterTokenizer:
    """
    Simple character-level tokenizer.
    """
    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def train(self, text):
        """Build vocabulary from text"""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}

    def encode(self, text):
        """Convert string to list of integers"""
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        """Convert list of integers to string"""
        return ''.join([self.itos[i] for i in ids])
