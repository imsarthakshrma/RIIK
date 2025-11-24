import random
import math
from .engine import Value
from .nn import Module, Layer, MLP

def softmax(logits):
    # logits: list of Value
    # Numerical stability: subtract max
    max_val = max(l.data for l in logits)
    counts = [(logit - Value(max_val)).exp() for logit in logits]
    denominator = sum(counts)
    out = [c / denominator for c in counts]
    return out

class Head(Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size):
        self.key = Layer(n_embd, head_size, nonlin=False)
        self.query = Layer(n_embd, head_size, nonlin=False)
        self.value = Layer(n_embd, head_size, nonlin=False)
        self.head_size = head_size
        self.block_size = block_size

    def __call__(self, x):
        # x is (T, C) list of list of Values
        T = len(x)
        C = len(x[0])
        
        # (T, C) @ (C, H) -> (T, H)
        k = [self.key(xi) for xi in x] # list of T vectors of size H
        q = [self.query(xi) for xi in x]
        v = [self.value(xi) for xi in x]
        
        # (T, H) @ (H, T) -> (T, T)
        # wei[t, i] = q[t] . k[i]
        wei = []
        for t in range(T):
            row = []
            for i in range(T):
                # dot product
                dot = sum([q[t][j] * k[i][j] for j in range(self.head_size)])
                row.append(dot)
            wei.append(row)
            
        # Scale
        scale = self.head_size**-0.5
        wei = [[w * scale for w in row] for row in wei]
        
        # Mask (tril)
        for t in range(T):
            for i in range(T):
                if i > t:
                    wei[t][i] = Value(-1e9) # -inf
                    
        # Softmax
        wei = [softmax(row) for row in wei]
        
        # (T, T) @ (T, H) -> (T, H)
        # out[t] = sum(wei[t, i] * v[i])
        out = []
        for t in range(T):
            # weighted sum of v
            # v[i] is a vector of size H
            # wei[t][i] is a scalar
            
            # Initialize with zeros
            acc = [Value(0) for _ in range(self.head_size)]
            
            for i in range(T):
                w = wei[t][i]
                vi = v[i]
                # acc += w * vi
                acc = [a + w * val for a, val in zip(acc, vi)]
            out.append(acc)
            
        return out

    def parameters(self):
        return self.key.parameters() + self.query.parameters() + self.value.parameters()

class MultiHeadAttention(Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        self.heads = [Head(head_size, n_embd, block_size) for _ in range(num_heads)]
        self.proj = Layer(head_size * num_heads, n_embd, nonlin=False)

    def __call__(self, x):
        # x is (T, C)
        out = [h(x) for h in self.heads] # list of num_heads lists of (T, H)
        
        # Concatenate over H dimension
        # out[0] is (T, H)
        T = len(x)
        # concat[t] = out[0][t] + out[1][t] ...
        concat = []
        for t in range(T):
            row = []
            for h in range(len(self.heads)):
                row.extend(out[h][t])
            concat.append(row)
            
        # Project back
        out = [self.proj(row) for row in concat]
        return out

    def parameters(self):
        return [p for h in self.heads for p in h.parameters()] + self.proj.parameters()

class Block(Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = MLP(n_embd, [4 * n_embd, n_embd])
        # LayerNorm would be here, but skipping for simplicity/speed in scalar engine for now
        # or we can add a simple normalization if needed.

    def __call__(self, x):
        # x is (T, C)
        
        # Self-attention + Residual
        sa_out = self.sa(x)
        x = [[xi + si for xi, si in zip(x_row, s_row)] for x_row, s_row in zip(x, sa_out)]
        
        # Feed-forward + Residual
        ff_out = [self.ffwd(xi) for xi in x]
        x = [[xi + fi for xi, fi in zip(x_row, f_row)] for x_row, f_row in zip(x, ff_out)]
        
        return x

    def parameters(self):
        return self.sa.parameters() + self.ffwd.parameters()

class GPT(Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        self.token_embedding_table = Layer(vocab_size, n_embd, nonlin=False) # Using Layer as embedding lookup (one-hot * weights = lookup)
        # Ideally use Embedding class, but Layer works if we pass one-hot. 
        # Actually let's use our Embedding class if we can import it, or just implement lookup.
        # Let's use the Embedding class we made.
        from .nn import Embedding
        self.token_embedding_table = Embedding(vocab_size, n_embd)
        self.position_embedding_table = Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        self.lm_head = Layer(n_embd, vocab_size, nonlin=False)
        self.block_size = block_size

    def __call__(self, idx, targets=None):
        # idx is list of integers (T)
        T = len(idx)
        
        # Token embeddings
        tok_emb = self.token_embedding_table(idx) # (T, C)
        
        # Positional embeddings
        pos_idxs = list(range(T))
        pos_emb = self.position_embedding_table(pos_idxs) # (T, C)
        
        # Combine
        x = [[t + p for t, p in zip(te, pe)] for te, pe in zip(tok_emb, pos_emb)]
        
        # Blocks
        for block in self.blocks:
            x = block(x)
            
        # Final head
        logits = [self.lm_head(xi) for xi in x] # (T, vocab_size)
        
        loss = None
        if targets is not None:
            # targets is list of integers (T)
            # Cross entropy loss
            loss = Value(0)
            for t in range(T):
                # logits[t] is list of Values
                # targets[t] is integer index
                
                # Softmax
                probs = softmax(logits[t])
                
                # NLL
                p = probs[targets[t]]
                # Clip for stability
                if p.data < 1e-7:
                    p = p + Value(1e-7)
                loss -= p.log()
                
            loss = loss / Value(T)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is list of integers (context)
        for _ in range(max_new_tokens):
            # Crop context
            idx_cond = idx[-self.block_size:]
            
            # Forward
            logits, _ = self(idx_cond)
            
            # Get last time step
            logits_last = logits[-1]
            
            # Softmax
            probs = softmax(logits_last)
            
            # Sample (greedy for now)
            # probs_data = [p.data for p in probs]
            # next_idx = probs_data.index(max(probs_data))
            
            # Sample (weighted)
            probs_data = [p.data for p in probs]
            next_idx = random.choices(range(len(probs_data)), weights=probs_data, k=1)[0]
            
            idx.append(next_idx)
            
        return idx

    def parameters(self):
        return self.token_embedding_table.parameters() + \
               self.position_embedding_table.parameters() + \
               [p for b in self.blocks for p in b.parameters()] + \
               self.lm_head.parameters()
