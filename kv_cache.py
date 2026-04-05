import torch
import torch.nn as nn
from torch.nn import functional as F
from train import Head, MultiHeadAttention, Block, GPTLanguageModel

#hyperparameters
batch_size = 32
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.2

#--------------------------------

class CacheHead(Head):
    
    def forward(self,x,kv_cache=None):

        B,T,C = x.shape

        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        if kv_cache['k'] is None:
            kv_cache['k'] = k
            kv_cache['v'] = v

        else:
            kv_cache['k'] = torch.cat([kv_cache['k'], k], dim=1)
            kv_cache['v'] = torch.cat([kv_cache['v'], v], dim=1)


        w = q @ kv_cache['k'].transpose(-2, -1) * self.head_size**-0.5 # (B,T,C) @ (B, C, T_new) -> (B,T,T_new)
        T_new = kv_cache['k'].shape[1]
        if T == 1:
            # no masking needed, single query attends to all cached keys
            w = torch.softmax(w, dim=-1)
        else:
            w = w.masked_fill(self.tril[:T, :T_new] == 0, float('-inf'))
            w = torch.softmax(w, dim=-1)
        out = w @ kv_cache['v']
        return out, kv_cache
    
class CachedMHA(MultiHeadAttention):
    
    def __init__(self, num_heads, head_size):
        super().__init__(num_heads, head_size)
        self.heads = nn.ModuleList([CacheHead(head_size=head_size) for _ in range(num_heads)])

    def forward(self,x, kv_cache=None):
        out = []
        for i,h in enumerate(self.heads):
            out_h, kv_cache[i] = h(x, kv_cache[i])
            out.append(out_h)
        out = torch.cat(out, dim=-1)
        out = self.proj(out)
        return out, kv_cache
    
class CachedBlock(Block):
    
    def __init__(self, n_embd, n_head):
        super().__init__(n_embd, n_head)
        head_size = n_embd // n_head
        self.sa = CachedMHA(n_head, head_size)

    def forward(self, x, kv_cache=None):
        sa_out, kv_cache = self.sa(self.ln1(x), kv_cache)
        x = x + sa_out
        x = x + self.ffd(self.ln2(x))
        return x, kv_cache
    
class CachedGPT(GPTLanguageModel):

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[CachedBlock(n_embd, n_head) for _ in range(n_layer)])

    def generate_with_cache(self, idx, max_new_tokens, kv_cache=None):

        kv_cache = [
        [{'k': None, 'v': None} for _ in range(n_head)]  # 8 heads
        for _ in range(n_layer)  # 6 layers
        ]

        for step in range(max_new_tokens):
            if step == 0:
                idx_cond = idx[:, -block_size:]
            else:
                idx_cond = idx[:, -1:]
            tok_emb = self.token_embedding_table(idx_cond)
            pos_emb = self.position_embedding_table(torch.arange(idx_cond.shape[1], device=device))
            x = tok_emb + pos_emb

            for i, block in enumerate(self.blocks):
                x, kv_cache[i] = block(x, kv_cache[i])

            x = self.ln(x)
            logits = self.out_proj(x)
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence

        return idx
    

    