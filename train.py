import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

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

torch.manual_seed(1337)
with open('input.txt', 'r') as f:
    text = f.read()

# all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mappings from characters to integers and vice versa
stoi = {ch:i for i, ch in enumerate(chars)} #string to integer
itos = {i:ch for i, ch in enumerate(chars)} #integer to string

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder:

# train test splits

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        w = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B,T,head_size) @ (B, head_size, T) -> (B,T,T)
        
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)

        v = self.value(x)
        out = w @ v
        return out
    

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)
    

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x
    

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.out_proj = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        #token embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        B, T, C = tok_emb.shape
        
        # 2. get position embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)

        # 3. add them together
        x = tok_emb + pos_emb # (B, T, n_embd)

        # 4. pass through blocks
        x = self.blocks(x) # (B, T, n_embd)

        # 5. apply final layernorm
        x = self.ln(x) # (B, T, n_embd)

        # 6. project to logits
        logits = self.out_proj(x) # (B, T, vocab_size)

        # 7. compute loss if targets given
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # same as bigram — crop context, forward pass, sample, append

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence

        return idx


if __name__ == '__main__':
    # Training loop
    model = GPTLanguageModel()
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters), desc="Training"):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save model and vocab
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
    }, 'gpt_checkpoint.pt')

    print("Model saved!")