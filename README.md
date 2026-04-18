# GPT From Scratch

A character-level GPT implementation built from scratch in PyTorch, following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) series. Trained on the TinyShakespeare dataset to generate Shakespeare-like text. Includes KV Cache inference optimization and LoRA fine-tuning, both implemented from scratch.

---

## Architecture

A decoder-only transformer (GPT-style) with the following components, all implemented from scratch:

- **Token + Positional Embeddings** — learned embeddings for both token identity and position
- **Multi-Head Self-Attention** — scaled dot-product attention with causal masking via `tril`
- **Feed-Forward Network** — 4x expansion with ReLU, projected back to `n_embd`
- **Residual Connections + Pre-LayerNorm** — applied before attention and FFN for training stability
- **Stacked Transformer Blocks** — 6 layers deep

```
Input (B, T) → Token Emb + Pos Emb → [Block x 6] → LayerNorm → Linear → Logits (B, T, vocab_size)
```

Each Block:
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| `n_embd` | 512 |
| `n_head` | 8 |
| `n_layer` | 6 |
| `block_size` | 256 |
| `batch_size` | 32 |
| `dropout` | 0.2 |
| `max_iters` | 5000 |
| `learning_rate` | 1e-3 |
| Optimizer | AdamW |

---

## Training

Trained on a T4 GPU (Kaggle) for ~38 minutes on the [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset (~1M characters).

### Loss Curve

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 0 | 4.3219 | 4.3214 |
| 500 | 2.2350 | 2.3068 |
| 1000 | 1.7940 | 1.9444 |
| 1500 | 1.5289 | 1.7181 |
| 2000 | 1.3785 | 1.6250 |
| **2500** | **1.2880** | **1.5517** ← best |
| 3000 | 1.2110 | 1.5751 |
| 3500 | 1.1304 | 1.5985 |
| 4000 | 1.0510 | 1.6567 |
| 4500 | 0.9529 | 1.7458 |

**Best val loss: 1.5517 at step 2500.** Model begins overfitting after step 2500 — train loss continues dropping while val loss rises. Early stopping at step 2500 would be the optimal checkpoint.

---

## Sample Output

Generated at inference (500 tokens, CPU):

```
Cominius, and be project'st face, not to be general:
Let me trust I know thee report of thy party,
And make the drownfall of Claudio's valour.
Rage, that doth frown and hot, and on earth th't
Most oble true sins, it may not stumblector view;
For thou love'st know thou fonfust: mark me from him.

GLOUCESTER:
Well met unborn, and be your wife should call you
of withose Edward proming with us.

BISHOP OF ELY:
Thou wilt had lived morrow, so were so at lawful,
With death's hopes from a beggast of blo
```

The model has learned character names (GLOUCESTER, BISHOP OF ELY, KING HENRY VI), dialogue formatting, Shakespearean vocabulary, and rough sentence structure — purely from character-level prediction.

---

## KV Cache Inference Optimization

Implemented KV Cache as an inference optimization on top of the trained model (`kv_cache.py`), without modifying `train.py`.

### Why KV Cache?

In naive autoregressive generation, every new token triggers a full forward pass over the entire growing sequence — recomputing K and V matrices for all previous tokens from scratch at every step. This is pure wasted computation: the context hasn't changed, only a new token has been appended.

KV Cache eliminates this by storing previously computed K and V tensors per layer per head, and appending only the new token's K and V at each step. Q is always computed fresh since it represents the current query position.

```
Naive (step n):   recompute K, V for all n tokens → O(n) work per step
KV Cache (step n): reuse cached K, V, compute only for token n → O(1) work per step
```

### Benchmark

Measured on CPU, generating 500 tokens from the same trained checkpoint:

| Method | Time | Tokens/sec |
|--------|------|------------|
| Naive autoregressive | 51.86s | 9.6 tok/s |
| KV Cache | 14.61s | 34.2 tok/s |
| **Speedup** | | **3.55x** |

Speedup scales with sequence length — longer sequences yield larger gains since the proportion of redundant computation grows.

### Implementation

Subclassed `Head`, `MultiHeadAttention`, `Block`, and `GPTLanguageModel` to thread the cache through the forward pass cleanly:

- **`CachedHead`** — computes K, V only for the new token, appends to cache, runs attention over full cached history
- **`CachedMHA`** — passes each head its own cache slice, collects updated caches
- **`CachedBlock`** — threads cache through self-attention, unchanged FFN
- **`CachedGPT`** — initializes cache as `kv_cache[layer][head] = {'k': tensor, 'v': tensor}`, runs manual block loop during generation

No changes to `train.py` — the optimization is entirely inference-side.

---

## LoRA Fine-Tuning

Implemented LoRA (Low-Rank Adaptation) from scratch to fine-tune the pretrained GPT on the King James Bible (`lora.py`), without modifying `train.py`.

### Why LoRA?

Full fine-tuning updates all 19.7M parameters — expensive and prone to catastrophic forgetting. LoRA freezes the pretrained weights and injects two small trainable matrices A and B alongside each attention projection. The weight update becomes:

```
output = W·x + (B·A)·x · (alpha/r)
```

Where W is frozen, and only A (r × d_in) and B (d_out × r) are trained. Since weight updates during fine-tuning have low intrinsic rank, this approximation loses very little expressivity while dramatically reducing trainable parameters.

### Parameter Efficiency

| | Full Fine-Tuning | LoRA (r=8) |
|---|---|---|
| Trainable parameters | 19,776,577 | 663,552 |
| % of total | 100% | **3.36%** |
| Layers trained | All | Key, Query, Value projections only |

### Fine-Tuning Results

Fine-tuned on King James Bible for 500 steps on CPU (~13 minutes):

| | Base GPT | LoRA Fine-tuned |
|---|---|---|
| Dataset | TinyShakespeare | King James Bible |
| Val loss | 1.5517 | 2.4000 |
| Steps | 5000 | 500 |
| Trainable params | 19.7M | 663K |
| Learning rate | 1e-3 | 1e-4 |

Val loss dropped from 5.52 → 2.40 over 500 steps, showing meaningful domain adaptation with only 3.36% of parameters trained.

### Implementation

Subclassed `Head`, `MultiHeadAttention`, `Block`, and `GPTLanguageModel` to inject LoRA layers without touching `train.py`:

- **`LoRALinear`** — wraps a frozen `nn.Linear` and adds trainable A and B matrices; forward pass is `linear(x) + (alpha/r) * x @ A.T @ B.T`
- **`LoRAHead`** — replaces key, query, value projections with `LoRALinear`
- **`LoRAMHA`** — uses `LoRAHead` instances, inherits forward pass unchanged
- **`LoRABlock`** — replaces self-attention with `LoRAMHA`, FFN unchanged
- **`LoRAGPT`** — loads pretrained checkpoint, freezes all params, unfreezes only A and B

---

## Project Structure

```
gpt-from-scratch/
├── train.py          # Full training script with all model classes
├── kv_cache.py       # KV Cache — CachedHead, CachedMHA, CachedBlock, CachedGPT
├── lora.py           # LoRA fine-tuning — LoRALinear, LoRAHead, LoRAMHA, LoRABlock, LoRAGPT
├── inference.ipynb   # Naive vs cached generation benchmark
├── gpt-dev.ipynb     # Development notebook (experimentation)
└── README.md
```

---

## Usage

### Training (requires GPU)
```bash
python train.py
```
Downloads TinyShakespeare automatically, trains for 5000 steps, saves `gpt_checkpoint.pt`.

### Inference (CPU)
Open `inference.ipynb` and run all cells. Benchmarks both naive and KV-cached generation over 500 tokens.

### LoRA Fine-Tuning (CPU)
```python
from lora import LoRAGPT
lora_model = LoRAGPT(r=8, lora_alpha=16)
lora_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# freeze all, unfreeze A and B, then fine-tune
```

---

## Key Learnings

- **Overfitting is visible early** — val loss plateaued at step 2500 while train loss kept dropping to 0.95. Early stopping or a larger dataset would help.
- **Pre-norm vs Post-norm** — modern GPT uses LayerNorm *before* attention (`x + Attn(LN(x))`), not after. More stable training.
- **`register_buffer` for `tril`** — the causal mask needs to move to the correct device automatically; registering it as a buffer handles this cleanly.
- **Shape tracking is everything** — every operation in the transformer has a specific `(B, T, C)` shape contract. Getting this right is the core implementation challenge.
- **KV Cache is inference-only** — training uses full parallel attention over the entire sequence; caching only makes sense when generating token by token.
- **Q is never cached** — only K and V are reused. Q represents the current query position and must always be recomputed fresh.
- **LoRA initializes B to zero** — so the LoRA contribution at the start of fine-tuning is zero, preserving the pretrained model's behavior. Only A is randomly initialized.
- **`strict=False` for partial loading** — when loading pretrained weights into a LoRA model, `strict=False` allows the extra A and B parameters to be randomly initialized while all other weights are loaded from the checkpoint.

---

## References

- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [TinyShakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)