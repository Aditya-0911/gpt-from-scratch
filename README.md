# GPT From Scratch

A character-level GPT implementation built from scratch in PyTorch, following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) series. Trained on the TinyShakespeare dataset to generate Shakespeare-like text. Includes a BPE tokenizer, KV Cache inference optimization, LoRA fine-tuning, and Flash Attention — all implemented from scratch.

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

## BPE Tokenizer

Implemented a Byte Pair Encoding (BPE) tokenizer from scratch (`tokenizer.py`), following the same algorithm used in GPT-2/GPT-4.

### Why BPE?

Character-level tokenization (used in the base GPT above) treats every character as a token — simple but inefficient. BPE learns subword merges from the corpus, producing tokens that are more semantically meaningful and require fewer tokens to represent the same text. This reduces sequence length, lowering memory and compute requirements.

### Algorithm

1. Start with all 256 UTF-8 byte values as the base vocabulary
2. Count all adjacent pair frequencies in the corpus
3. Merge the most frequent pair into a new token (ID 256, 257, ...)
4. Repeat until target `vocab_size` is reached

### Benchmark

Trained on TinyShakespeare with `vocab_size=500`:

| Metric | Value |
|---|---|
| Base vocabulary | 256 (UTF-8 bytes) |
| Vocab size after training | 500 |
| Number of merges | 244 |
| Compression ratio | **1.89x** |
| Round-trip verified | ✅ `decode(encode(text)) == text` |

### Implementation

```python
from tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=500)

encoded = tokenizer.encode("What light through yonder window breaks?")
decoded = tokenizer.decode(encoded)
assert decoded == "What light through yonder window breaks?"
```

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

## Flash Attention

Implemented Flash Attention (Dao et al., 2022) from scratch in three stages: naive baseline, tiled PyTorch, and a Triton GPU kernel (`flash_attention.py`).

### Why Flash Attention?

Standard attention materializes the full N×N attention score matrix in GPU HBM (High Bandwidth Memory) — once after QKᵀ, again for softmax, again for multiplication with V. These repeated HBM round trips are the bottleneck, not the FLOPs.

Flash Attention is IO-aware: it tiles the computation into blocks that fit in SRAM (on-chip memory), fuses QKᵀ + softmax + V multiplication into a single kernel, and never writes the N×N matrix to HBM at all. Memory complexity drops from O(N²) to O(N).

```
Standard attention:  Q @ Kᵀ → write N×N to HBM → softmax → write to HBM → @ V → write output
Flash Attention:     tile into SRAM → fused kernel → write output once
```

### The Online Softmax Problem

Tiling creates a problem: softmax requires seeing all scores to compute the denominator `Σexp(xᵢ)`. You can't tile softmax naively without seeing all values first.

The solution is online softmax — maintain a running max `m` and running denominator `l` that get corrected as each new tile arrives:

```
# On seeing a new tile with scores s:
m_new = max(m, max(s))
correction = exp(m - m_new)          # rescale old accumulator
l = l * correction + sum(exp(s - m_new))
O = O * correction + exp(s - m_new) @ V_block
m = m_new
```

The correction factor `exp(m - m_new)` rescales previously accumulated values when a larger maximum is found — equivalent to `exp(old_max) / exp(new_max)`. This gives numerically identical results to standard softmax, computed entirely in tiles.

### Recomputation

Flash Attention discards the N×N attention matrix after the forward pass and recomputes it during backward from the stored Q, K, V tiles. This trades one extra forward pass (cheap) for O(N²) memory savings (large). This is a targeted application of gradient checkpointing to the attention matrix specifically.

### Implementation

Three implementations in increasing hardware fidelity:

**1. Naive attention** — baseline, materializes full N×N matrix:
```python
def naive_attention(Q, K, V):
    d_k = Q.shape[-1]
    score = Q @ K.transpose(-2, -1) / d_k**0.5
    attn = torch.softmax(score, dim=-1)
    return attn @ V
```

**2. Flash Attention PyTorch** — correct tiling algorithm, verifies online softmax:
```python
def flash_attention_forward(Q, K, V, tile_size=16):
    # tiles through Q blocks (outer) and K,V blocks (inner)
    # maintains m, run_denom, run_numer with correction factor
    # never materializes N×N matrix
```

**3. Flash Attention Triton kernel** — explicit GPU memory control:
```python
@triton.jit
def flash_attention_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, T, d_k: tl.constexpr, TILE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)          # each program handles one Q tile
    # tl.load → compute in SRAM → tl.store
    # inner loop over K, V tiles with online softmax correction
```

Each `program_id` handles one Q tile independently — the outer loop becomes parallelism across GPU cores. The inner loop over K, V tiles runs inside the kernel, keeping all intermediate values in SRAM.

### Correctness Verification

```python
Q = torch.randn(1, 1, 64, 32).cuda()
K = torch.randn(1, 1, 64, 32).cuda()
V = torch.randn(1, 1, 64, 32).cuda()

naive_out = naive_attention(Q, K, V)
triton_out = flash_attention_triton(Q, K, V)

print(torch.allclose(naive_out, triton_out, atol=1e-3))   # True
print(torch.max(torch.abs(naive_out - triton_out)))        # 4.17e-07
```

Triton output matches naive attention to 4e-7 — floating point precision only.

### Memory Benchmark

The core result. Memory measured on T4 GPU (Kaggle):

| Sequence Length | Naive Attention | Flash Attention (Triton) |
|---|---|---|
| T=1024 | 34.7 MB | 26.7 MB |
| T=2048 | 59.7 MB | 27.7 MB |
| T=4096 | 157.7 MB | 29.7 MB |

Naive attention memory grows quadratically — it roughly doubles every time T doubles. Flash Attention stays flat at ~28 MB regardless of sequence length. At T=4096 the difference is already 5x. Extrapolated to T=32768 (modern LLM context lengths), naive attention would require ~10 GB for the attention matrix alone; Flash Attention stays constant.

**This is why Flash Attention exists — not primarily for speed, but to make long-context attention memory-feasible.**

### A Note on Speed

The PyTorch tiled implementation is slower than naive attention — Python for loops launch separate CUDA kernels per iteration, adding overhead that outweighs the memory savings. The Triton kernel also doesn't outperform naive attention at short sequences because the T4's naive attention is already well-optimized at small N.

Real-world Flash Attention speedups (2-4x) require a production-grade fused CUDA kernel (as in the original paper) that eliminates all kernel launch overhead and maximizes SRAM utilization. This implementation demonstrates the algorithm and memory behavior correctly; the speed story requires the full CUDA implementation.

---

## Project Structure

```
gpt-from-scratch/
├── train.py                  # Full training script with all model classes
├── tokenizer.py              # BPE Tokenizer — train, encode, decode from scratch
├── kv_cache.py               # KV Cache — CachedHead, CachedMHA, CachedBlock, CachedGPT
├── lora.py                   # LoRA fine-tuning — LoRALinear, LoRAHead, LoRAMHA, LoRABlock, LoRAGPT
├── Flash_Attn.ipynb          # Flash Attention — naive baseline + PyTorch tiled implementation
├── triton-tutorials.ipynb    # Triton tutorials (vector add, fused softmax) + Flash Attention Triton kernel
├── inference.ipynb           # Naive vs cached generation benchmark
├── gpt-dev.ipynb             # Development notebook (experimentation)
└── README.md
```

---

## Usage

### Training (requires GPU)
```bash
python train.py
```
Downloads TinyShakespeare automatically, trains for 5000 steps, saves `gpt_checkpoint.pt`.

### BPE Tokenizer (CPU)
```python
from tokenizer import BPETokenizer
tokenizer = BPETokenizer()
tokenizer.train(open('input.txt').read(), vocab_size=500)
print(tokenizer.encode("To be or not to be"))
print(tokenizer.decode(tokenizer.encode("To be or not to be")))
```

### Inference (CPU)
Open `inference.ipynb` and run all cells. Benchmarks both naive and KV-cached generation over 500 tokens.

### LoRA Fine-Tuning (CPU)
```python
from lora import LoRAGPT
lora_model = LoRAGPT(r=8, lora_alpha=16)
lora_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# freeze all, unfreeze A and B, then fine-tune
```

### Flash Attention (requires GPU)
Open `Flash_Attn.ipynb` for the naive baseline and PyTorch tiled implementation with correctness verification and memory benchmarks.

Open `triton-tutorials.ipynb` for the Triton kernel implementation — includes the two introductory Triton tutorials (vector add, fused softmax) followed by the full Flash Attention Triton kernel with correctness check and memory benchmark.

---

## Key Learnings

- **Overfitting is visible early** — val loss plateaued at step 2500 while train loss kept dropping to 0.95. Early stopping or a larger dataset would help.
- **Pre-norm vs Post-norm** — modern GPT uses LayerNorm *before* attention (`x + Attn(LN(x))`), not after. More stable training.
- **`register_buffer` for `tril`** — the causal mask needs to move to the correct device automatically; registering it as a buffer handles this cleanly.
- **Shape tracking is everything** — every operation in the transformer has a specific `(B, T, C)` shape contract. Getting this right is the core implementation challenge.
- **BPE encode order matters** — merges must be applied in the same order they were learned during training. Using `min(stats, key=lambda p: merges.get(p, inf))` ensures this.
- **Off-by-one in merge loop** — incrementing `i` twice (once inside the merge branch, once unconditionally) causes tokens to be silently dropped. Always increment inside each branch separately.
- **KV Cache is inference-only** — training uses full parallel attention over the entire sequence; caching only makes sense when generating token by token.
- **Q is never cached** — only K and V are reused. Q represents the current query position and must always be recomputed fresh.
- **LoRA initializes B to zero** — so the LoRA contribution at the start of fine-tuning is zero, preserving the pretrained model's behavior. Only A is randomly initialized.
- **`strict=False` for partial loading** — when loading pretrained weights into a LoRA model, `strict=False` allows the extra A and B parameters to be randomly initialized while all other weights are loaded from the checkpoint.
- **HBM vs SRAM is the Flash Attention insight** — the bottleneck in standard attention is memory bandwidth, not compute. Tiling into SRAM eliminates repeated HBM round trips.
- **Online softmax requires a correction factor** — when tiling softmax, each new tile may reveal a larger maximum. The correction `exp(old_max - new_max)` rescales the previous accumulator to keep results numerically identical to full softmax.
- **PyTorch can't control SRAM** — Python for loops over PyTorch ops launch separate CUDA kernels per iteration. Real Flash Attention speedup requires a fused Triton or CUDA kernel that keeps the entire tiled computation in a single kernel launch.
- **Memory is the primary win** — Flash Attention's O(N) memory vs O(N²) is the real story. Speed improvements are secondary and require production-grade CUDA kernels to realize fully.
- **`tl.constexpr` for shapes in Triton** — any value used as a tensor shape inside a Triton kernel must be declared `tl.constexpr` in the kernel signature, or `tl.arange` will error at compile time.

---

## References

- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Andrej Karpathy — Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Neural Machine Translation of Rare Words with Subword Units (BPE paper)](https://arxiv.org/abs/1508.07909)
- [TinyShakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- [Triton Language Documentation](https://triton-lang.org/main/getting-started/tutorials/index.html)