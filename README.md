# GPT From Scratch

A character-level GPT implementation built from scratch in PyTorch, following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) series. Trained on the TinyShakespeare dataset to generate Shakespeare-like text.

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

## Project Structure

```
gpt-from-scratch/
├── train.py          # Full training script with all model classes
├── inference.ipynb   # Load checkpoint and generate text locally
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
Open `inference.ipynb` and run all cells. Loads the checkpoint and generates 500 tokens of Shakespeare-like text.

---

## Key Learnings

- **Overfitting is visible early** — val loss plateaued at step 2500 while train loss kept dropping to 0.95. Early stopping or a larger dataset would help.
- **Pre-norm vs Post-norm** — modern GPT uses LayerNorm *before* attention (`x + Attn(LN(x))`), not after. More stable training.
- **`register_buffer` for `tril`** — the causal mask needs to move to the correct device automatically; registering it as a buffer handles this cleanly.
- **Shape tracking is everything** — every operation in the transformer has a specific `(B, T, C)` shape contract. Getting this right is the core implementation challenge.

---

## References

- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [TinyShakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)