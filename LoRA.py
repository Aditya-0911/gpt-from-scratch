import torch 
import torch.nn as nn
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

class LoRALinear(nn.Module):

    def __init__(self,in_features,out_features, r,lora_alpha):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha

        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False

        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):

        scale = self.lora_alpha / self.r
        return self.linear(x) + scale*x@self.A.T@self.B.T
    

class LoraHead(Head):

    def __init__(self,n_embd, head_size, r, lora_alpha):
        super().__init__(head_size)
        self.key = LoRALinear(n_embd, head_size, r, lora_alpha)
        self.query = LoRALinear(n_embd, head_size, r, lora_alpha)
        self.value = LoRALinear(n_embd, head_size, r, lora_alpha)

class LoraMHA(MultiHeadAttention):
    
    def __init__(self, n_embd, num_heads, head_size, r, lora_alpha):
        super().__init__(num_heads, head_size)
        self.heads = nn.ModuleList([LoraHead(n_embd, head_size, r, lora_alpha) for _ in range(num_heads)])

class LoraBlock(Block):

    def __init__(self, n_embd, n_head, r, lora_alpha):
        super().__init__(n_embd, n_head)
        head_size = n_embd // n_head
        self.sa = LoraMHA(n_embd, n_head, head_size, r, lora_alpha)


class LoraGPT(GPTLanguageModel):

    def __init__(self, r, lora_alpha):
        super().__init__()
        self.blocks = nn.Sequential(*[LoraBlock(n_embd, n_head, r, lora_alpha) for _ in range(n_layer)])


    