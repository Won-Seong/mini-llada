import torch.nn as nn
import torch.nn.functional as F
from rope import precompute_freqs_cis, apply_rotary_emb
from transformers import BertForMaskedLM

def get_pretrained_bert_model(model_name="klue/roberta-large"):
    model = BertForMaskedLM.from_pretrained(model_name)
    return model

class BERT_Wrapper(nn.Module):
    def __init__(self, bert_model: BertForMaskedLM):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, x):
        outputs = self.bert_model(input_ids=x)
        return outputs.logits

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n_heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        freqs_cis = freqs_cis.to(x.device)
        q, k = apply_rotary_emb(q, k, freqs_cis[:L])

        # LLaDA: is_causal=False (Bidirectional Attention)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = attn_output.transpose(1, 2).reshape(B, L, D)
        x = self.o_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, heads, intermediate_size):
        super().__init__()
        self.mha = Attention(dim, heads) # Multi-Head Attention
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        # Feed-Forward Network with Gated Linear Unit
        self.gate_up_proj = nn.Linear(dim, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x, freqs_cis):
        # Multi-Head Attention
        residual = x
        x = self.norm1(x)
        x = self.mha(x, freqs_cis)
        x = residual + x

        # Feed-Forward Network
        residual = x
        x = self.norm2(x)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        x = residual + x
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, intermediate_size, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim) # Token Embedding
        self.layers = nn.ModuleList([
            Block(dim, heads, intermediate_size) for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        freqs_cis = precompute_freqs_cis(dim // heads, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        return self.head(self.norm(x))