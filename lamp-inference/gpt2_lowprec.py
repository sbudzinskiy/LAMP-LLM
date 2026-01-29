import math
import torch
import torch.nn as nn
from gpt2_vanilla import VanillaGPT2Config
from custom_float import round_mantissa_, custom_accum_gemm
from dataclasses import dataclass, asdict

@dataclass
class LowPrecGPT2Config(VanillaGPT2Config):
    m_bits_mlp_fc: int = 23
    m_bits_mlp_proj: int = 23
    m_bits_attn_qkv: int = 23
    m_bits_attn_score: int = 23
    m_bits_attn_value: int = 23
    m_bits_attn_proj: int = 23
    m_bits_lm_proj: int = 23
    block_size_k: int = 1      # 1 = Strict, 32 = Efficient

    @classmethod
    def from_vanilla(cls, vanilla_config: VanillaGPT2Config, **kwargs):
        params = asdict(vanilla_config)
        params.update(kwargs)
        return cls(**params)

class LowPrecLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, m_bits=23, block_size_k=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m_bits = m_bits
        self.block_size_k = block_size_k
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        w_t = self.weight.t()
        
        out = custom_accum_gemm(
            x, w_t, 
            m_bits=self.m_bits, 
            block_size_k=self.block_size_k
        )
        
        if self.bias is not None:
            out = out + self.bias
            out = round_mantissa_(out, self.m_bits)
        return out

class LowPrecAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = LowPrecLinear(
            config.n_embd,
            3 * config.n_embd,
            m_bits=config.m_bits_attn_qkv,
            block_size_k=config.block_size_k
        )
        self.c_proj = LowPrecLinear(
            config.n_embd,
            config.n_embd,
            m_bits=config.m_bits_attn_proj,
            block_size_k=config.block_size_k
        )

        self.m_bits_score = config.m_bits_attn_score
        self.m_bits_value = config.m_bits_attn_value
        self.block_size_k = config.block_size_k

        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                                     .view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x) 
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k_t = k.transpose(-2, -1).contiguous()
        att = custom_accum_gemm(
            q, k_t,
            m_bits=self.m_bits_score,
            block_size_k=self.block_size_k
        )
        att = att.div_(math.sqrt(self.head_dim))
        att = round_mantissa_(att, self.m_bits_score)
               
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)

        y = custom_accum_gemm(
            att, v,
            m_bits=self.m_bits_value,
            block_size_k=self.block_size_k
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

class LowPrecMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = LowPrecLinear(
            config.n_embd,
            4 * config.n_embd,
            m_bits=config.m_bits_mlp_fc,
            block_size_k=config.block_size_k
        )
        self.act = nn.GELU()
        self.c_proj = LowPrecLinear(
            4 * config.n_embd,
            config.n_embd,
            m_bits=config.m_bits_mlp_proj,
            block_size_k=config.block_size_k
        )

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class LowPrecBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = LowPrecAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = LowPrecMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class LowPrecGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([LowPrecBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.lm_head = LowPrecLinear(
            config.n_embd,
            config.vocab_size,
            bias=False,
            m_bits=config.m_bits_lm_proj,
            block_size_k=config.block_size_k
        )
        self.lm_head.weight = self.wte.weight 

    def forward(self, idx):
        device = idx.device
        B, T = idx.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=device) 
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.h:
            x = block(x)
            
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        return logits
