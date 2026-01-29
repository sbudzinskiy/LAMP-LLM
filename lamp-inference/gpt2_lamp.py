import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt2_vanilla import VanillaGPT2Config
from gpt2_lowprec import LowPrecGPT2Config, LowPrecLinear 
from custom_float import round_mantissa_, custom_accum_gemm
from dataclasses import dataclass, asdict, field

@dataclass
class LampGPT2Config(LowPrecGPT2Config):
    tau_activation: float = 2.0
    tau_softmax: float = 2.0
    fake_lamp: bool = False
    sparsity_logs: list[float] = field(default_factory=list)

    @classmethod
    def from_vanilla(cls, vanilla_config: VanillaGPT2Config, **kwargs):
        params = asdict(vanilla_config)
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def from_lowprec(cls, lowprec_config: LowPrecGPT2Config, **kwargs):
        params = asdict(lowprec_config)
        params.update(kwargs)
        return cls(**params)

# ---------------------------------------------------------------------------------------------------------------------------

def get_lamp_softmax_mask(x, threshold):
    original_shape = x.shape
    D = original_shape[-1]
    flat_x = x.view(-1, D)

    sorted_values, sorted_indices = torch.sort(flat_x, descending=True, dim=-1)
    cumulative_sums = sorted_values.cumsum(dim=-1)
    previous_sums = F.pad(cumulative_sums, (1,-1), value=0.0)
    lamp_sums = previous_sums + 2*sorted_values[:,-1:] 
    cutoff_mask = lamp_sums < threshold

    original_order_mask = torch.zeros_like(cutoff_mask).scatter_(
        dim=-1,
        index=sorted_indices,
        src=cutoff_mask
    )

    return original_order_mask.view(original_shape)

class LampAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        self.tau = config.tau_softmax
        self.fake_lamp = config.fake_lamp

        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                                     .view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels
        
        qkv = self.c_attn(x) 
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k_t = k.transpose(-2, -1).contiguous()
        scores = custom_accum_gemm(
            q, k_t,
            m_bits=self.m_bits_score,
            block_size_k=self.block_size_k
        )
        scores = scores.div_(math.sqrt(self.head_dim))
        scores = round_mantissa_(scores, self.m_bits_score)

        causal_mask = self.bias[:,:,:T,:T]
        causal_mask_card = B * self.n_head * causal_mask.sum().item()

        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        att = torch.softmax(scores, dim=-1)

        # Lamp section begins
        lamp_mask = get_lamp_softmax_mask(att, 2-self.tau)
        if self.fake_lamp:
            # Scatter the True values randomly in each row
            mask_shape = lamp_mask.shape
            flat_shape = lamp_mask.view(-1, mask_shape[-1]).shape
            weights = torch.ones(flat_shape, device=lamp_mask.device)
            perm_indices = torch.multinomial(weights, num_samples=mask_shape[-1], replacement=False).view(mask_shape)
            lamp_mask = lamp_mask.gather(dim=-1, index=perm_indices)

        scores_hp = torch.matmul(q, k_t).div_(math.sqrt(self.head_dim))
        #   When using higher precision, but not full precision:
        #   scores_hp = custom_accum_gemm(
        #       q, k_t,
        #       m_bits=23,
        #       block_size_k=self.block_size_k
        #   )
        scores[lamp_mask] = scores_hp[lamp_mask]
        scores = scores.masked_fill(causal_mask == 0, float('-inf')) # when Lamp threshold cannot be achieved
        att = torch.softmax(scores, dim=-1)

        lamp_mask_card = lamp_mask.sum().item()
        self.config.sparsity_logs.append(lamp_mask_card / causal_mask_card)
        # Lamp section ends

        y = custom_accum_gemm(
            att, v,
            m_bits=self.m_bits_value,
            block_size_k=self.block_size_k
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

# ---------------------------------------------------------------------------------------------------------------------------

class LampMLP(nn.Module):
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
        self.tau = config.tau_activation

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

# ---------------------------------------------------------------------------------------------------------------------------

class LampBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = LampAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = LampMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ---------------------------------------------------------------------------------------------------------------------------

class LampGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([LampBlock(config) for _ in range(config.n_layer)])
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
        self.config.sparsity_logs.clear()

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

    def mean_sparsity(self):
        if len(self.config.sparsity_logs) % self.config.n_layer == 0:
            mean_per_layer = torch.tensor(self.config.sparsity_logs).view(-1, self.config.n_layer).mean(dim=0)
            return mean_per_layer
        else:
            raise ValueError("Logs are incomplete: Forward pass likely failed.")
