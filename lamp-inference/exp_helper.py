import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_model_precision(model, target_gemm, m_bits):
    setattr(model.config, target_gemm, m_bits)
    
    for block in model.h:
        if target_gemm == 'm_bits_mlp_fc':     block.mlp.c_fc.m_bits = m_bits
        elif target_gemm == 'm_bits_mlp_proj': block.mlp.c_proj.m_bits = m_bits
        elif target_gemm == 'm_bits_attn_qkv': block.attn.c_attn.m_bits = m_bits
        elif target_gemm == 'm_bits_attn_proj': block.attn.c_proj.m_bits = m_bits
        elif target_gemm == 'm_bits_attn_score': block.attn.m_bits_score = m_bits
        elif target_gemm == 'm_bits_attn_value': block.attn.m_bits_value = m_bits
            
    if target_gemm == 'm_bits_head':
        model.lm_head.m_bits = m_bits

def update_model_lamp(model, target, tau):
    setattr(model.config, target, tau)
    
    for block in model.h:
        if target == 'tau_softmax': block.attn.tau = tau
        elif target == 'tau_activation': block.mlp.tau = tau

def reset_model_precision(model, default_bits=23):
    targets = [
        'm_bits_mlp_fc', 'm_bits_mlp_proj', 
        'm_bits_attn_qkv', 'm_bits_attn_proj', 
        'm_bits_attn_score', 'm_bits_attn_value',
        'm_bits_head'
    ]
    for t in targets:
        update_model_precision(model, t, default_bits)

def reset_model_lamp(model, default_tau=2.0):
    targets = ['tau_softmax', 'tau_activation']
    for t in targets:
        update_model_lamp(model, t, default_tau)

# ---------------------------------------------------------------------------------------------------------------------------

def compute_loss(logits, inputs):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    return loss

def compute_metrics(ref_logits, test_logits):
    ref_flat = ref_logits.view(-1, ref_logits.size(-1))
    test_flat = test_logits.view(-1, test_logits.size(-1))

    kl = F.kl_div(
        F.log_softmax(test_flat, dim=-1),
        F.log_softmax(ref_flat, dim=-1),
        reduction='batchmean',
        log_target=True
    )

    ref_top1 = ref_flat.argmax(dim=-1)
    test_top1 = test_flat.argmax(dim=-1)
    flip_rate = (ref_top1 != test_top1).float().mean()

    return kl.item(), flip_rate.item()

def compute_average_metrics(ref_model, test_model, token_provider, max_batches, use_pbar=True):
    total_kl = 0.0
    total_fr = 0.0
    count = 0

    if use_pbar:
        pbar = tqdm(total=max_batches, desc="Evaluating")

    for _ in range(max_batches):
        tokens = token_provider.get_batch()

        with torch.no_grad():
            ref_logits = ref_model(tokens)
            test_logits = test_model(tokens)

        kl, fr = compute_metrics(ref_logits, test_logits)
        total_kl += kl
        total_fr += fr
        count += 1
        if use_pbar:
            pbar.update(1)
            pbar.set_postfix({'KL': f"{total_kl/count:.3e}", 'FR' : f"{total_fr/count:.3e}"})
    
    if use_pbar:
        pbar.close()

    return total_kl / count, total_fr / count
