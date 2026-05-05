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

def reset_model_lamp(model, default_tau=1.0):
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
        pbar = tqdm(total=max_batches, desc="Evaluating Metrics")

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

def compute_perplexity(model, token_provider, max_batches, use_pbar=True):
    total_loss = 0.0
    count = 0

    if use_pbar:
        pbar = tqdm(total=max_batches, desc="Evaluating Perplexity")

    for _ in range(max_batches):
        tokens = token_provider.get_batch()

        with torch.no_grad():
            logits = model(tokens)

        loss = compute_loss(logits, tokens)
        total_loss += loss.item()
        count += 1
        
        if use_pbar:
            pbar.update(1)
            pbar.set_postfix({'PPL': f"{np.exp(total_loss/count):.3f}"})
    
    if use_pbar:
        pbar.close()

    return np.exp(total_loss / count)

def compute_choice_accuracy(model, choice_provider):
    correct = 0
    total = 0
    device = next(model.parameters()).device

    for item in choice_provider:
        lps = []
        for choice in item['choices']:
            # Move lists to GPU tensors just in time for the forward pass
            input_ids = torch.tensor([choice['input_ids']], device=device)
            shift_labels = torch.tensor(choice['cont_toks'], device=device)
            cont_len = choice['cont_len']

            with torch.no_grad():
                logits = model(input_ids)

            # Isolate the log probabilities of the continuation tokens
            shift_logits = logits[0, -cont_len-1:-1, :]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)

            lps.append(token_log_probs.sum().item())

        pred = lps.index(max(lps))
        if pred == item['label']:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0
