import os
import sys
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

from token_provider import *
from experiment_helper import *
from weight_loader import get_gpt2_files
from gpt2_pretrained import load_pretrained_gpt2
from gpt2_lamp import LampGPT2Config, LampGPT2Model

def one_lamp_softmax_experiment(ref_model, test_model, m_bits_attn_score, tau_softmax, token_provider, nbatches, use_pbar):
    reset_model_lamp(test_model)
    update_model_lamp(test_model, 'tau_softmax', tau_softmax)

    reset_model_precision(test_model)
    update_model_precision(test_model, 'm_bits_attn_score', m_bits_attn_score)

    try:
        # Computes the metrics and resets the token provider
        kl, fr = compute_average_metrics(ref_model, test_model, token_provider, nbatches, use_pbar)
        sparsity_per_layer = test_model.mean_sparsity()
        sp = sparsity_per_layer.mean().item()
        pass
    finally:
        if hasattr(token_provider, 'reset'):
            token_provider.reset()
 
    return kl, fr, sp

def many_lamp_softmax_experiments(seed, model_type, weights_dir, split, shuffle, fake_lamp):
    set_seed(seed)
    print(f"TF32 Allowed: {torch.backends.cuda.matmul.allow_tf32}")

    ref_model = load_pretrained_gpt2(model_type, weights_dir).cuda().eval()
    ref_config = ref_model.config
    
    test_config = LampGPT2Config.from_vanilla(ref_config, fake_lamp=fake_lamp)
    test_model = LampGPT2Model(test_config).cuda().eval()
    test_model.load_state_dict(ref_model.state_dict())

    nbatches = 200 

    m_bits_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    tau_range = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.08, 1.06, 1.04, 1.02]
    dataset_names = ["OpenWebText", "CodeParrot", "ArXiv Science"]

    grid = list(itertools.product(dataset_names, m_bits_range, tau_range))
    pbar = tqdm(grid, desc="Grid Search")

    results = []

    for dataset_name in dataset_names:
        try:
            dataset, text_column_name = stream_dataset_from_config(dataset_name, split=split)
            token_provider = TokenFromTextProvider(
                    dataset,
                    batch_size=1,
                    seq_len=1024,
                    shuffle_tokens=shuffle,
                    text_column_name=text_column_name
            )
        except Exception as e:
            print(f"\nCannot load {dataset_name} ({split}): {e}")
            continue

        for tau in tau_range:
            for m_bits in m_bits_range:
                pbar.set_description(f"DS {dataset_name} | Bits {m_bits} | Thresh {tau:.2f}")
                try:
                    kl, fr, sp = one_lamp_softmax_experiment(ref_model, test_model, m_bits, tau, token_provider, nbatches, False)
                    pbar.update(1)
                    results.append({
                        'model': model_type,
                        'seed': seed,
                        'nbatches': nbatches,
                        'dataset': dataset_name,
                        'split': split,
                        'shuffle': shuffle,
                        'fake_lamp': fake_lamp,
                        'm_bits': m_bits,
                        'tau': tau,
                        'kl_div': kl,
                        'flip_rate': fr,
                        'sparsity': sp
                    })
                except Exception as e:
                    print(f"\nCrash at {model_type}, {dataset_name}, {m_bits}b, {tau:.1f}t: {e}")
                    results.append({
                        'model': model_type,
                        'seed': seed,
                        'nbatches': nbatches,
                        'dataset': dataset_name,
                        'split': split,
                        'shuffle': shuffle,
                        'fake_lamp': fake_lamp,
                        'm_bits': m_bits,
                        'tau': tau,
                        'kl_div': -1,
                        'flip_rate': -1,
                        'sparsity': -1
                    })

        token_provider.close()

    pbar.close()
    df = pd.DataFrame(results)
    filename = f"lamp_softmax_{model_type}_{seed}s_{nbatches}nb.csv"
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    print(f"Results saved to {filename}.")
    return df

if __name__ == "__main__":
    seed = 42  
    model_type = "gpt2"
    split = "train"
    weights_dir = get_gpt2_files(model_type, ".")

    shuffle = False
    fake_lamp = False
    df = many_lamp_softmax_experiments(seed, model_type, weights_dir, split, shuffle, fake_lamp)
    fake_lamp = True
    df = many_lamp_softmax_experiments(seed, model_type, weights_dir, split, shuffle, fake_lamp)

    shuffle = True
    fake_lamp = False
    df = many_lamp_softmax_experiments(seed, model_type, weights_dir, split, shuffle, fake_lamp)
