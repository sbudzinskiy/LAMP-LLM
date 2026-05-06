import os
import sys
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

from token_provider import *
from exp_helper import *
from weight_loader import get_gpt2_files
from gpt2_pretrained import load_pretrained_gpt2
from gpt2_lowprec import LowPrecGPT2Config, LowPrecGPT2Model
from gpt2_lamp import LampGPT2Config, LampGPT2Model

def one_lamp_perplexity_experiment(model, token_provider, nbatches, use_pbar):
    try:
        perp = compute_perplexity(model, token_provider, nbatches, use_pbar)
        if isinstance(model, LampGPT2Model):
            sparsity_per_layer = model.mean_sparsity()
            sp = sparsity_per_layer.mean().item()
        else:
            sp = 0.0
        pass
    finally:
        if hasattr(token_provider, 'reset'):
            token_provider.reset()
 
    return perp, sp

def one_lamp_accuracy_experiment(model, choice_provider):
    acc = compute_choice_accuracy(model, choice_provider)
    
    if isinstance(model, LampGPT2Model):
        sp = model.mean_sparsity().mean().item()
    else:
        sp = 0.0
    return acc, sp

def many_lamp_downstream_experiments(seed, nbatches, seq_len, model_type, weights_dir, length_lamp):
    set_seed(seed)
    print(f"TF32 Allowed: {torch.backends.cuda.matmul.allow_tf32}")

    m_bits_ref = 23

    ref_model = load_pretrained_gpt2(model_type, weights_dir).cuda().eval()
    ref_config = ref_model.config

    lowprec_config = LowPrecGPT2Config.from_vanilla(ref_config)
    lowprec_model = LowPrecGPT2Model(lowprec_config).cuda().eval()
    lowprec_model.load_state_dict(ref_model.state_dict())

    lamp_config = LampGPT2Config.from_lowprec(lowprec_config, relax_lamp=True, length_lamp=length_lamp, fake_lamp=False)
    lamp_model = LampGPT2Model(lamp_config).cuda().eval()
    lamp_model.load_state_dict(ref_model.state_dict())

    m_bits_range = [4]
    tau_range = [9e-2, 3e-2]

    results = []

    datasets_perplexity = {
        "GSM8k Math": "test",
        "WikiText-2": "test",
        "CodeParrot": "train"
    }

    datasets_accuracy = {
    }

    # Perplexity experiments
    iterations_perplexity = len(datasets_perplexity) * ((len(tau_range) + 1) * len(m_bits_range) + 1)
    pbar = tqdm(total=iterations_perplexity)
    for dataset_name, dataset_split in datasets_perplexity.items():
        try:
            dataset, text_column_name = stream_dataset_from_config(dataset_name, split=dataset_split)
            token_provider = TokenFromTextProvider(
                    dataset,
                    batch_size=1,
                    seq_len=seq_len,
                    shuffle_tokens=False,
                    text_column_name=text_column_name
            )
        except Exception as e:
            print(f"\nCannot load {dataset_name} ({dataset_split}): {e}")
            continue

        # Full-precision model
        try:
            pbar.set_description(f"DS {dataset_name} | Bits {m_bits_ref} | Thresh N/A")
            perp, sp = one_lamp_perplexity_experiment(ref_model, token_provider, nbatches, False)
            pbar.update(1)
            results.append({
                'model': model_type, 'seed': seed, 'nbatches': nbatches, 'seq_len': seq_len, 'dataset': dataset_name,
                'split': dataset_split, 'length_lamp': 'n/a', 'm_bits': m_bits_ref, 'tau': 'n/a', 'type': 'perplexity', 'score': perp, 'sparsity': sp
            })
        except Exception as e:
            print(f"\nCrash at {model_type}, {dataset_name}, {m_bits_ref}b, reference: {e}")
            results.append({
                'model': model_type, 'seed': seed, 'nbatches': nbatches, 'seq_len': seq_len, 'dataset': dataset_name,
                'split': dataset_split, 'length_lamp': 'n/a', 'm_bits': m_bits_ref, 'tau': 'n/a', 'type': 'perplexity', 'score': -1, 'sparsity': -1
            })

        for m_bits in m_bits_range:
            reset_model_precision(lowprec_model)
            update_model_precision(lowprec_model, 'm_bits_attn_score', m_bits)
            reset_model_precision(lamp_model)
            update_model_precision(lamp_model, 'm_bits_attn_score', m_bits)

            # Low-precision model
            try:
                pbar.set_description(f"DS {dataset_name} | Bits {m_bits} | Thresh N/A")
                perp, sp = one_lamp_perplexity_experiment(lowprec_model, token_provider, nbatches, False)
                pbar.update(1)
                results.append({
                    'model': model_type, 'seed': seed, 'nbatches': nbatches, 'seq_len': seq_len, 'dataset': dataset_name,
                    'split': dataset_split, 'length_lamp': 'n/a', 'm_bits': m_bits, 'tau': 'n/a', 'type': 'perplexity', 'score': perp, 'sparsity': sp
                })
            except Exception as e:
                print(f"\nCrash at {model_type}, {dataset_name}, {m_bits}b, lowprec: {e}")
                results.append({
                    'model': model_type, 'seed': seed, 'nbatches': nbatches, 'seq_len': seq_len, 'dataset': dataset_name,
                    'split': dataset_split, 'length_lamp': 'n/a', 'm_bits': m_bits, 'tau': 'n/a', 'type': 'perplexity', 'score': -1, 'sparsity': -1
                })
    
            # LAMP models
            for tau in tau_range:
                pbar.set_description(f"DS {dataset_name} | Bits {m_bits} | Thresh {tau:.2f}")
                try:
                    reset_model_lamp(lamp_model)
                    update_model_lamp(lamp_model, 'tau_softmax', tau)
    
                    perp, sp = one_lamp_perplexity_experiment(lamp_model, token_provider, nbatches, False)
                    pbar.update(1)
                    results.append({
                        'model': model_type, 'seed': seed, 'nbatches': nbatches, 'seq_len': seq_len, 'dataset': dataset_name,
                        'split': dataset_split, 'length_lamp': length_lamp, 'm_bits': m_bits, 'tau': tau, 'type': 'perplexity', 'score': perp, 'sparsity': sp
                    })
                except Exception as e:
                    print(f"\nCrash at {model_type}, {dataset_name}, {m_bits}b, {tau:.1f}t: {e}")
                    results.append({
                        'model': model_type, 'seed': seed, 'nbatches': nbatches, 'seq_len': seq_len, 'dataset': dataset_name,
                        'split': dataset_split, 'length_lamp': length_lamp, 'm_bits': m_bits, 'tau': tau, 'type': 'perplexity', 'score': -1, 'sparsity': -1
                    })

        token_provider.close()

    # Accuracy experiments
    choice_providers = {}
    for dataset_name, dataset_split in datasets_accuracy.items():
        ds_stream, _ = stream_dataset_from_config(dataset_name, split=dataset_split)
        mat_data = list(ds_stream)
        choice_providers[dataset_name] = MultipleChoiceProvider(mat_data, seq_len=seq_len)

    iterations_accuracy = len(choice_providers) * ((len(tau_range) + 1) * len(m_bits_range) + 1)
    pbar = tqdm(total=iterations_accuracy)
    for dataset_name, choice_provider in choice_providers.items():
        dataset_split = datasets_accuracy[dataset_name]
        # Full-precision model
        try:
            pbar.set_description(f"DS {dataset_name} | Bits {m_bits_ref} | Thresh N/A")
            acc, sp = one_lamp_accuracy_experiment(ref_model, choice_provider)
            pbar.update(1)
            results.append({
                'model': model_type, 'seed': seed, 'nbatches': 'all', 'seq_len': seq_len, 'dataset': dataset_name,
                'split': dataset_split, 'length_lamp': 'n/a',  'm_bits': m_bits_ref, 'tau': 'n/a', 'type': 'accuracy', 'score': acc, 'sparsity': sp
            })
        except Exception as e:
            print(f"\nCrash at {model_type}, {dataset_name}, {m_bits_ref}b, reference: {e}")
            results.append({
                'model': model_type, 'seed': seed, 'nbatches': 'all', 'seq_len': seq_len, 'dataset': dataset_name,
                'split': dataset_split, 'length_lamp': 'n/a', 'm_bits': m_bits_ref, 'tau': 'n/a', 'type': 'accuracy', 'score': -1, 'sparsity': -1
            })

        for m_bits in m_bits_range:
            reset_model_precision(lowprec_model)
            update_model_precision(lowprec_model, 'm_bits_attn_score', m_bits)
            reset_model_precision(lamp_model)
            update_model_precision(lamp_model, 'm_bits_attn_score', m_bits)

            # Low-precision model
            try:
                pbar.set_description(f"DS {dataset_name} | Bits {m_bits} | Thresh N/A")
                acc, sp = one_lamp_accuracy_experiment(lowprec_model, choice_provider)
                pbar.update(1)
                results.append({
                    'model': model_type, 'seed': seed, 'nbatches': 'all', 'seq_len': seq_len, 'dataset': dataset_name,
                    'split': dataset_split, 'length_lamp': 'n/a', 'm_bits': m_bits, 'tau': 'n/a', 'type': 'accuracy', 'score': acc, 'sparsity': sp
                })
            except Exception as e:
                print(f"\nCrash at {model_type}, {dataset_name}, {m_bits}b, lowprec: {e}")
                results.append({
                    'model': model_type, 'seed': seed, 'nbatches': 'all', 'seq_len': seq_len, 'dataset': dataset_name,
                    'split': dataset_split, 'length_lamp': 'n/a', 'm_bits': m_bits, 'tau': 'n/a', 'type': 'accuracy', 'score': -1, 'sparsity': -1
                })
    
            # LAMP models
            for tau in tau_range:
                pbar.set_description(f"DS {dataset_name} | Bits {m_bits} | Thresh {tau:.2f}")
                try:
                    reset_model_lamp(lamp_model)
                    update_model_lamp(lamp_model, 'tau_softmax', tau)
    
                    acc, sp = one_lamp_accuracy_experiment(lamp_model, choice_provider)
                    pbar.update(1)
                    results.append({
                        'model': model_type, 'seed': seed, 'nbatches': 'all', 'seq_len': seq_len, 'dataset': dataset_name,
                        'split': dataset_split, 'length_lamp': length_lamp, 'm_bits': m_bits, 'tau': tau, 'type': 'accuracy', 'score': acc, 'sparsity': sp
                    })
                except Exception as e:
                    print(f"\nCrash at {model_type}, {dataset_name}, {m_bits}b, {tau:.1f}t: {e}")
                    results.append({
                        'model': model_type, 'seed': seed, 'nbatches': 'all', 'seq_len': seq_len, 'dataset': dataset_name,
                        'split': dataset_split, 'length_lamp': length_lamp,  'm_bits': m_bits, 'tau': tau, 'type': 'accuracy', 'score': -1, 'sparsity': -1
                    })

    pbar.close()
    df = pd.DataFrame(results)
    filename = f"lamp_downstream_{model_type}_{seed}s_{nbatches}nb.csv"
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False, float_format='%.4e')
    print(f"Results saved to {filename}.")
    return df

if __name__ == "__main__":
    seed = 42
    nbatches = 100
    seq_len = 1024
    model_type = sys.argv[1]
    weights_dir = get_gpt2_files(model_type, ".")

    for length_lamp in [False, True]:
        df = many_lamp_downstream_experiments(seed, nbatches, seq_len, model_type, weights_dir, length_lamp)
