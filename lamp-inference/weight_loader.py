import os
import torch
import urllib.request
import json
import sys

def get_gpt2_files(model_type, master_dir):
    """
    Downloads raw GPT-2 weights and config from HuggingFace.
    """
    base_url = f"https://huggingface.co/{model_type}/resolve/main"
    files = ['pytorch_model.bin', 'config.json', 'vocab.json', 'merges.txt']

    save_dir = f"{master_dir}/{model_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Downloading {model_type} to {save_dir}...")
    
    for filename in files:
        file_path = os.path.join(save_dir, filename)
        if not os.path.exists(file_path):
            url = f"{base_url}/{filename}"
            print(f"Fetching {filename}...")
            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                sys.exit(1)
        else:
            print(f"Found {filename} locally.")

    print("="*50)
    return save_dir

def load_weights_into_model(model, weights_path):
    """
    Loads HuggingFace weights into a standard PyTorch GPT-2 model.
    Handles the Transpose logic required because HF uses Conv1D (In, Out) 
    while PyTorch nn.Linear uses (Out, In).
    """
    print("Loading state dict...")
    hf_sd = torch.load(weights_path, map_location='cpu', weights_only=True)
    my_sd = model.state_dict()
    
    ignore_keys = ['.attn.masked_bias', '.attn.bias'] 
    
    transpose_layers = ['c_attn.weight', 'c_proj.weight', 'c_fc.weight']
    
    print("Adapting weights...")
    for k, v in hf_sd.items():
        if any(i in k for i in ignore_keys):
            continue
            
        if any(k.endswith(suffix) for suffix in transpose_layers):
            v = v.t()
        
        if k in my_sd:
            assert my_sd[k].shape == v.shape, f"Shape mismatch: {k} | Mine: {my_sd[k].shape} vs HF: {v.shape}"
            with torch.no_grad():
                my_sd[k].copy_(v)
        else:
            print(f"Unused key in checkpoint: {k}")

    print("Weights loaded successfully!")
    print("="*50)
    return model
