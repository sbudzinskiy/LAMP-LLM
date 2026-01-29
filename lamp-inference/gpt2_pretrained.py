import json
import os
from weight_loader import get_gpt2_files, load_weights_into_model
from gpt2_vanilla import VanillaGPT2Model, VanillaGPT2Config
from gpt2_lowprec import LowPrecGPT2Model, LowPrecGPT2Config

def pretrained_gpt2_from_config(config_path):
    """
    Reads the downloaded config.json and initializes the correct model architecture
    (Small, Medium, Large, or XL).
    """
    with open(config_path, 'r') as f:
        config_raw = json.load(f)
    
    print(f"Detected Model Config: {config_raw['n_layer']} layers, {config_raw['n_embd']} hidden size.")

    # Override defaults with values from the file
    config = VanillaGPT2Config(
        vocab_size=config_raw.get('vocab_size', 50257),
        n_positions=config_raw.get('n_positions', 1024),
        n_embd=config_raw['n_embd'],
        n_layer=config_raw['n_layer'],
        n_head=config_raw['n_head'],
        layer_norm_epsilon=config_raw.get('layer_norm_epsilon', 1e-5)
    )
    
    return VanillaGPT2Model(config)

def load_pretrained_gpt2(model_type, weights_dir):
    save_dir = get_gpt2_files(model_type, weights_dir)
    
    config_path = os.path.join(save_dir, 'config.json')
    model = pretrained_gpt2_from_config(config_path)
    
    weights_path = os.path.join(save_dir, 'pytorch_model.bin')
    model = load_weights_into_model(model, weights_path)
    
    return model

def lowprec_gpt2_from_config(
        config_path,
        m_bits_mlp_fc,
        m_bits_mlp_proj,
        m_bits_attn_qkv,
        m_bits_attn_score,
        m_bits_attn_value,
        m_bits_attn_proj,
        m_bits_lm_proj,
        block_size_k
):
    """
    Reads the downloaded config.json and initializes the correct model architecture
    (Small, Medium, Large, or XL).
    """
    with open(config_path, 'r') as f:
        config_raw = json.load(f)
    
    print(f"Detected Model Config: {config_raw['n_layer']} layers, {config_raw['n_embd']} hidden size.")

    config = LowPrecGPT2Config(
        vocab_size=config_raw.get('vocab_size', 50257),
        n_positions=config_raw.get('n_positions', 1024),
        n_embd=config_raw['n_embd'],
        n_layer=config_raw['n_layer'],
        n_head=config_raw['n_head'],
        layer_norm_epsilon=config_raw.get('layer_norm_epsilon', 1e-5),
        m_bits_mlp_fc=m_bits_mlp_fc,
        m_bits_mlp_proj=m_bits_mlp_proj,
        m_bits_attn_qkv=m_bits_attn_qkv,
        m_bits_attn_score=m_bits_attn_score,
        m_bits_attn_value=m_bits_attn_value,
        m_bits_attn_proj=m_bits_attn_proj,
        m_bits_lm_proj=m_bits_lm_proj,
        block_size_k=block_size_k
    )
    
    return LowPrecGPT2Model(config)

def load_lowprec_gpt2(
        model_type, weights_dir,
        m_bits_mlp_fc,
        m_bits_mlp_proj,
        m_bits_attn_qkv,
        m_bits_attn_score,
        m_bits_attn_value,
        m_bits_attn_proj,
        m_bits_lm_proj,
        block_size_k
):
    save_dir = get_gpt2_files(model_type, weights_dir)
    
    config_path = os.path.join(save_dir, 'config.json')
    model = lowprec_gpt2_from_config(
            config_path,
            m_bits_mlp_fc,
            m_bits_mlp_proj,
            m_bits_attn_qkv,
            m_bits_attn_score,
            m_bits_attn_value,
            m_bits_attn_proj,
            m_bits_lm_proj,
            block_size_k
    )
    
    weights_path = os.path.join(save_dir, 'pytorch_model.bin')
    model = load_weights_into_model(model, weights_path)
    
    return model

