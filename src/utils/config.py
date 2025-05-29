"""Configuration utilities for Vietnamese summarization."""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for Vietnamese summarization."""
    return {
        'model': {
            'base_model': 'VietAI/vit5-large-vietnews-summarization',
            'max_input_length': 512,
            'max_output_length': 128,
            'enhancements': {
                'coverage_loss_weight': 1.0,
                'pointer_gen_loss_weight': 1.0
            }
        },
        'data': {
            'preprocessing': {
                'normalize_unicode': True,
                'remove_special_chars': False,
                'min_input_length': 50,
                'max_input_length': 512,
                'min_summary_length': 10,
                'max_summary_length': 128
            }
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 5e-5,
            'num_epochs': 3,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'gradient_clipping': 1.0,
            'curriculum_learning': False
        },
        'evaluation': {
            'metrics': ['rouge', 'bleu']
        },
        'hardware': {
            'device': 'auto',
            'mixed_precision': False
        }
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default config")
        return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Merge with default config to ensure all keys exist
        default_config = get_default_config()
        merged_config = _merge_configs(default_config, config)
        
        return merged_config
    
    except Exception as e:
        print(f"Error loading config: {e}, using default config")
        return get_default_config()


def _merge_configs(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge custom config with default config."""
    result = default.copy()
    
    for key, value in custom.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
