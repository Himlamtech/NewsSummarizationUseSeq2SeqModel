#!/usr/bin/env python3
"""
Training Script for Enhanced Vietnamese Summarization

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/custom_config.yaml
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.training.trainer import EnhancedTrainer
from src.utils.config import load_config, get_default_config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Enhanced Vietnamese Summarization Model")
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting Enhanced Vietnamese Summarization Training")
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
        else:
            logger.info("Using default configuration")
            config = get_default_config()
        
        # Override config with command line arguments
        if args.epochs:
            config['training']['num_epochs'] = args.epochs
            logger.info(f"Overriding epochs to {args.epochs}")
        
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
            logger.info(f"Overriding batch size to {args.batch_size}")
        
        # Print configuration summary
        logger.info("Configuration Summary:")
        logger.info(f"  Model: {config['model']['base_model']}")
        logger.info(f"  Epochs: {config['training']['num_epochs']}")
        logger.info(f"  Batch Size: {config['training']['batch_size']}")
        logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
        
        # Create trainer and start training
        trainer = EnhancedTrainer(config)
        trainer.train()
        
        logger.info("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
