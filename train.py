#!/usr/bin/env python3
"""
Main Training Script for Enhanced Vietnamese Summarization
Entry point for training the enhanced model with all advanced techniques
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import torch
import wandb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.trainer import EnhancedTrainer
from src.evaluation.metrics import SummarizationEvaluator
from src.utils.data_augmentation import VietnameseDataAugmenter


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> bool:
    """Validate configuration parameters"""
    required_sections = ['model', 'data', 'training', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model config
    if 'base_model' not in config['model']:
        raise ValueError("Missing base_model in model configuration")
    
    # Validate training config
    training_config = config['training']
    required_training_params = ['batch_size', 'learning_rate', 'num_epochs']
    
    for param in required_training_params:
        if param not in training_config:
            raise ValueError(f"Missing required training parameter: {param}")
    
    return True


def setup_directories(config: dict):
    """Create necessary directories"""
    paths = config.get('paths', {})
    
    directories = [
        paths.get('data_dir', 'data/'),
        paths.get('checkpoint_dir', 'checkpoints/'),
        paths.get('results_dir', 'results/'),
        paths.get('logs_dir', 'logs/')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def print_config_summary(config: dict, logger):
    """Print configuration summary"""
    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("="*60)
    
    # Model configuration
    model_config = config.get('model', {})
    logger.info(f"Base Model: {model_config.get('base_model', 'N/A')}")
    logger.info(f"Model Type: {model_config.get('model_type', 'N/A')}")
    logger.info(f"Max Input Length: {model_config.get('max_input_length', 'N/A')}")
    logger.info(f"Max Output Length: {model_config.get('max_output_length', 'N/A')}")
    
    # Enhancement configuration
    enhancements = model_config.get('enhancements', {})
    logger.info("\nEnhancements:")
    for enhancement, config_dict in enhancements.items():
        if isinstance(config_dict, dict):
            enabled = config_dict.get('enabled', False)
            logger.info(f"  {enhancement}: {'✓' if enabled else '✗'}")
        else:
            logger.info(f"  {enhancement}: {config_dict}")
    
    # Training configuration
    training_config = config.get('training', {})
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Batch Size: {training_config.get('batch_size', 'N/A')}")
    logger.info(f"  Learning Rate: {training_config.get('learning_rate', 'N/A')}")
    logger.info(f"  Epochs: {training_config.get('num_epochs', 'N/A')}")
    logger.info(f"  Optimizer: {training_config.get('optimizer', 'N/A')}")
    logger.info(f"  Scheduler: {training_config.get('scheduler', 'N/A')}")
    
    # Advanced techniques
    curriculum = training_config.get('curriculum_learning', {})
    teacher_forcing = training_config.get('teacher_forcing', {})
    
    logger.info(f"\nAdvanced Techniques:")
    logger.info(f"  Curriculum Learning: {'✓' if curriculum.get('enabled', False) else '✗'}")
    logger.info(f"  Teacher Forcing: {'✓' if teacher_forcing.get('enabled', False) else '✗'}")
    
    # Hardware configuration
    hardware_config = config.get('hardware', {})
    logger.info(f"\nHardware Configuration:")
    logger.info(f"  Device: {hardware_config.get('device', 'auto')}")
    logger.info(f"  Mixed Precision: {'✓' if hardware_config.get('mixed_precision', False) else '✗'}")
    logger.info(f"  DataLoader Workers: {hardware_config.get('dataloader_num_workers', 4)}")
    
    logger.info("="*60)


def run_data_augmentation(config: dict, logger):
    """Run data augmentation if enabled"""
    augmentation_config = config.get('data', {}).get('augmentation', {})
    
    if not augmentation_config.get('enabled', False):
        logger.info("Data augmentation disabled, skipping...")
        return
    
    logger.info("Starting data augmentation...")
    
    augmenter = VietnameseDataAugmenter(config)
    
    # Load original dataset
    data_dir = config.get('paths', {}).get('data_dir', 'data/')
    original_data_path = os.path.join(data_dir, 'original_dataset.json')
    
    if not os.path.exists(original_data_path):
        logger.warning(f"Original dataset not found at {original_data_path}, skipping augmentation")
        return
    
    original_data = augmenter.load_augmented_dataset(original_data_path)
    
    # Augment dataset
    strategies = augmentation_config.get('strategies', ['paraphrase', 'sentence_reorder'])
    ratio = augmentation_config.get('ratio', 0.5)
    
    augmented_data = augmenter.augment_dataset(
        original_data, 
        strategies=strategies,
        augmentation_ratio=ratio
    )
    
    # Save augmented dataset
    augmented_data_path = os.path.join(data_dir, 'augmented_dataset.json')
    augmenter.save_augmented_dataset(augmented_data, augmented_data_path)
    
    logger.info(f"Data augmentation completed. Augmented dataset saved to {augmented_data_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Enhanced Vietnamese Summarization Training")
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/enhanced_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--eval-only', 
        action='store_true',
        help='Only run evaluation, skip training'
    )
    
    parser.add_argument(
        '--augment-data', 
        action='store_true',
        help='Run data augmentation before training'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Enhanced Vietnamese Summarization Training")
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        validate_config(config)
        
        # Setup directories
        setup_directories(config)
        
        # Print configuration summary
        print_config_summary(config, logger)
        
        # Run data augmentation if requested
        if args.augment_data:
            run_data_augmentation(config, logger)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = EnhancedTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        if args.eval_only:
            # Run evaluation only
            logger.info("Running evaluation only...")
            trainer.initialize_model()
            trainer.initialize_data_loaders()
            
            # Evaluate on test set
            test_metrics = trainer.evaluate(trainer.test_loader, split='test')
            
            # Print results
            logger.info("Test Results:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Save results
            results_dir = config.get('paths', {}).get('results_dir', 'results/')
            results_file = os.path.join(results_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            import json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {results_file}")
            
        else:
            # Run full training
            logger.info("Starting training...")
            trainer.train()
            
            # Final evaluation on test set
            logger.info("Running final evaluation on test set...")
            test_metrics = trainer.evaluate(trainer.test_loader, split='test')
            
            # Print final results
            logger.info("Final Test Results:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if config.get('logging', {}).get('use_wandb', False):
            wandb.finish()


if __name__ == "__main__":
    main()
