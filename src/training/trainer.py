"""
Enhanced Training Script for Vietnamese Summarization
Implements curriculum learning, teacher forcing, and advanced optimization
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import yaml
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import json

from ..models.enhanced_t5 import create_enhanced_model
from ..data.dataset_loader import create_data_loaders
from ..evaluation.metrics import SummarizationEvaluator


class CurriculumLearningScheduler:
    """Scheduler for curriculum learning progression"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_length = config.get('start_length', 256)
        self.end_length = config.get('end_length', 1024)
        self.progression_epochs = config.get('progression_epochs', 3)
        
    def get_max_length(self, epoch: int) -> int:
        """Get maximum input length for current epoch"""
        if epoch >= self.progression_epochs:
            return self.end_length
        
        progress = epoch / self.progression_epochs
        current_length = self.start_length + (self.end_length - self.start_length) * progress
        return int(current_length)


class TeacherForcingScheduler:
    """Scheduler for teacher forcing ratio decay"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.initial_ratio = config.get('initial_ratio', 1.0)
        self.final_ratio = config.get('final_ratio', 0.5)
        self.decay_steps = config.get('decay_steps', 5000)
        
    def get_teacher_forcing_ratio(self, step: int) -> float:
        """Get teacher forcing ratio for current step"""
        if step >= self.decay_steps:
            return self.final_ratio
        
        progress = step / self.decay_steps
        ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * progress
        return ratio


class EnhancedTrainer:
    """Enhanced trainer with advanced techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        
        # Advanced schedulers
        self.curriculum_scheduler = None
        self.teacher_forcing_scheduler = None
        
        # Evaluation
        self.evaluator = SummarizationEvaluator(config)
        
        # Setup directories
        self._setup_directories()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup wandb if enabled
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['experiment_name'],
                config=self.config
            )
    
    def _setup_device(self) -> torch.device:
        """Setup computing device"""
        device_config = self.config.get('hardware', {}).get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = ['checkpoint_dir', 'results_dir', 'logs_dir']
        for dir_key in dirs:
            dir_path = self.config.get('paths', {}).get(dir_key, f"{dir_key.replace('_dir', '')}s/")
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_model(self):
        """Initialize the enhanced model"""
        self.logger.info("Initializing enhanced model...")
        
        # Create model with enhancements
        enhancement_config = self.config.get('model', {}).get('enhancements', {})
        self.model = create_enhanced_model(
            model_name=self.config['model']['base_model'],
            **enhancement_config
        )
        
        self.model.to(self.device)
        
        # Enable mixed precision if configured
        if self.config.get('hardware', {}).get('mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def initialize_data_loaders(self):
        """Initialize data loaders"""
        self.logger.info("Initializing data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(self.config)
        
        self.logger.info(f"Data loaders initialized - Train: {len(self.train_loader)}, "
                        f"Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")
    
    def initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler"""
        # Optimizer
        optimizer_config = self.config.get('training', {})
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_config.get('learning_rate', 5e-5),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        scheduler_type = optimizer_config.get('scheduler', 'linear')
        total_steps = len(self.train_loader) * optimizer_config.get('num_epochs', 10)
        warmup_steps = optimizer_config.get('warmup_steps', 1000)
        
        if scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == 'cosine_with_restarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps // 4,
                T_mult=2
            )
        
        # Advanced schedulers
        curriculum_config = optimizer_config.get('curriculum_learning', {})
        if curriculum_config.get('enabled', False):
            self.curriculum_scheduler = CurriculumLearningScheduler(curriculum_config)
        
        teacher_forcing_config = optimizer_config.get('teacher_forcing', {})
        if teacher_forcing_config.get('enabled', False):
            self.teacher_forcing_scheduler = TeacherForcingScheduler(teacher_forcing_config)
        
        self.logger.info("Optimizer and schedulers initialized")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get curriculum learning max length
        if self.curriculum_scheduler:
            max_length = self.curriculum_scheduler.get_max_length(epoch)
            self.logger.info(f"Curriculum learning max length for epoch {epoch}: {max_length}")
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Filter by curriculum learning if enabled
            if self.curriculum_scheduler:
                max_length = self.curriculum_scheduler.get_max_length(epoch)
                valid_indices = batch['length'] <= max_length
                if not valid_indices.any():
                    continue
                
                # Filter batch
                batch = {k: v[valid_indices] if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Get teacher forcing ratio
            teacher_forcing_ratio = 1.0
            if self.teacher_forcing_scheduler:
                teacher_forcing_ratio = self.teacher_forcing_scheduler.get_teacher_forcing_ratio(
                    self.global_step
                )
            
            # Forward pass
            if self.config.get('hardware', {}).get('mixed_precision', False):
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('training', {}).get('gradient_clipping', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('training', {}).get('gradient_clipping', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'tf_ratio': f"{teacher_forcing_ratio:.3f}"
            })
            
            # Logging
            if self.global_step % self.config.get('training', {}).get('logging_steps', 100) == 0:
                self._log_metrics({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/teacher_forcing_ratio': teacher_forcing_ratio,
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
            
            # Save checkpoint
            if (self.global_step % self.config.get('training', {}).get('save_steps', 1000) == 0):
                self.save_checkpoint(f"checkpoint_step_{self.global_step}")
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and console"""
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.log(metrics, step=self.global_step)
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {self.global_step} - {metrics_str}")
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints/')
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'config': self.config
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Initialize all components
        self.initialize_model()
        self.initialize_data_loaders()
        self.initialize_optimizer_and_scheduler()
        
        num_epochs = self.config.get('training', {}).get('num_epochs', 10)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.get('training', {}).get('eval_steps', 1) == 0:
                val_metrics = self.evaluate(self.val_loader, split='val')
                
                # Check if best model
                val_score = val_metrics.get('rouge_l', 0.0)
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.save_checkpoint("best_model")
                    self.logger.info(f"New best model saved with ROUGE-L: {val_score:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}")
        
        self.logger.info("Training completed!")
    
    def evaluate(self, dataloader, split='val') -> Dict[str, float]:
        """Evaluate model on given dataloader"""
        self.model.eval()
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate predictions
                generated = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.config.get('model', {}).get('max_output_length', 256),
                    num_beams=self.config.get('evaluation', {}).get('generation', {}).get('num_beams', 4),
                    length_penalty=self.config.get('evaluation', {}).get('generation', {}).get('length_penalty', 2.0),
                    repetition_penalty=self.config.get('evaluation', {}).get('generation', {}).get('repetition_penalty', 1.2),
                    no_repeat_ngram_size=self.config.get('evaluation', {}).get('generation', {}).get('no_repeat_ngram_size', 3),
                    early_stopping=self.config.get('evaluation', {}).get('generation', {}).get('early_stopping', True)
                )
                
                # Decode predictions and references
                batch_predictions = self.model.tokenizer.batch_decode(generated, skip_special_tokens=True)
                batch_references = self.model.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(predictions, references)
        
        # Log metrics
        log_metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        self._log_metrics(log_metrics)
        
        return metrics


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Enhanced Vietnamese Summarization Model")
    parser.add_argument("--config", type=str, default="configs/enhanced_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = EnhancedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
