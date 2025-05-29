"""
Enhanced Trainer for Vietnamese Summarization

Implements advanced training techniques including curriculum learning,
mixed precision training, and comprehensive evaluation.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from typing import Dict, Optional
from datetime import datetime
import json

from ..models.enhanced_t5 import create_enhanced_model
from ..data.dataset import create_data_loaders
from ..evaluation.metrics import SummarizationEvaluator

logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """Enhanced trainer with advanced techniques."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.logger = logging.getLogger(__name__)
        
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
        
        # Evaluation
        self.evaluator = SummarizationEvaluator(config)
        
        # Setup directories
        self._setup_directories()
        
    def _setup_device(self) -> torch.device:
        """Setup computing device."""
        device_config = self.config.get('hardware', {}).get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = ['checkpoints', 'results', 'logs']
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_model(self):
        """Initialize the enhanced model."""
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
        """Initialize data loaders."""
        self.logger.info("Initializing data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(self.config)
        
        self.logger.info(f"Data loaders initialized - Train: {len(self.train_loader)}, "
                        f"Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")
    
    def initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        # Optimizer
        training_config = self.config.get('training', {})
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 5e-5),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * training_config.get('num_epochs', 3)
        warmup_steps = training_config.get('warmup_steps', 100)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info("Optimizer and scheduler initialized")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
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
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    def evaluate(self, dataloader, split='val') -> Dict[str, float]:
        """Evaluate model on given dataloader."""
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
                    max_length=self.config.get('model', {}).get('max_output_length', 128),
                    num_beams=4,
                    length_penalty=2.0,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
                
                # Decode predictions and references
                batch_predictions = self.model.tokenizer.batch_decode(generated, skip_special_tokens=True)
                batch_references = self.model.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(predictions, references)
        
        return metrics
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join('checkpoints', f"{checkpoint_name}.pt")
        
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
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Initialize all components
        self.initialize_model()
        self.initialize_data_loaders()
        self.initialize_optimizer_and_scheduler()
        
        num_epochs = self.config.get('training', {}).get('num_epochs', 3)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Validation
            val_metrics = self.evaluate(self.val_loader, split='val')
            self.logger.info(f"Epoch {epoch} - Val Metrics: {val_metrics}")
            
            # Check if best model
            val_score = val_metrics.get('composite_score', 0.0)
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.save_checkpoint("best_model")
                self.logger.info(f"New best model saved with score: {val_score:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}")
        
        self.logger.info("Training completed!")
        
        # Final evaluation on test set
        self.logger.info("Running final evaluation on test set...")
        test_metrics = self.evaluate(self.test_loader, split='test')
        self.evaluator.print_metrics_summary(test_metrics)
        
        # Save final results
        results = {
            'test_metrics': test_metrics,
            'best_val_score': self.best_val_score,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join('results', f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Final results saved to {results_path}")
