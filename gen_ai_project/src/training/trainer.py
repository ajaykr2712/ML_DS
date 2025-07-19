"""
Training infrastructure for generative models.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
import yaml
from tqdm import tqdm
import wandb


class BaseTrainer:
    """Base trainer class for all generative models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Setup checkpointing
        self.checkpoint_dir = config.get('checkpointing', {}).get('save_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup TensorBoard
        log_dir = log_config.get('log_dir', 'logs')
        self.writer = SummaryWriter(log_dir)
        
        # Setup W&B if configured
        wandb_config = log_config.get('wandb')
        if wandb_config:
            wandb.init(
                project=wandb_config.get('project'),
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags', []),
                config=self.config
            )
    
    def setup_monitoring(self):
        """Setup model monitoring."""
        pass  # Override in subclasses
    
    def save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None):
        """Save model checkpoint."""
        if epoch is None:
            epoch = self.epoch
            
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self.cleanup_checkpoints()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        keep_last_n = self.config.get('checkpointing', {}).get('keep_last_n', 3)
        
        # Get all checkpoint files
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir) 
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
        ]
        
        if len(checkpoint_files) > keep_last_n:
            # Sort by epoch number
            checkpoint_files.sort(
                key=lambda x: int(x.split('_')[2].split('.')[0])
            )
            
            # Remove oldest checkpoints
            for file_to_remove in checkpoint_files[:-keep_last_n]:
                file_path = os.path.join(self.checkpoint_dir, file_to_remove)
                os.remove(file_path)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all monitoring systems."""
        if step is None:
            step = self.step
            
        # TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        
        # W&B
        if wandb.run is not None:
            wandb.log(metrics, step=step)
        
        # Console
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch. Override in subclasses."""
        raise NotImplementedError
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model. Override in subclasses."""
        raise NotImplementedError
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config.get('training', {}).get('max_epochs', 100)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.log_metrics({f'train/{k}': v for k, v in train_metrics.items()})
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.log_metrics({f'val/{k}': v for k, v in val_metrics.items()})
                
                # Check if this is the best model
                val_loss = val_metrics.get('loss', float('inf'))
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
            else:
                is_best = False
            
            # Save checkpoint
            save_every_n = self.config.get('checkpointing', {}).get('save_every_n_epochs', 1)
            if (epoch + 1) % save_every_n == 0:
                self.save_checkpoint(is_best=is_best)
        
        self.logger.info("Training completed!")


class LanguageModelTrainer(BaseTrainer):
    """Trainer for language models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device = None):
        super().__init__(model, config, device)
        
        # Setup optimizer
        training_config = config.get('training', {})
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.get('learning_rate', 5e-5),
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999)),
            eps=training_config.get('eps', 1e-8)
        )
        
        # Setup scheduler
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 10000),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        # Mixed precision
        self.use_mixed_precision = training_config.get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.model.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train language model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        labels.view(-1)
                    )
            else:
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / num_batches})
        
        return {'loss': total_loss / num_batches}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate language model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)
                
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'loss': total_loss / num_batches}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_trainer(
    model_type: str, 
    model: nn.Module, 
    config: Dict[str, Any]
) -> BaseTrainer:
    """Factory function to create appropriate trainer."""
    trainers = {
        'language_model': LanguageModelTrainer,
        'gpt': LanguageModelTrainer,
        'transformer': LanguageModelTrainer,
    }
    
    trainer_class = trainers.get(model_type, BaseTrainer)
    return trainer_class(model, config)


if __name__ == "__main__":
    # Example usage
    from ..models.transformer import GPTModel
    
    # Load config
    config = load_config("config/gpt_config.yaml")
    
    # Create model and trainer
    model = GPTModel(vocab_size=50257)
    trainer = create_trainer('gpt', model, config)
    
    print(f"Trainer created: {type(trainer).__name__}")
    print(f"Device: {trainer.device}")
    print(f"Optimizer: {type(trainer.optimizer).__name__}")
