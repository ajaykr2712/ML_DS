"""
Advanced training utilities for generative AI models.
Includes modern training techniques, optimization strategies, and monitoring.
"""

import time
import warnings
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, 
    LambdaLR
)
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with: pip install wandb")

import numpy as np
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for advanced training."""
    
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd, rmsprop
    scheduler: str = "cosine"  # cosine, onecycle, plateau, lambda, multistep
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    # Advanced techniques
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    ema_decay: float = 0.999
    use_ema: bool = True
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0
    noise_scheduling: bool = False
    
    # Monitoring and checkpointing
    eval_every: int = 1000
    save_every: int = 5000
    log_every: int = 100
    max_checkpoints: int = 5
    
    # Paths
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Wandb
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Device and performance
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories
        for path in [self.output_dir, self.log_dir, self.checkpoint_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class EMAModel(nn.Module):
    """Exponential Moving Average model wrapper."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class AdvancedTrainer:
    """Advanced trainer with modern training techniques."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.metrics_fn = metrics_fn
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Setup EMA
        self.ema_model = EMAModel(model, config.ema_decay) if config.use_ema else None
        
        # Setup monitoring
        self.writer = SummaryWriter(config.log_dir)
        self._setup_wandb()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Timing
        self.start_time = None
        self.step_times = []
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                params, 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler."""
        total_steps = len(self.train_dataloader) * self.config.epochs
        
        if self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.scheduler.lower() == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif self.config.scheduler.lower() == "warmup":
            return LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(
                    (step + 1) / self.config.warmup_steps,
                    1.0
                )
            )
        return None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if WANDB_AVAILABLE and self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                tags=self.config.wandb_tags,
                config=self.config.__dict__
            )
    
    @contextmanager
    def autocast_context(self):
        """Context manager for mixed precision."""
        if self.config.mixed_precision:
            with autocast():
                yield
        else:
            yield
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with autocast
        with self.autocast_context():
            outputs = self.model(**batch)
            loss = self.loss_fn(outputs, batch.get('labels', batch.get('targets')))
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation steps
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            # Optimizer step
            if self.config.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Update EMA
            if self.ema_model:
                self.ema_model.update()
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Update learning rate
            if self.scheduler and self.config.scheduler != "plateau":
                self.scheduler.step()
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if not self.eval_dataloader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_metrics = {}
        
        # Use EMA model for evaluation if available
        if self.ema_model:
            self.ema_model.apply_shadow()
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                with self.autocast_context():
                    outputs = self.model(**batch)
                    loss = self.loss_fn(outputs, batch.get('labels', batch.get('targets')))
                
                batch_size = outputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Compute additional metrics
                if self.metrics_fn:
                    batch_metrics = self.metrics_fn(outputs, batch)
                    for key, value in batch_metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)
        
        # Restore original model parameters
        if self.ema_model:
            self.ema_model.restore()
        
        # Aggregate metrics
        eval_metrics = {'eval_loss': total_loss / total_samples}
        for key, values in all_metrics.items():
            eval_metrics[f'eval_{key}'] = np.mean(values)
        
        return eval_metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        self.start_time = time.time()
        
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Total steps: {len(self.train_dataloader) * self.config.epochs}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training loop
            train_losses = []
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.epochs}"
            )
            
            for batch in progress_bar:
                step_start_time = time.time()
                
                # Training step
                train_metrics = self.train_step(batch)
                train_losses.append(train_metrics['loss'])
                
                # Track step time
                step_time = time.time() - step_start_time
                self.step_times.append(step_time)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{train_metrics['loss']:.4f}",
                    'lr': f"{train_metrics['learning_rate']:.2e}",
                    'step_time': f"{step_time:.3f}s"
                })
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    self._log_metrics(train_metrics, prefix="train")
                
                # Evaluation
                if (self.global_step % self.config.eval_every == 0 and 
                    self.eval_dataloader):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")
                    
                    # Update learning rate scheduler
                    if (self.scheduler and 
                        self.config.scheduler == "plateau"):
                        self.scheduler.step(eval_metrics['eval_loss'])
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_loss:
                        self.best_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            # End of epoch logging
            epoch_time = time.time() - epoch_start_time
            avg_loss = np.mean(train_losses)
            
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            print(f"Average loss: {avg_loss:.4f}")
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'epoch_time': epoch_time,
                'global_step': self.global_step
            })
        
        # Final save
        self.save_checkpoint(is_best=False)
        
        # Cleanup
        self.writer.close()
        if WANDB_AVAILABLE and wandb.run:
            wandb.finish()
        
        return self.training_history
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to all monitoring systems."""
        # TensorBoard
        for key, value in metrics.items():
            log_key = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(log_key, value, self.global_step)
        
        # Wandb
        if WANDB_AVAILABLE and wandb.run:
            log_dict = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
            log_dict['global_step'] = self.global_step
            wandb.log(log_dict)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema_model.shadow if self.ema_model else None,
            'best_loss': self.best_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if len(checkpoint_files) > self.config.max_checkpoints:
            # Sort by step number
            checkpoint_files.sort(
                key=lambda x: int(x.stem.split('_')[-1])
            )
            
            # Remove oldest checkpoints
            for old_checkpoint in checkpoint_files[:-self.config.max_checkpoints]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if checkpoint['ema_state_dict'] and self.ema_model:
            self.ema_model.shadow = checkpoint['ema_state_dict']
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.step_times:
            return {}
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_training_time': total_time,
            'average_step_time': np.mean(self.step_times),
            'steps_per_second': 1.0 / np.mean(self.step_times),
            'total_steps': self.global_step,
            'epochs_completed': self.epoch,
            'best_loss': self.best_loss
        }


# Utility functions for common training scenarios

def create_text_generation_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    **kwargs
) -> AdvancedTrainer:
    """Create trainer optimized for text generation tasks."""
    
    config = TrainingConfig(
        optimizer="adamw",
        scheduler="cosine",
        learning_rate=5e-5,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        mixed_precision=True,
        use_ema=True,
        **kwargs
    )
    
    def text_loss_fn(outputs, targets):
        # Flatten for language modeling loss
        return F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
    
    return AdvancedTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        loss_fn=text_loss_fn
    )


def create_diffusion_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    **kwargs
) -> AdvancedTrainer:
    """Create trainer optimized for diffusion models."""
    
    config = TrainingConfig(
        optimizer="adamw",
        scheduler="cosine",
        learning_rate=2e-4,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        mixed_precision=True,
        use_ema=True,
        ema_decay=0.995,
        **kwargs
    )
    
    def diffusion_loss_fn(outputs, targets):
        # MSE loss for diffusion models
        return F.mse_loss(outputs, targets)
    
    return AdvancedTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        loss_fn=diffusion_loss_fn
    )
