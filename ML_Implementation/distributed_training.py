"""
Distributed Training Framework
Advanced distributed training implementation for large-scale ML models
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
from typing import Dict, Optional, Callable
import os
import time
from dataclasses import dataclass
import socket

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    backend: str = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    checkpoint_freq: int = 1000
    log_freq: int = 100

class DistributedTrainer:
    """
    Distributed training framework supporting various parallelism strategies
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: DistributedConfig,
                 optimizer_class: type = torch.optim.Adam,
                 loss_fn: Callable = nn.CrossEntropyLoss(),
                 scheduler_class: Optional[type] = None):
        """
        Initialize distributed trainer
        
        Args:
            model: PyTorch model to train
            config: Distributed training configuration
            optimizer_class: Optimizer class
            loss_fn: Loss function
            scheduler_class: Learning rate scheduler class
        """
        self.model = model
        self.config = config
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.scheduler_class = scheduler_class
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf')
        self.training_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    def setup_distributed(self):
        """Setup distributed training environment"""
        # Set environment variables
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            rank=self.config.rank,
            world_size=self.config.world_size
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f'cuda:{self.config.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        self.logger.info(f"Initialized distributed training on rank {self.config.rank}")
    
    def setup_model(self):
        """Setup model for distributed training"""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP
        if self.config.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True
            )
        
        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Setup scheduler
        if self.scheduler_class:
            self.scheduler = self.scheduler_class(self.optimizer)
        else:
            self.scheduler = None
        
        self.logger.info(f"Model setup completed on rank {self.config.rank}")
    
    def setup_data(self, train_dataset, val_dataset=None):
        """Setup data loaders for distributed training"""
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True
        )
        
        val_sampler = None
        if val_dataset:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=False
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )
        
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        
        self.logger.info(f"Data loaders setup completed on rank {self.config.rank}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target) / self.config.gradient_accumulation_steps
            else:
                output = self.model(data)
                loss = self.loss_fn(output, target) / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step()
            
            # Accumulate metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_samples += data.size(0)
            
            # Logging
            if batch_idx % self.config.log_freq == 0 and self.config.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch: {self.epoch}, Batch: {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.6f}, LR: {current_lr:.6f}, "
                    f"Step: {self.global_step}"
                )
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_freq == 0 and self.config.rank == 0:
                self.save_checkpoint()
        
        # Calculate average loss across all processes
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.config.world_size
        
        epoch_time = time.time() - start_time
        throughput = total_samples / epoch_time
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'throughput': throughput,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.loss_fn(output, target)
                else:
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        # Aggregate metrics across all processes
        metrics_tensor = torch.tensor([total_loss, correct, total], dtype=torch.float).to(self.device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = metrics_tensor[0].item() / len(self.val_loader) / self.config.world_size
        accuracy = metrics_tensor[1].item() / metrics_tensor[2].item()
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(self, train_dataset, val_dataset=None):
        """Main training loop"""
        self.setup_distributed()
        self.setup_model()
        self.setup_data(train_dataset, val_dataset)
        
        self.logger.info(f"Starting training on rank {self.config.rank}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            self.training_history.append(epoch_metrics)
            
            # Log metrics (only on rank 0)
            if self.config.rank == 0:
                self.logger.info(f"Epoch {epoch} completed: {epoch_metrics}")
                
                # Save best model
                current_metric = val_metrics.get('val_accuracy', train_metrics['loss'])
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint(is_best=True)
        
        self.cleanup()
        
        if self.config.rank == 0:
            self.logger.info("Training completed successfully")
            return self.training_history
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if self.config.rank != 0:
            return
        
        # Get model state dict (unwrap DDP if necessary)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = f'checkpoint_epoch_{self.epoch}_step_{self.global_step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'best_model.pth')
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.config.world_size > 1:
            dist.destroy_process_group()

class DataParallelTrainer:
    """Simple data parallel trainer for single-node multi-GPU training"""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model for data parallel training
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
        
        self.model = self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_loader, val_loader=None, epochs: int = 10):
        """Training loop for data parallel training"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    self.logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}")
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.6f}")

def launch_distributed_training(
    train_fn: Callable,
    world_size: int,
    args: tuple = ()
):
    """
    Launch distributed training using multiprocessing
    
    Args:
        train_fn: Training function to execute
        world_size: Number of processes
        args: Additional arguments for training function
    """
    mp.spawn(
        train_fn,
        args=(world_size,) + args,
        nprocs=world_size,
        join=True
    )

def find_free_port() -> int:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Example usage
if __name__ == "__main__":
    # Example training function
    def train_worker(rank, world_size, model, train_dataset, val_dataset):
        config = DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            master_port=str(find_free_port()),
            batch_size=32,
            epochs=5,
            learning_rate=0.001
        )
        
        trainer = DistributedTrainer(model, config)
        history = trainer.train(train_dataset, val_dataset)
        
        return history
    
    # Create dummy model and datasets
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create dummy datasets
    train_data = torch.randn(1000, 784)
    train_targets = torch.randint(0, 10, (1000,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    
    val_data = torch.randn(200, 784)
    val_targets = torch.randint(0, 10, (200,))
    val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
    
    # Launch distributed training
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1:
        launch_distributed_training(
            train_worker,
            world_size,
            (model, train_dataset, val_dataset)
        )
    else:
        # Single GPU/CPU training
        history = train_worker(0, 1, model, train_dataset, val_dataset)
        print(f"Training completed. Final metrics: {history[-1] if history else 'No metrics'}")
