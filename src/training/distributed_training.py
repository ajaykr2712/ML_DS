"""
Distributed Training Framework
Implements distributed training capabilities for large-scale ML models.
"""

import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Callable, Tuple
import os
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import socket
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    init_method: str = 'env://'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    accumulation_steps: int = 1
    checkpoint_freq: int = 1000
    log_freq: int = 100

class DistributedTrainer(ABC):
    """Abstract base class for distributed training."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.rank = config.rank
        self.local_rank = config.local_rank
        self.world_size = config.world_size
        self.device = None
        self.model = None
        self.optimizer = None
        self.scaler = None
        
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the model to be trained."""
        pass
    
    @abstractmethod
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for the model."""
        pass
    
    @abstractmethod
    def compute_loss(self, model_output: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a batch."""
        pass
    
    def setup_distributed(self):
        """Setup distributed training environment."""
        if self.world_size > 1:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
                self.device = torch.device(f'cuda:{self.config.local_rank}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Rank {self.rank}: Using device {self.device}")
    
    def setup_model_and_optimizer(self):
        """Setup model and optimizer for distributed training."""
        # Create model
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        # Wrap model with DDP if distributed
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=True
            )
        
        # Create optimizer
        self.optimizer = self.create_optimizer(self.model)
        
        # Setup mixed precision
        if self.config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        batch_count = 0
        
        # Set epoch for DistributedSampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(batch)
                    loss = self.compute_loss(output, batch)
                    loss = loss / self.config.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clipping
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training
                output = self.model(batch)
                loss = self.compute_loss(output, batch)
                loss = loss / self.config.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    if self.config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clipping
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            batch_size = self._get_batch_size(batch)
            total_loss += loss.item() * self.config.accumulation_steps * batch_size
            total_samples += batch_size
            batch_count += 1
            
            # Logging
            if batch_idx % self.config.log_freq == 0 and self.rank == 0:
                avg_loss = total_loss / total_samples
                elapsed_time = time.time() - start_time
                samples_per_sec = total_samples / elapsed_time
                
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {avg_loss:.6f}, "
                    f"Samples/sec: {samples_per_sec:.2f}"
                )
        
        # Synchronize losses across processes
        avg_loss = total_loss / total_samples
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        
        epoch_time = time.time() - start_time
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'samples_per_second': total_samples / epoch_time,
            'batches_processed': batch_count
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                if self.config.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = self.model(batch)
                        loss = self.compute_loss(output, batch)
                else:
                    output = self.model(batch)
                    loss = self.compute_loss(output, batch)
                
                batch_size = self._get_batch_size(batch)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        
        # Synchronize losses across processes
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], checkpoint_path: str):
        """Save training checkpoint."""
        if self.rank == 0:  # Only save from main process
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
        
        return epoch
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device, non_blocking=True)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Get batch size from batch."""
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        return 1
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.world_size > 1:
            dist.destroy_process_group()

class ModelParallelTrainer(DistributedTrainer):
    """Trainer for model parallelism across multiple GPUs."""
    
    def __init__(self, config: DistributedConfig, model_parallel_size: int = 2):
        super().__init__(config)
        self.model_parallel_size = model_parallel_size
    
    def setup_model_parallel(self):
        """Setup model parallelism."""
        if torch.cuda.device_count() < self.model_parallel_size:
            raise ValueError(f"Need at least {self.model_parallel_size} GPUs for model parallelism")
        
        # Split model across devices
        self.devices = [torch.device(f'cuda:{i}') for i in range(self.model_parallel_size)]
        
    def create_pipeline_parallel_model(self, model_layers: List[nn.Module]) -> nn.Module:
        """Create pipeline parallel model."""
        class PipelineParallelModel(nn.Module):
            def __init__(self, layers, devices):
                super().__init__()
                self.layers = nn.ModuleList()
                self.devices = devices
                
                # Distribute layers across devices
                layers_per_device = len(layers) // len(devices)
                
                for i, layer in enumerate(layers):
                    device_idx = min(i // layers_per_device, len(devices) - 1)
                    layer = layer.to(devices[device_idx])
                    self.layers.append(layer)
            
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    # Move input to correct device if needed
                    layer_device = next(layer.parameters()).device
                    if x.device != layer_device:
                        x = x.to(layer_device)
                    
                    x = layer(x)
                
                return x
        
        return PipelineParallelModel(model_layers, self.devices)

class FederatedTrainer:
    """Trainer for federated learning scenarios."""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.model = model
        self.config = config
        self.client_models = {}
        self.global_round = 0
    
    def federated_averaging(self, client_weights: List[Dict[str, torch.Tensor]], 
                          client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Perform federated averaging of client model weights."""
        total_samples = sum(client_sizes)
        
        # Initialize averaged weights
        avg_weights = {}
        for key in client_weights[0].keys():
            avg_weights[key] = torch.zeros_like(client_weights[0][key])
        
        # Weight by number of samples
        for i, weights in enumerate(client_weights):
            weight = client_sizes[i] / total_samples
            for key in weights.keys():
                avg_weights[key] += weight * weights[key]
        
        return avg_weights
    
    def update_global_model(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]]):
        """Update global model with client updates."""
        client_weights = [update[0] for update in client_updates]
        client_sizes = [update[1] for update in client_updates]
        
        # Perform federated averaging
        avg_weights = self.federated_averaging(client_weights, client_sizes)
        
        # Update global model
        self.model.load_state_dict(avg_weights)
        self.global_round += 1
        
        logger.info(f"Global model updated, round {self.global_round}")
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        return self.model.state_dict()

def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    return port

@contextmanager
def distributed_context(world_size: int, rank: int, backend: str = 'nccl'):
    """Context manager for distributed training setup and cleanup."""
    if world_size > 1:
        # Setup
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(find_free_port())
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        try:
            yield
        finally:
            # Cleanup
            dist.destroy_process_group()
    else:
        yield

def launch_distributed_training(trainer_class, config: DistributedConfig, 
                               training_fn: Callable, **kwargs):
    """Launch distributed training across multiple processes."""
    if config.world_size <= 1:
        # Single process training
        trainer = trainer_class(config)
        training_fn(trainer, **kwargs)
        return
    
    # Multi-process training
    def worker(rank: int):
        config.rank = rank
        config.local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        trainer = trainer_class(config)
        trainer.setup_distributed()
        trainer.setup_model_and_optimizer()
        
        try:
            training_fn(trainer, **kwargs)
        finally:
            trainer.cleanup()
    
    # Spawn processes
    mp.spawn(worker, args=(), nprocs=config.world_size, join=True)

# Example implementation
class ExampleDistributedTrainer(DistributedTrainer):
    """Example implementation of distributed trainer."""
    
    def __init__(self, config: DistributedConfig, input_dim: int = 784, 
                 hidden_dim: int = 512, output_dim: int = 10):
        super().__init__(config)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def create_model(self) -> nn.Module:
        """Create a simple neural network."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create Adam optimizer."""
        return torch.optim.Adam(model.parameters(), lr=0.001)
    
    def compute_loss(self, model_output: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return nn.CrossEntropyLoss()(model_output, batch['targets'])

def demo_distributed_training():
    """Demonstrate distributed training capabilities."""
    config = DistributedConfig(
        world_size=1,  # Single GPU for demo
        mixed_precision=True,
        gradient_clipping=1.0,
        accumulation_steps=2,
        log_freq=10
    )
    
    def training_function(trainer: ExampleDistributedTrainer):
        """Example training function."""
        # Create dummy data
        train_data = []
        for _ in range(100):
            x = torch.randn(32, 784)  # Batch of 32 samples
            y = torch.randint(0, 10, (32,))  # Random labels
            train_data.append({'inputs': x, 'targets': y})
        
        # Train for a few batches
        for epoch in range(2):
            trainer.model.train()
            for i, batch in enumerate(train_data[:10]):  # Only first 10 batches
                batch = trainer._move_batch_to_device(batch)
                
                # Forward pass
                outputs = trainer.model(batch['inputs'])
                loss = trainer.compute_loss(outputs, batch)
                
                # Backward pass
                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                
                if i % 5 == 0:
                    logger.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
    
    # Create and setup trainer
    trainer = ExampleDistributedTrainer(config)
    trainer.setup_distributed()
    trainer.setup_model_and_optimizer()
    
    # Run training
    training_function(trainer)
    
    logger.info("Distributed training demo completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_distributed_training()
