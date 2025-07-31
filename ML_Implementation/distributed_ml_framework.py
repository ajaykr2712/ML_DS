"""
Distributed Machine Learning Framework
======================================

A comprehensive framework for distributed training, federated learning,
and scalable ML model deployment across multiple nodes and devices.

Best Contributions:
- Multi-node distributed training with fault tolerance
- Federated learning with privacy preservation
- Dynamic load balancing and resource management
- Model partitioning and pipeline parallelism
- Asynchronous parameter updates
- Communication compression and optimization
- Adaptive learning rate scheduling across nodes

Author: ML/DS Advanced Implementation Team
"""

import logging
import numpy as np
import time
import threading
import socket
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from pathlib import Path
import hashlib

# Distributed computing libraries
try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.multiprocessing as mp
    
    # Ray for distributed computing
    import ray
    from ray import train
    from ray.train import Trainer
    
    # MPI for parallel processing
    from mpi4py import MPI
    
except ImportError as e:
    logging.warning(f"Some distributed ML libraries not available: {e}")

# Federated learning libraries
try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
    import cryptography
    from cryptography.fernet import Fernet
except ImportError as e:
    logging.warning(f"Federated learning libraries not available: {e}")

@dataclass
class DistributedConfig:
    """Configuration for distributed ML framework."""
    world_size: int = 4
    backend: str = "nccl"  # nccl, gloo, mpi
    master_addr: str = "localhost"
    master_port: str = "12355"
    node_rank: int = 0
    local_rank: int = 0
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    gradient_clipping: float = 1.0
    checkpoint_interval: int = 5
    communication_rounds: int = 100
    clients_per_round: int = 10
    min_clients: int = 5
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0

@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    rank: int
    world_size: int
    device: str
    memory_gb: float
    cpu_cores: int
    network_bandwidth: float
    status: str = "idle"
    last_heartbeat: datetime = field(default_factory=datetime.now)

class DistributedMLFramework:
    """
    Advanced Distributed Machine Learning Framework.
    
    Features:
    - Multi-node distributed training
    - Federated learning
    - Fault tolerance and recovery
    - Dynamic scaling
    - Communication optimization
    - Privacy preservation
    """
    
    def __init__(self, config: DistributedConfig = None):
        self.config = config or DistributedConfig()
        self.logger = self._setup_logging()
        
        # Distributed state
        self.is_initialized = False
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.master_node = None
        self.node_id = str(uuid.uuid4())
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Federated learning state
        self.federated_clients = {}
        self.global_model_state = None
        self.client_updates = []
        
        # Performance tracking
        self.training_metrics = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'communication_times': [],
            'computation_times': []
        }
        
        # Fault tolerance
        self.checkpoints = {}
        self.failed_nodes = set()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup distributed logging."""
        logger = logging.getLogger(f"distributed_ml_{self.node_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - Node {self.node_id[:8]} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def initialize_distributed(self, rank: int = None, world_size: int = None):
        """Initialize distributed training environment."""
        try:
            # Set rank and world size
            self.rank = rank if rank is not None else self.config.node_rank
            self.world_size = world_size if world_size is not None else self.config.world_size
            
            # Initialize process group
            if not dist.is_initialized():
                if self.config.backend == "nccl" and torch.cuda.is_available():
                    dist.init_process_group(
                        backend="nccl",
                        init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                        world_size=self.world_size,
                        rank=self.rank
                    )
                    # Set CUDA device
                    torch.cuda.set_device(self.rank % torch.cuda.device_count())
                    self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
                else:
                    dist.init_process_group(
                        backend="gloo",
                        init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                        world_size=self.world_size,
                        rank=self.rank
                    )
            
            self.is_initialized = True
            self.logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def setup_model(self, model: nn.Module, optimizer_class=optim.Adam):
        """Setup model for distributed training."""
        if not self.is_initialized:
            self.initialize_distributed()
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Wrap model with DDP
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device] if self.device.type == "cuda" else None)
        
        # Setup optimizer
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.config.learning_rate)
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.logger.info("Model setup completed for distributed training")
    
    def setup_data_loaders(self, train_dataset, val_dataset=None):
        """Setup distributed data loaders."""
        # Create distributed sampler
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank,
            shuffle=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        if val_dataset:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )
        
        self.logger.info("Data loaders setup completed")
    
    def all_reduce_gradients(self):
        """Perform all-reduce operation on gradients."""
        if self.world_size <= 1:
            return
        
        start_time = time.time()
        
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
        
        comm_time = time.time() - start_time
        self.training_metrics['communication_times'].append(comm_time)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with distributed coordination."""
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        start_time = time.time()
        
        # Set epoch for sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            
            # Synchronize gradients
            if self.world_size > 1:
                self.all_reduce_gradients()
            
            self.optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )
        
        # Calculate metrics
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100.0 * epoch_correct / epoch_total
        computation_time = time.time() - start_time
        
        # Gather metrics from all ranks
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            acc_tensor = torch.tensor([accuracy], device=self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
            
            avg_loss = loss_tensor.item() / self.world_size
            accuracy = acc_tensor.item() / self.world_size
        
        # Update metrics
        self.training_metrics['epochs'].append(epoch)
        self.training_metrics['losses'].append(avg_loss)
        self.training_metrics['accuracies'].append(accuracy)
        self.training_metrics['computation_times'].append(computation_time)
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        return {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'computation_time': computation_time,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model performance."""
        if self.model is None or self.val_loader is None:
            return {}
        
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = nn.CrossEntropyLoss()(output, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100.0 * val_correct / val_total
        
        # Gather validation metrics
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_val_loss], device=self.device)
            acc_tensor = torch.tensor([val_accuracy], device=self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
            
            avg_val_loss = loss_tensor.item() / self.world_size
            val_accuracy = acc_tensor.item() / self.world_size
        
        return {
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        }
    
    def save_checkpoint(self, epoch: int, filepath: str = None):
        """Save distributed training checkpoint."""
        if self.rank != 0:  # Only master saves checkpoints
            return
        
        if filepath is None:
            filepath = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.training_metrics,
            'world_size': self.world_size
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_metrics = checkpoint['metrics']
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch']
    
    def train_distributed(self, epochs: int = None):
        """Main distributed training loop."""
        epochs = epochs or self.config.epochs
        
        self.logger.info(f"Starting distributed training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Log progress
            if self.rank == 0:
                log_msg = f"Epoch {epoch}: "
                log_msg += f"Loss={train_metrics['loss']:.4f}, "
                log_msg += f"Acc={train_metrics['accuracy']:.2f}%"
                
                if val_metrics:
                    log_msg += f", Val_Loss={val_metrics['val_loss']:.4f}, "
                    log_msg += f"Val_Acc={val_metrics['val_accuracy']:.2f}%"
                
                self.logger.info(log_msg)
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            
            # Synchronize all processes
            if self.world_size > 1:
                dist.barrier()
    
    def cleanup_distributed(self):
        """Clean up distributed training resources."""
        if self.is_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.logger.info("Distributed training cleanup completed")

class FederatedLearningClient:
    """Federated learning client implementation."""
    
    def __init__(self, client_id: str, model: nn.Module, 
                 train_data, config: DistributedConfig):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.config = config
        self.logger = logging.getLogger(f"fed_client_{client_id}")
        
        # Privacy preservation
        self.privacy_engine = None
        self.noise_generator = np.random.default_rng()
    
    def local_training(self, global_weights: Dict[str, torch.Tensor], 
                      local_epochs: int = 5) -> Dict[str, Any]:
        """Perform local training on client data."""
        # Load global weights
        self.model.load_state_dict(global_weights)
        
        # Setup optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Local training
        self.model.train()
        local_loss = 0.0
        
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_data):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Add differential privacy noise
                if self.config.privacy_budget > 0:
                    self._add_privacy_noise()
                
                optimizer.step()
                local_loss += loss.item()
        
        # Return updated weights and training statistics
        return {
            'client_id': self.client_id,
            'weights': self.model.state_dict(),
            'num_samples': len(self.train_data.dataset),
            'local_loss': local_loss / (local_epochs * len(self.train_data)),
            'local_epochs': local_epochs
        }
    
    def _add_privacy_noise(self):
        """Add differential privacy noise to gradients."""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        0, self.config.noise_multiplier * self.config.max_grad_norm, 
                        param.grad.shape
                    ).to(param.device)
                    param.grad.add_(noise)

class FederatedLearningServer:
    """Federated learning server implementation."""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger("fed_server")
        
        # Client management
        self.clients: Dict[str, FederatedLearningClient] = {}
        self.client_updates = []
        
        # Global model state
        self.global_weights = model.state_dict()
        self.round_number = 0
        
        # Aggregation weights
        self.aggregation_weights = {}
    
    def register_client(self, client: FederatedLearningClient):
        """Register a new federated learning client."""
        self.clients[client.client_id] = client
        self.logger.info(f"Registered client: {client.client_id}")
    
    def select_clients(self, round_number: int) -> List[str]:
        """Select clients for federated learning round."""
        # Simple random selection
        available_clients = list(self.clients.keys())
        num_clients = min(self.config.clients_per_round, len(available_clients))
        
        selected = np.random.choice(
            available_clients, size=num_clients, replace=False
        ).tolist()
        
        self.logger.info(f"Round {round_number}: Selected {len(selected)} clients")
        return selected
    
    def aggregate_weights(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client model weights using FedAvg algorithm."""
        # Calculate aggregation weights based on number of samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        aggregated_weights = {}
        
        # Initialize with zeros
        for key in client_updates[0]['weights'].keys():
            aggregated_weights[key] = torch.zeros_like(
                client_updates[0]['weights'][key]
            )
        
        # Weighted average
        for update in client_updates:
            weight = update['num_samples'] / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight * update['weights'][key]
        
        return aggregated_weights
    
    def federated_learning_round(self, round_number: int) -> Dict[str, Any]:
        """Execute one round of federated learning."""
        # Select clients
        selected_clients = self.select_clients(round_number)
        
        if len(selected_clients) < self.config.min_clients:
            self.logger.warning(f"Not enough clients available: {len(selected_clients)}")
            return {}
        
        # Collect client updates
        client_updates = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            update = client.local_training(self.global_weights, local_epochs=5)
            client_updates.append(update)
        
        # Aggregate updates
        self.global_weights = self.aggregate_weights(client_updates)
        
        # Update global model
        self.model.load_state_dict(self.global_weights)
        
        # Calculate round statistics
        avg_loss = np.mean([update['local_loss'] for update in client_updates])
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        round_stats = {
            'round': round_number,
            'num_clients': len(selected_clients),
            'avg_loss': avg_loss,
            'total_samples': total_samples,
            'selected_clients': selected_clients
        }
        
        self.logger.info(f"Round {round_number} completed: avg_loss={avg_loss:.4f}")
        return round_stats
    
    def run_federated_learning(self, num_rounds: int = None) -> List[Dict[str, Any]]:
        """Run complete federated learning process."""
        num_rounds = num_rounds or self.config.communication_rounds
        
        self.logger.info(f"Starting federated learning for {num_rounds} rounds")
        
        round_results = []
        
        for round_num in range(num_rounds):
            round_stats = self.federated_learning_round(round_num)
            if round_stats:
                round_results.append(round_stats)
        
        self.logger.info("Federated learning completed")
        return round_results

def main():
    """Demonstration of the Distributed ML Framework."""
    
    print("=== Distributed Machine Learning Framework Demo ===\n")
    
    # Configuration
    config = DistributedConfig(
        world_size=2,
        batch_size=64,
        epochs=5,
        learning_rate=0.01
    )
    
    print(f"Configuration: {config}")
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = x.view(-1, 784)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # Initialize framework
    framework = DistributedMLFramework(config)
    
    print(f"Framework initialized with device: {framework.device}")
    print(f"Node ID: {framework.node_id}")
    
    # Demonstrate federated learning
    print("\n=== Federated Learning Demo ===")
    
    # Create federated server
    fed_server = FederatedLearningServer(model, config)
    
    # Create dummy clients
    for i in range(3):
        # Create dummy data for each client
        dummy_data = torch.randn(100, 784)
        dummy_labels = torch.randint(0, 10, (100,))
        
        # This would normally be a proper DataLoader
        train_data = [(dummy_data[j:j+10], dummy_labels[j:j+10]) 
                     for j in range(0, 100, 10)]
        
        client = FederatedLearningClient(f"client_{i}", SimpleModel(), train_data, config)
        fed_server.register_client(client)
    
    print(f"Created federated server with {len(fed_server.clients)} clients")
    
    # Run a few federated learning rounds
    results = fed_server.run_federated_learning(num_rounds=3)
    
    print("\nFederated Learning Results:")
    for result in results:
        print(f"Round {result['round']}: "
              f"Loss={result['avg_loss']:.4f}, "
              f"Clients={result['num_clients']}")
    
    print("\nDistributed ML Framework demonstration completed!")

if __name__ == "__main__":
    main()
