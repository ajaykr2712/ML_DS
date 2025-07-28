"""
Advanced Meta-Learning Framework
Implementation of Model-Agnostic Meta-Learning (MAML) and related algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict
import copy

class MAMLModel(nn.Module):
    """Base model for MAML implementation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MAML:
    """Model-Agnostic Meta-Learning implementation."""
    
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 num_inner_steps: int = 5,
                 first_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
    
    def inner_update(self, 
                     task_data: Tuple[torch.Tensor, torch.Tensor],
                     fast_weights: Optional[OrderedDict] = None) -> OrderedDict:
        """Perform inner loop update for a single task."""
        x_support, y_support = task_data
        
        if fast_weights is None:
            fast_weights = OrderedDict(self.model.named_parameters())
        
        for step in range(self.num_inner_steps):
            # Forward pass with current weights
            logits = self._forward_with_weights(x_support, fast_weights)
            loss = F.cross_entropy(logits, y_support)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), 
                                      create_graph=not self.first_order)
            
            # Update fast weights
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )
        
        return fast_weights
    
    def meta_update(self, batch_tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                                torch.Tensor, torch.Tensor]]):
        """Perform meta-update across a batch of tasks."""
        self.meta_optimizer.zero_grad()
        
        total_loss = 0
        for task in batch_tasks:
            x_support, y_support, x_query, y_query = task
            
            # Inner loop adaptation
            fast_weights = self.inner_update((x_support, y_support))
            
            # Query loss with adapted weights
            query_logits = self._forward_with_weights(x_query, fast_weights)
            query_loss = F.cross_entropy(query_logits, y_query)
            total_loss += query_loss
        
        # Meta-gradient step
        total_loss /= len(batch_tasks)
        total_loss.backward()
        self.meta_optimizer.step()
        
        return total_loss.item()
    
    def _forward_with_weights(self, x: torch.Tensor, weights: OrderedDict) -> torch.Tensor:
        """Forward pass using specific weights."""
        # Simple implementation for linear layers
        output = x
        weight_items = list(weights.items())
        
        for i in range(0, len(weight_items), 2):  # Skip bias for simplicity
            if i + 1 < len(weight_items):
                weight_name, weight = weight_items[i]
                bias_name, bias = weight_items[i + 1]
                
                if 'weight' in weight_name:
                    output = F.linear(output, weight, bias)
                    if i < len(weight_items) - 2:  # Apply ReLU except last layer
                        output = F.relu(output)
        
        return output

class PrototypicalNetworks:
    """Prototypical Networks for few-shot learning."""
    
    def __init__(self, encoder: nn.Module, distance_metric: str = 'euclidean'):
        self.encoder = encoder
        self.distance_metric = distance_metric
    
    def compute_prototypes(self, 
                          support_embeddings: torch.Tensor, 
                          support_labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes from support set."""
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = (support_labels == label)
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def compute_distances(self, 
                         query_embeddings: torch.Tensor, 
                         prototypes: torch.Tensor) -> torch.Tensor:
        """Compute distances between queries and prototypes."""
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(query_embeddings, prototypes)
        elif self.distance_metric == 'cosine':
            query_norm = F.normalize(query_embeddings, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            distances = 1 - torch.mm(query_norm, proto_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(self, 
                support_data: torch.Tensor, 
                support_labels: torch.Tensor,
                query_data: torch.Tensor) -> torch.Tensor:
        """Forward pass for prototypical networks."""
        # Encode support and query sets
        support_embeddings = self.encoder(support_data)
        query_embeddings = self.encoder(query_data)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        # Compute distances and convert to logits
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances  # Negative distance as logits
        
        return logits

class ReptileOptimizer:
    """Reptile meta-learning algorithm."""
    
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 num_inner_steps: int = 10):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
    
    def meta_update(self, batch_tasks: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Perform Reptile meta-update."""
        for task_data in batch_tasks:
            # Clone model for task-specific adaptation
            task_model = copy.deepcopy(self.model)
            task_optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)
            
            # Inner loop adaptation
            x_task, y_task = task_data
            for _ in range(self.num_inner_steps):
                task_optimizer.zero_grad()
                logits = task_model(x_task)
                loss = F.cross_entropy(logits, y_task)
                loss.backward()
                task_optimizer.step()
            
            # Reptile update: move towards adapted parameters
            for initial_param, adapted_param in zip(self.model.parameters(), 
                                                  task_model.parameters()):
                initial_param.data += self.outer_lr * (adapted_param.data - initial_param.data)

class MetaLearningFramework:
    """Unified meta-learning framework."""
    
    def __init__(self, 
                 algorithm: str = 'maml',
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None):
        self.algorithm = algorithm
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize algorithm
        self.meta_learner = self._create_meta_learner()
    
    def _create_model(self) -> nn.Module:
        """Create the base model."""
        return MAMLModel(
            input_dim=self.model_config.get('input_dim', 784),
            hidden_dim=self.model_config.get('hidden_dim', 128),
            output_dim=self.model_config.get('output_dim', 10),
            num_layers=self.model_config.get('num_layers', 3)
        )
    
    def _create_meta_learner(self):
        """Create the meta-learning algorithm."""
        if self.algorithm == 'maml':
            return MAML(
                model=self.model,
                inner_lr=self.training_config.get('inner_lr', 0.01),
                outer_lr=self.training_config.get('outer_lr', 0.001),
                num_inner_steps=self.training_config.get('num_inner_steps', 5)
            )
        elif self.algorithm == 'prototypical':
            return PrototypicalNetworks(
                encoder=self.model,
                distance_metric=self.training_config.get('distance_metric', 'euclidean')
            )
        elif self.algorithm == 'reptile':
            return ReptileOptimizer(
                model=self.model,
                inner_lr=self.training_config.get('inner_lr', 0.01),
                outer_lr=self.training_config.get('outer_lr', 0.001),
                num_inner_steps=self.training_config.get('num_inner_steps', 10)
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, meta_dataset: List[List[Tuple]], num_epochs: int = 1000):
        """Train the meta-learning model."""
        for epoch in range(num_epochs):
            # Sample batch of tasks
            batch_tasks = np.random.choice(meta_dataset, 
                                         size=self.training_config.get('batch_size', 4),
                                         replace=False)
            
            if self.algorithm in ['maml', 'reptile']:
                loss = self.meta_learner.meta_update(batch_tasks)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
            else:
                # Prototypical networks training would be different
                pass
    
    def adapt_to_task(self, 
                      support_data: torch.Tensor, 
                      support_labels: torch.Tensor,
                      num_adaptation_steps: int = 5) -> nn.Module:
        """Adapt the model to a new task."""
        if self.algorithm == 'maml':
            adapted_weights = self.meta_learner.inner_update(
                (support_data, support_labels)
            )
            # Return model with adapted weights
            adapted_model = copy.deepcopy(self.model)
            adapted_model.load_state_dict(adapted_weights)
            return adapted_model
        else:
            # For other algorithms, return the base model
            return self.model
    
    def evaluate_task(self, 
                      support_data: torch.Tensor,
                      support_labels: torch.Tensor,
                      query_data: torch.Tensor,
                      query_labels: torch.Tensor) -> float:
        """Evaluate performance on a new task."""
        if self.algorithm == 'prototypical':
            logits = self.meta_learner.forward(support_data, support_labels, query_data)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_labels).float().mean()
            return accuracy.item()
        else:
            # Adapt and evaluate
            adapted_model = self.adapt_to_task(support_data, support_labels)
            with torch.no_grad():
                logits = adapted_model(query_data)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean()
            return accuracy.item()

# Example usage
if __name__ == "__main__":
    # Configuration
    model_config = {
        'input_dim': 784,
        'hidden_dim': 128,
        'output_dim': 5,  # 5-way classification
        'num_layers': 3
    }
    
    training_config = {
        'inner_lr': 0.01,
        'outer_lr': 0.001,
        'num_inner_steps': 5,
        'batch_size': 4
    }
    
    # Initialize framework
    framework = MetaLearningFramework(
        algorithm='maml',
        model_config=model_config,
        training_config=training_config
    )
    
    # Generate dummy data for testing
    def generate_dummy_task(n_way=5, k_shot=5, query_size=15):
        support_data = torch.randn(n_way * k_shot, 784)
        support_labels = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_data = torch.randn(n_way * query_size, 784)
        query_labels = torch.repeat_interleave(torch.arange(n_way), query_size)
        return (support_data, support_labels, query_data, query_labels)
    
    # Create meta-dataset
    meta_dataset = [generate_dummy_task() for _ in range(100)]
    
    print("Training meta-learning model...")
    framework.train(meta_dataset, num_epochs=200)
    
    # Test on new task
    test_task = generate_dummy_task()
    accuracy = framework.evaluate_task(*test_task)
    print(f"Test accuracy on new task: {accuracy:.4f}")
    
    print("Meta-learning framework implemented successfully! 🚀")
