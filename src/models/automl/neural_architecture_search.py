"""
Neural Architecture Search (NAS) 2.0
Advanced implementation of differentiable and evolutionary NAS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
import random
from collections import namedtuple

# Define architecture components
ArchitectureConfig = namedtuple('ArchitectureConfig', [
    'num_layers', 'channels', 'operations', 'connections'
])

class SearchableOperation(nn.Module):
    """Differentiable operation for NAS."""
    
    OPERATIONS = {
        'conv3x3': lambda C_in, C_out, stride: nn.Conv2d(C_in, C_out, 3, stride, 1, bias=False),
        'conv1x1': lambda C_in, C_out, stride: nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False),
        'maxpool3x3': lambda C_in, C_out, stride: nn.MaxPool2d(3, stride, 1),
        'avgpool3x3': lambda C_in, C_out, stride: nn.AvgPool2d(3, stride, 1),
        'skip': lambda C_in, C_out, stride: nn.Identity() if stride == 1 and C_in == C_out else nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False),
        'sepconv3x3': lambda C_in, C_out, stride: SeparableConv2d(C_in, C_out, 3, stride, 1),
        'depthconv3x3': lambda C_in, C_out, stride: nn.Conv2d(C_in, C_out, 3, stride, 1, groups=C_in, bias=False),
        'zero': lambda C_in, C_out, stride: Zero()
    }
    
    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super().__init__()
        self.ops = nn.ModuleDict()
        self.alpha = nn.Parameter(torch.randn(len(self.OPERATIONS)))
        
        for name, op_func in self.OPERATIONS.items():
            self.ops[name] = op_func(C_in, C_out, stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted combination of operations."""
        if self.training:
            # Differentiable search: weighted sum of all operations
            weights = F.softmax(self.alpha, dim=0)
            output = sum(w * op(x) for w, (name, op) in zip(weights, self.ops.items()))
        else:
            # Evaluation: use the operation with highest weight
            best_op_idx = torch.argmax(self.alpha)
            best_op_name = list(self.ops.keys())[best_op_idx]
            output = self.ops[best_op_name](x)
        
        return output
    
    def get_best_operation(self) -> str:
        """Get the operation with highest alpha value."""
        best_idx = torch.argmax(self.alpha)
        return list(self.OPERATIONS.keys())[best_idx]

class SeparableConv2d(nn.Module):
    """Separable convolution operation."""
    
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False)
        self.pointwise = nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class Zero(nn.Module):
    """Zero operation for NAS."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

class SearchableCell(nn.Module):
    """Searchable cell containing multiple nodes and operations."""
    
    def __init__(self, C_in: int, C_out: int, num_nodes: int = 4, reduction: bool = False):
        super().__init__()
        self.num_nodes = num_nodes
        self.reduction = reduction
        stride = 2 if reduction else 1
        
        # Input preprocessing
        self.preprocess0 = nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False)
        self.preprocess1 = nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False)
        
        # Searchable operations between nodes
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(i + 2):  # Connect to previous nodes + 2 inputs
                self.ops.append(SearchableOperation(C_out, C_out, 1))
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Forward pass through the searchable cell."""
        # Preprocess inputs
        s0 = self.preprocess0(x0)
        s1 = self.preprocess1(x1)
        
        states = [s0, s1]
        op_idx = 0
        
        # Process each intermediate node
        for i in range(self.num_nodes):
            # Collect inputs from all previous states
            node_inputs = []
            for j in range(len(states)):
                node_inputs.append(self.ops[op_idx](states[j]))
                op_idx += 1
            
            # Sum all inputs to this node
            states.append(sum(node_inputs))
        
        # Concatenate all intermediate nodes (excluding inputs)
        return torch.cat(states[2:], dim=1)

class DARTSNetwork(nn.Module):
    """DARTS (Differentiable Architecture Search) network."""
    
    def __init__(self, 
                 C_in: int, 
                 num_classes: int, 
                 num_layers: int = 8,
                 C_init: int = 16,
                 num_nodes: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_init, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_init)
        )
        
        # Build cells
        self.cells = nn.ModuleList()
        C_curr = C_init
        
        for i in range(num_layers):
            reduction = i in [num_layers // 3, 2 * num_layers // 3]
            C_out = C_curr * 2 if reduction else C_curr
            
            cell = SearchableCell(C_curr, C_out, num_nodes, reduction)
            self.cells.append(cell)
            C_curr = C_out * num_nodes  # Account for concatenation
        
        # Global average pooling and classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_curr, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DARTS network."""
        s0 = s1 = self.stem(x)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        
        # Global pooling and classification
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits
    
    def get_architecture(self) -> Dict[str, Any]:
        """Extract the current best architecture."""
        architecture = {}
        
        for i, cell in enumerate(self.cells):
            cell_arch = []
            for j, op in enumerate(cell.ops):
                best_op = op.get_best_operation()
                cell_arch.append(best_op)
            architecture[f'cell_{i}'] = cell_arch
        
        return architecture

class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search."""
    
    def __init__(self, 
                 search_space: Dict[str, List[Any]],
                 population_size: int = 50,
                 num_generations: int = 100,
                 mutation_rate: float = 0.1):
        self.search_space = search_space
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_scores = []
    
    def initialize_population(self):
        """Initialize random population of architectures."""
        self.population = []
        for _ in range(self.population_size):
            individual = self._random_architecture()
            self.population.append(individual)
    
    def _random_architecture(self) -> Dict[str, Any]:
        """Generate a random architecture."""
        arch = {}
        for param, choices in self.search_space.items():
            arch[param] = random.choice(choices)
        return arch
    
    def evaluate_fitness(self, architecture: Dict[str, Any]) -> float:
        """Evaluate fitness of an architecture (simplified)."""
        # In practice, this would train and evaluate the model
        # For demo, we use a simple heuristic
        score = 0
        
        # Prefer deeper networks
        score += architecture.get('num_layers', 0) * 0.1
        
        # Prefer moderate channel sizes
        channels = architecture.get('channels', 32)
        score += max(0, 1 - abs(channels - 64) / 64)
        
        # Add some randomness to simulate actual performance
        score += np.random.normal(0, 0.1)
        
        return max(0, score)
    
    def select_parents(self, k: int = 3) -> List[Dict[str, Any]]:
        """Tournament selection for parents."""
        parents = []
        for _ in range(2):  # Select 2 parents
            tournament = random.sample(
                list(zip(self.population, self.fitness_scores)), k
            )
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two parent architectures."""
        child = {}
        for param in self.search_space.keys():
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an architecture."""
        mutated = architecture.copy()
        for param, choices in self.search_space.items():
            if random.random() < self.mutation_rate:
                mutated[param] = random.choice(choices)
        return mutated
    
    def evolve(self) -> Dict[str, Any]:
        """Run evolutionary search."""
        self.initialize_population()
        
        for generation in range(self.num_generations):
            # Evaluate fitness
            self.fitness_scores = [
                self.evaluate_fitness(arch) for arch in self.population
            ]
            
            # Print best fitness
            best_fitness = max(self.fitness_scores)
            print(f"Generation {generation}, Best Fitness: {best_fitness:.4f}")
            
            # Create new population
            new_population = []
            
            # Keep best individuals (elitism)
            elite_size = self.population_size // 10
            elite_indices = np.argsort(self.fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parents = self.select_parents()
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
        
        # Return best architecture
        final_fitness = [self.evaluate_fitness(arch) for arch in self.population]
        best_idx = np.argmax(final_fitness)
        return self.population[best_idx]

class NASFramework:
    """Unified Neural Architecture Search framework."""
    
    def __init__(self, method: str = 'darts', **kwargs):
        self.method = method
        self.kwargs = kwargs
    
    def search(self, 
               train_loader: Any,
               val_loader: Any,
               num_epochs: int = 50) -> Dict[str, Any]:
        """Run architecture search."""
        
        if self.method == 'darts':
            return self._darts_search(train_loader, val_loader, num_epochs)
        elif self.method == 'evolutionary':
            return self._evolutionary_search()
        else:
            raise ValueError(f"Unknown NAS method: {self.method}")
    
    def _darts_search(self, train_loader: Any, val_loader: Any, num_epochs: int) -> Dict[str, Any]:
        """Run DARTS search."""
        # Initialize supernet
        model = DARTSNetwork(
            C_in=self.kwargs.get('input_channels', 3),
            num_classes=self.kwargs.get('num_classes', 10),
            num_layers=self.kwargs.get('num_layers', 8)
        )
        
        # Training loop (simplified)
        for epoch in range(num_epochs):
            # In practice, you would alternate between training weights and architecture
            print(f"DARTS Epoch {epoch}/{num_epochs}")
            # Optimizers would be used here in actual training
            
        # Extract final architecture
        return model.get_architecture()
    
    def _evolutionary_search(self) -> Dict[str, Any]:
        """Run evolutionary search."""
        search_space = {
            'num_layers': [6, 8, 12, 16, 20],
            'channels': [16, 32, 64, 128],
            'operations': [
                ['conv3x3', 'conv1x1', 'maxpool3x3'],
                ['conv3x3', 'sepconv3x3', 'skip'],
                ['conv1x1', 'depthconv3x3', 'avgpool3x3']
            ]
        }
        
        evo_nas = EvolutionaryNAS(
            search_space=search_space,
            population_size=self.kwargs.get('population_size', 50),
            num_generations=self.kwargs.get('num_generations', 100)
        )
        
        return evo_nas.evolve()

# Example usage
if __name__ == "__main__":
    print("Testing Neural Architecture Search...")
    
    # Test DARTS
    print("\n1. Testing DARTS Network:")
    darts_model = DARTSNetwork(C_in=3, num_classes=10, num_layers=6)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = darts_model(dummy_input)
    print(f"DARTS output shape: {output.shape}")
    
    architecture = darts_model.get_architecture()
    print(f"Discovered architecture: {list(architecture.keys())}")
    
    # Test Evolutionary NAS
    print("\n2. Testing Evolutionary NAS:")
    nas_framework = NASFramework(
        method='evolutionary',
        population_size=20,
        num_generations=10
    )
    
    best_arch = nas_framework.search(None, None)
    print(f"Best evolutionary architecture: {best_arch}")
    
    print("\nNeural Architecture Search 2.0 implemented successfully! 🚀")
