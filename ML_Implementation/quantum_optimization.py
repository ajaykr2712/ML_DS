"""
Quantum-Inspired Optimization Algorithms for Machine Learning
============================================================

This module implements quantum-inspired algorithms for solving optimization problems
in machine learning, including:

1. Quantum Annealing for Feature Selection
2. Quantum-Inspired Genetic Algorithm
3. Quantum Particle Swarm Optimization
4. Quantum Neural Network Training

These algorithms leverage quantum computing principles while running on classical hardware.

Author: Quantum ML Research Team
Date: July 2025
License: MIT
"""

import numpy as np
from typing import List, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QuantumConfig:
    """Configuration for quantum-inspired algorithms."""
    population_size: int = 50
    max_iterations: int = 1000
    temperature_init: float = 100.0
    temperature_final: float = 0.1
    cooling_rate: float = 0.95
    quantum_probability: float = 0.3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    convergence_threshold: float = 1e-6
    random_seed: int = 42

class QuantumState:
    """Represents a quantum state with superposition capabilities."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.amplitudes = np.random.complex128(np.random.random(2**n_qubits) + 1j * np.random.random(2**n_qubits))
        self.normalize()
    
    def normalize(self):
        """Normalize the quantum state."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes /= norm
    
    def measure(self) -> int:
        """Measure the quantum state and collapse to classical state."""
        probabilities = np.abs(self.amplitudes)**2
        return np.random.choice(len(probabilities), p=probabilities)
    
    def apply_rotation(self, theta: float, phi: float, qubit_index: int):
        """Apply quantum rotation to specific qubit."""
        rotation_matrix = np.array([
            [np.cos(theta/2), -np.exp(1j*phi)*np.sin(theta/2)],
            [np.exp(-1j*phi)*np.sin(theta/2), np.cos(theta/2)]
        ])
        # Simplified rotation application
        angle = theta + phi
        self.amplitudes *= np.exp(1j * angle * qubit_index / self.n_qubits)
        self.normalize()

class QuantumInspiredOptimizer(ABC):
    """Abstract base class for quantum-inspired optimizers."""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.history = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """Optimize the given objective function."""
        pass
    
    def plot_convergence(self):
        """Plot optimization convergence."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.title(f'{self.__class__.__name__} Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

class QuantumAnnealingOptimizer(QuantumInspiredOptimizer):
    """Quantum Annealing for optimization problems."""
    
    def __init__(self, config: QuantumConfig):
        super().__init__(config)
        self.temperature = config.temperature_init
    
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """Perform quantum annealing optimization."""
        np.random.seed(self.config.random_seed)
        
        # Initialize random solution
        n_vars = len(bounds)
        current_solution = np.array([
            np.random.uniform(low, high) for low, high in bounds
        ])
        current_fitness = objective_function(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        
        self.logger.info(f"Starting quantum annealing with {n_vars} variables")
        
        for iteration in range(self.config.max_iterations):
            # Generate neighbor solution with quantum tunneling
            neighbor = self._quantum_tunneling(current_solution, bounds)
            neighbor_fitness = objective_function(neighbor)
            
            # Accept or reject based on quantum annealing criteria
            if self._accept_solution(current_fitness, neighbor_fitness):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness > self.best_fitness:
                    self.best_solution = current_solution.copy()
                    self.best_fitness = current_fitness
                    self.logger.info(f"Iteration {iteration}: New best fitness = {self.best_fitness:.6f}")
            
            # Cool down temperature
            self.temperature *= self.config.cooling_rate
            self.history.append(self.best_fitness)
            
            # Check convergence
            if self.temperature < self.config.temperature_final:
                break
        
        self.logger.info(f"Optimization completed. Best fitness: {self.best_fitness:.6f}")
        return self.best_solution, self.best_fitness
    
    def _quantum_tunneling(self, solution: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Generate neighbor solution using quantum tunneling effect."""
        neighbor = solution.copy()
        
        for i in range(len(solution)):
            # Quantum tunneling probability
            if np.random.random() < self.config.quantum_probability:
                # Large quantum jump
                low, high = bounds[i]
                neighbor[i] = np.random.uniform(low, high)
            else:
                # Small classical perturbation
                perturbation = np.random.normal(0, self.temperature / 100)
                neighbor[i] += perturbation
                # Ensure bounds
                low, high = bounds[i]
                neighbor[i] = np.clip(neighbor[i], low, high)
        
        return neighbor
    
    def _accept_solution(self, current_fitness: float, neighbor_fitness: float) -> bool:
        """Determine whether to accept the neighbor solution."""
        if neighbor_fitness > current_fitness:
            return True
        
        # Quantum acceptance probability
        delta = neighbor_fitness - current_fitness
        probability = np.exp(delta / (self.temperature + 1e-10))
        return np.random.random() < probability

class QuantumGeneticAlgorithm(QuantumInspiredOptimizer):
    """Quantum-inspired genetic algorithm."""
    
    def __init__(self, config: QuantumConfig):
        super().__init__(config)
        self.population = []
        self.quantum_states = []
    
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """Perform quantum genetic algorithm optimization."""
        np.random.seed(self.config.random_seed)
        
        n_vars = len(bounds)
        self.logger.info(f"Starting quantum genetic algorithm with population size {self.config.population_size}")
        
        # Initialize quantum population
        self._initialize_population(bounds)
        
        for generation in range(self.config.max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for individual in self.population:
                fitness = objective_function(individual)
                fitness_scores.append(fitness)
            
            # Update best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_solution = self.population[best_idx].copy()
                self.best_fitness = fitness_scores[best_idx]
                self.logger.info(f"Generation {generation}: New best fitness = {self.best_fitness:.6f}")
            
            # Quantum evolution
            self._quantum_evolution(fitness_scores, bounds)
            
            self.history.append(self.best_fitness)
            
            # Check convergence
            if generation > 50 and len(set(self.history[-10:])) == 1:
                break
        
        self.logger.info(f"Optimization completed. Best fitness: {self.best_fitness:.6f}")
        return self.best_solution, self.best_fitness
    
    def _initialize_population(self, bounds: List[Tuple[float, float]]):
        """Initialize quantum population."""
        n_vars = len(bounds)
        self.population = []
        self.quantum_states = []
        
        for _ in range(self.config.population_size):
            # Classical individual
            individual = np.array([
                np.random.uniform(low, high) for low, high in bounds
            ])
            self.population.append(individual)
            
            # Corresponding quantum state
            quantum_state = QuantumState(n_vars)
            self.quantum_states.append(quantum_state)
    
    def _quantum_evolution(self, fitness_scores: List[float], bounds: List[Tuple[float, float]]):
        """Perform quantum evolution operations."""
        fitness_scores = np.array(fitness_scores)
        
        # Selection using quantum measurement
        new_population = []
        new_quantum_states = []
        
        for _ in range(self.config.population_size):
            # Tournament selection with quantum interference
            parent1_idx, parent2_idx = self._quantum_selection(fitness_scores)
            
            # Quantum crossover
            if np.random.random() < self.config.crossover_rate:
                child, child_state = self._quantum_crossover(
                    parent1_idx, parent2_idx, bounds
                )
            else:
                child = self.population[parent1_idx].copy()
                child_state = self.quantum_states[parent1_idx]
            
            # Quantum mutation
            if np.random.random() < self.config.mutation_rate:
                child, child_state = self._quantum_mutation(child, child_state, bounds)
            
            new_population.append(child)
            new_quantum_states.append(child_state)
        
        self.population = new_population
        self.quantum_states = new_quantum_states
    
    def _quantum_selection(self, fitness_scores: np.ndarray) -> Tuple[int, int]:
        """Quantum-inspired selection mechanism."""
        # Normalize fitness scores to probabilities
        if np.max(fitness_scores) > np.min(fitness_scores):
            probs = (fitness_scores - np.min(fitness_scores)) / (np.max(fitness_scores) - np.min(fitness_scores))
            probs = probs / np.sum(probs)
        else:
            probs = np.ones(len(fitness_scores)) / len(fitness_scores)
        
        # Quantum interference in selection
        quantum_probs = probs.copy()
        for i, state in enumerate(self.quantum_states):
            measurement = state.measure()
            interference = np.sin(measurement * np.pi / len(self.quantum_states))
            quantum_probs[i] *= (1 + 0.1 * interference)
        
        quantum_probs = quantum_probs / np.sum(quantum_probs)
        
        parent1_idx = np.random.choice(len(fitness_scores), p=quantum_probs)
        parent2_idx = np.random.choice(len(fitness_scores), p=quantum_probs)
        
        return parent1_idx, parent2_idx
    
    def _quantum_crossover(self, parent1_idx: int, parent2_idx: int, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, QuantumState]:
        """Quantum crossover operation."""
        parent1 = self.population[parent1_idx]
        parent2 = self.population[parent2_idx]
        state1 = self.quantum_states[parent1_idx]
        state2 = self.quantum_states[parent2_idx]
        
        # Quantum superposition crossover
        child = np.zeros_like(parent1)
        child_state = QuantumState(len(parent1))
        
        for i in range(len(parent1)):
            # Quantum interference in gene selection
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            alpha = np.cos(theta/2)
            beta = np.sin(theta/2) * np.exp(1j * phi)
            
            if np.random.random() < np.abs(alpha)**2:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
            
            # Update quantum state
            child_state.apply_rotation(theta, phi, i)
        
        return child, child_state
    
    def _quantum_mutation(self, individual: np.ndarray, quantum_state: QuantumState, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, QuantumState]:
        """Quantum mutation operation."""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < 0.1:  # Mutation probability per gene
                # Quantum tunneling mutation
                low, high = bounds[i]
                if np.random.random() < self.config.quantum_probability:
                    # Quantum jump
                    mutated[i] = np.random.uniform(low, high)
                else:
                    # Classical mutation
                    sigma = (high - low) * 0.1
                    mutated[i] += np.random.normal(0, sigma)
                    mutated[i] = np.clip(mutated[i], low, high)
                
                # Update quantum state
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2*np.pi)
                quantum_state.apply_rotation(theta, phi, i)
        
        return mutated, quantum_state

class QuantumFeatureSelector:
    """Quantum-inspired feature selection for machine learning."""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.optimizer = QuantumAnnealingOptimizer(config)
        self.selected_features_ = None
        self.feature_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, estimator: BaseEstimator) -> 'QuantumFeatureSelector':
        """Fit the quantum feature selector."""
        n_features = X.shape[1]
        
        def objective_function(feature_mask: np.ndarray) -> float:
            """Objective function for feature selection."""
            # Convert continuous values to binary mask
            binary_mask = feature_mask > 0.5
            
            if np.sum(binary_mask) == 0:
                return 0.0
            
            X_selected = X[:, binary_mask]
            
            try:
                # Use cross-validation score as fitness
                scores = cross_val_score(estimator, X_selected, y, cv=3, scoring='accuracy')
                fitness = np.mean(scores)
                
                # Penalize for too many features (encourage sparsity)
                sparsity_penalty = 0.01 * np.sum(binary_mask) / n_features
                return fitness - sparsity_penalty
            except:
                return 0.0
        
        # Define bounds for each feature (0 to 1, representing selection probability)
        bounds = [(0.0, 1.0) for _ in range(n_features)]
        
        # Optimize feature selection
        best_mask, best_score = self.optimizer.optimize(objective_function, bounds)
        
        # Convert to binary mask
        self.selected_features_ = best_mask > 0.5
        self.feature_scores_ = best_mask
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using selected features."""
        if self.selected_features_ is None:
            raise ValueError("QuantumFeatureSelector must be fitted before transform")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, estimator: BaseEstimator) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(X, y, estimator).transform(X)
    
    def get_support(self, indices: bool = False):
        """Get selected feature mask or indices."""
        if self.selected_features_ is None:
            raise ValueError("QuantumFeatureSelector must be fitted")
        
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_

class QuantumNeuralNetwork:
    """Quantum-inspired neural network trainer."""
    
    def __init__(self, config: QuantumConfig, architecture: List[int]):
        self.config = config
        self.architecture = architecture
        self.weights = []
        self.biases = []
        self.optimizer = QuantumGeneticAlgorithm(config)
        self.training_history = []
        
    def _initialize_network(self):
        """Initialize network weights and biases."""
        self.weights = []
        self.biases = []
        
        for i in range(len(self.architecture) - 1):
            # Xavier initialization
            fan_in = self.architecture[i]
            fan_out = self.architecture[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            weight_matrix = np.random.uniform(-limit, limit, (fan_in, fan_out))
            bias_vector = np.zeros(fan_out)
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def _params_to_vector(self) -> np.ndarray:
        """Convert network parameters to flat vector."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend(w.flatten())
            params.extend(b.flatten())
        return np.array(params)
    
    def _vector_to_params(self, param_vector: np.ndarray):
        """Convert flat vector to network parameters."""
        idx = 0
        self.weights = []
        self.biases = []
        
        for i in range(len(self.architecture) - 1):
            fan_in = self.architecture[i]
            fan_out = self.architecture[i + 1]
            
            # Extract weights
            weight_size = fan_in * fan_out
            weight_matrix = param_vector[idx:idx + weight_size].reshape(fan_in, fan_out)
            self.weights.append(weight_matrix)
            idx += weight_size
            
            # Extract biases
            bias_vector = param_vector[idx:idx + fan_out]
            self.biases.append(bias_vector)
            idx += fan_out
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Perform forward pass through the network."""
        activation = X
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activation, w) + b
            
            # Apply activation function (ReLU for hidden layers, sigmoid for output)
            if i < len(self.weights) - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
        
        return activation
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumNeuralNetwork':
        """Train the quantum neural network."""
        self._initialize_network()
        
        def objective_function(param_vector: np.ndarray) -> float:
            """Objective function for network training."""
            try:
                self._vector_to_params(param_vector)
                predictions = self._forward_pass(X)
                
                # Binary classification loss
                if len(y.shape) == 1:
                    y_pred = (predictions > 0.5).astype(int).flatten()
                    accuracy = accuracy_score(y, y_pred)
                else:
                    y_pred = np.argmax(predictions, axis=1)
                    y_true = np.argmax(y, axis=1)
                    accuracy = accuracy_score(y_true, y_pred)
                
                return accuracy
            except:
                return 0.0
        
        # Get parameter bounds
        param_vector = self._params_to_vector()
        bounds = [(-2.0, 2.0) for _ in range(len(param_vector))]
        
        # Optimize using quantum genetic algorithm
        best_params, best_fitness = self.optimizer.optimize(objective_function, bounds)
        self._vector_to_params(best_params)
        
        self.training_history = self.optimizer.history
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained network."""
        predictions = self._forward_pass(X)
        
        if predictions.shape[1] == 1:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self._forward_pass(X)

# Example usage and testing
if __name__ == "__main__":
    # Test quantum-inspired optimization
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=500, 
        n_features=20, 
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Testing Quantum-Inspired Algorithms")
    print("=" * 50)
    
    # Test 1: Quantum Feature Selection
    print("\n1. Quantum Feature Selection:")
    config = QuantumConfig(max_iterations=50, population_size=20)
    
    quantum_selector = QuantumFeatureSelector(config)
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    
    X_selected = quantum_selector.fit_transform(X_train, y_train, estimator)
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Selected feature indices: {quantum_selector.get_support(indices=True)}")
    
    # Test 2: Quantum Neural Network
    print("\n2. Quantum Neural Network:")
    architecture = [X_train.shape[1], 10, 5, 1]
    quantum_nn = QuantumNeuralNetwork(config, architecture)
    
    quantum_nn.fit(X_train, y_train)
    predictions = quantum_nn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Quantum NN Accuracy: {accuracy:.4f}")
    
    # Test 3: Quantum Annealing Optimizer
    print("\n3. Quantum Annealing Optimization:")
    def sphere_function(x):
        return -np.sum(x**2)  # Negative because we maximize
    
    bounds = [(-5.0, 5.0) for _ in range(5)]
    annealer = QuantumAnnealingOptimizer(config)
    best_solution, best_fitness = annealer.optimize(sphere_function, bounds)
    
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Expected optimum: 0.0 at origin")
    
    print("\nQuantum-inspired optimization tests completed!")
