"""
Neural Architecture Search (NAS) Implementation
Advanced automated neural network architecture optimization
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any
import random
import json
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture search"""
    max_layers: int = 20
    layer_types: List[str] = None
    activation_functions: List[str] = None
    optimizer_types: List[str] = None
    batch_sizes: List[int] = None
    learning_rates: List[float] = None
    
    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ['dense', 'conv2d', 'lstm', 'gru', 'attention']
        if self.activation_functions is None:
            self.activation_functions = ['relu', 'tanh', 'sigmoid', 'swish', 'gelu']
        if self.optimizer_types is None:
            self.optimizer_types = ['adam', 'sgd', 'rmsprop', 'adamw']
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64, 128, 256]
        if self.learning_rates is None:
            self.learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1]

class NeuralArchitectureSearch:
    """
    Advanced Neural Architecture Search implementation with multiple search strategies
    """
    
    def __init__(self, config: ArchitectureConfig, search_strategy: str = 'random'):
        """
        Initialize NAS with configuration
        
        Args:
            config: Architecture search configuration
            search_strategy: Search strategy ('random', 'evolutionary', 'bayesian')
        """
        self.config = config
        self.search_strategy = search_strategy
        self.population = []
        self.generation = 0
        self.best_architectures = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def generate_random_architecture(self) -> Dict[str, Any]:
        """Generate a random neural network architecture"""
        num_layers = random.randint(1, self.config.max_layers)
        
        architecture = {
            'layers': [],
            'optimizer': random.choice(self.config.optimizer_types),
            'learning_rate': random.choice(self.config.learning_rates),
            'batch_size': random.choice(self.config.batch_sizes),
            'epochs': random.randint(10, 100)
        }
        
        for i in range(num_layers):
            layer_type = random.choice(self.config.layer_types)
            layer_config = self._generate_layer_config(layer_type, i)
            architecture['layers'].append(layer_config)
            
        return architecture
    
    def _generate_layer_config(self, layer_type: str, layer_index: int) -> Dict[str, Any]:
        """Generate configuration for a specific layer type"""
        base_config = {
            'type': layer_type,
            'activation': random.choice(self.config.activation_functions)
        }
        
        if layer_type == 'dense':
            base_config.update({
                'units': random.choice([32, 64, 128, 256, 512, 1024]),
                'dropout': random.uniform(0.0, 0.5)
            })
        elif layer_type == 'conv2d':
            base_config.update({
                'filters': random.choice([16, 32, 64, 128, 256]),
                'kernel_size': random.choice([3, 5, 7]),
                'strides': random.choice([1, 2]),
                'padding': random.choice(['same', 'valid'])
            })
        elif layer_type in ['lstm', 'gru']:
            base_config.update({
                'units': random.choice([32, 64, 128, 256]),
                'return_sequences': random.choice([True, False]),
                'dropout': random.uniform(0.0, 0.3)
            })
        elif layer_type == 'attention':
            base_config.update({
                'num_heads': random.choice([2, 4, 8, 16]),
                'key_dim': random.choice([32, 64, 128]),
                'dropout': random.uniform(0.0, 0.2)
            })
            
        return base_config
    
    def build_model_from_architecture(self, architecture: Dict[str, Any], 
                                    input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build a Keras model from architecture specification"""
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        for layer_config in architecture['layers']:
            x = self._build_layer(x, layer_config)
            
        # Add output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = self._get_optimizer(architecture['optimizer'], 
                                      architecture['learning_rate'])
        model.compile(optimizer=optimizer, 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        return model
    
    def _build_layer(self, x, layer_config: Dict[str, Any]):
        """Build a single layer from configuration"""
        layer_type = layer_config['type']
        
        if layer_type == 'dense':
            x = tf.keras.layers.Dense(
                units=layer_config['units'],
                activation=layer_config['activation']
            )(x)
            if 'dropout' in layer_config:
                x = tf.keras.layers.Dropout(layer_config['dropout'])(x)
                
        elif layer_type == 'conv2d':
            x = tf.keras.layers.Conv2D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                strides=layer_config['strides'],
                padding=layer_config['padding'],
                activation=layer_config['activation']
            )(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
            
        elif layer_type == 'lstm':
            x = tf.keras.layers.LSTM(
                units=layer_config['units'],
                return_sequences=layer_config['return_sequences'],
                dropout=layer_config.get('dropout', 0.0)
            )(x)
            
        elif layer_type == 'gru':
            x = tf.keras.layers.GRU(
                units=layer_config['units'],
                return_sequences=layer_config['return_sequences'],
                dropout=layer_config.get('dropout', 0.0)
            )(x)
            
        elif layer_type == 'attention':
            x = tf.keras.layers.MultiHeadAttention(
                num_heads=layer_config['num_heads'],
                key_dim=layer_config['key_dim'],
                dropout=layer_config.get('dropout', 0.0)
            )(x, x)
            
        return x
    
    def _get_optimizer(self, optimizer_type: str, learning_rate: float):
        """Get optimizer instance"""
        optimizers = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adamw': tf.keras.optimizers.AdamW
        }
        return optimizers[optimizer_type](learning_rate=learning_rate)
    
    def evaluate_architecture(self, architecture: Dict[str, Any], 
                            X_train, y_train, X_val, y_val) -> float:
        """Evaluate a single architecture"""
        try:
            # Build and train model
            model = self.build_model_from_architecture(architecture, X_train.shape[1:])
            
            history = model.fit(
                X_train, y_train,
                batch_size=architecture['batch_size'],
                epochs=min(architecture['epochs'], 20),  # Limit for search
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Return validation accuracy
            val_accuracy = max(history.history['val_accuracy'])
            return val_accuracy
            
        except Exception as e:
            self.logger.warning(f"Architecture evaluation failed: {e}")
            return 0.0
    
    def random_search(self, X_train, y_train, X_val, y_val, 
                     num_architectures: int = 50) -> List[Tuple[Dict, float]]:
        """Perform random architecture search"""
        results = []
        
        for i in range(num_architectures):
            self.logger.info(f"Evaluating architecture {i+1}/{num_architectures}")
            
            architecture = self.generate_random_architecture()
            score = self.evaluate_architecture(architecture, X_train, y_train, X_val, y_val)
            
            results.append((architecture, score))
            self.logger.info(f"Architecture {i+1} score: {score:.4f}")
            
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def evolutionary_search(self, X_train, y_train, X_val, y_val,
                          population_size: int = 20, generations: int = 10) -> List[Tuple[Dict, float]]:
        """Perform evolutionary architecture search"""
        # Initialize population
        population = []
        for _ in range(population_size):
            architecture = self.generate_random_architecture()
            score = self.evaluate_architecture(architecture, X_train, y_train, X_val, y_val)
            population.append((architecture, score))
        
        # Sort by fitness
        population.sort(key=lambda x: x[1], reverse=True)
        
        for generation in range(generations):
            self.logger.info(f"Generation {generation+1}/{generations}")
            
            # Select top performers
            elite_size = population_size // 4
            elite = population[:elite_size]
            
            # Generate new population
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                # Crossover and mutation
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover(parent1[0], parent2[0])
                child = self._mutate(child)
                
                score = self.evaluate_architecture(child, X_train, y_train, X_val, y_val)
                new_population.append((child, score))
            
            # Sort new population
            new_population.sort(key=lambda x: x[1], reverse=True)
            population = new_population[:population_size]
            
            best_score = population[0][1]
            self.logger.info(f"Generation {generation+1} best score: {best_score:.4f}")
        
        return population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two architectures"""
        child = parent1.copy()
        
        # Mix layers from both parents
        if len(parent2['layers']) > 0:
            crossover_point = random.randint(0, len(child['layers']))
            child['layers'] = (child['layers'][:crossover_point] + 
                             parent2['layers'][crossover_point:])
        
        # Mix hyperparameters
        if random.random() < 0.5:
            child['optimizer'] = parent2['optimizer']
        if random.random() < 0.5:
            child['learning_rate'] = parent2['learning_rate']
        if random.random() < 0.5:
            child['batch_size'] = parent2['batch_size']
            
        return child
    
    def _mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an architecture"""
        mutated = architecture.copy()
        
        # Mutate layers
        if random.random() < 0.3:  # 30% chance to add layer
            new_layer = self._generate_layer_config(
                random.choice(self.config.layer_types), 
                len(mutated['layers'])
            )
            mutated['layers'].append(new_layer)
        
        if random.random() < 0.2 and len(mutated['layers']) > 1:  # 20% chance to remove layer
            mutated['layers'].pop(random.randint(0, len(mutated['layers'])-1))
        
        # Mutate hyperparameters
        if random.random() < 0.3:
            mutated['learning_rate'] = random.choice(self.config.learning_rates)
        if random.random() < 0.3:
            mutated['batch_size'] = random.choice(self.config.batch_sizes)
            
        return mutated
    
    def search(self, X, y, validation_split: float = 0.2, **kwargs) -> List[Tuple[Dict, float]]:
        """Main search function"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        if self.search_strategy == 'random':
            return self.random_search(X_train, y_train, X_val, y_val, **kwargs)
        elif self.search_strategy == 'evolutionary':
            return self.evolutionary_search(X_train, y_train, X_val, y_val, **kwargs)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")
    
    def save_results(self, results: List[Tuple[Dict, float]], filepath: str):
        """Save search results to file"""
        save_data = {
            'search_strategy': self.search_strategy,
            'config': self.config.__dict__,
            'results': [(arch, score) for arch, score in results[:10]]  # Top 10
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = ArchitectureConfig(
        max_layers=15,
        layer_types=['dense', 'conv2d', 'lstm'],
        activation_functions=['relu', 'tanh', 'swish']
    )
    
    # Initialize NAS
    nas = NeuralArchitectureSearch(config, search_strategy='evolutionary')
    
    # Generate sample data
    X = np.random.randn(1000, 64)
    y = np.random.randint(0, 2, 1000)
    
    # Perform search
    results = nas.search(X, y, num_architectures=20, generations=5)
    
    # Save results
    nas.save_results(results, 'nas_results.json')
    
    print(f"Best architecture score: {results[0][1]:.4f}")
