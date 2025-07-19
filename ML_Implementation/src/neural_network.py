"""
Advanced Neural Network Implementation from Scratch
==================================================

A comprehensive neural network implementation with multiple activation functions,
loss functions, optimizers, and regularization techniques.

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Callable, Dict, Any
from abc import ABC, abstractmethod
import pickle


class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of activation function."""
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        pass


class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class ReLU(ActivationFunction):
    """ReLU activation function."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class LeakyReLU(ActivationFunction):
    """Leaky ReLU activation function."""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Softmax(ActivationFunction):
    """Softmax activation function for multi-class classification."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerical stability: subtract max value
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        # For softmax, the derivative is computed in the loss function
        return np.ones_like(x)


class LossFunction(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss."""
        pass
    
    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient of loss with respect to predictions."""
        pass


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss for regression."""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / len(y_true)


class CrossEntropyLoss(LossFunction):
    """Cross-entropy loss for classification."""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        if y_true.ndim == 1:
            # Binary classification
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Multi-class classification
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Clip to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        if y_true.ndim == 1:
            # Binary classification
            return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)
        else:
            # Multi-class classification
            return (y_pred - y_true) / len(y_true)


class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    @abstractmethod
    def update(self, layer: 'Dense', grad_w: np.ndarray, grad_b: np.ndarray) -> None:
        """Update layer parameters."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, layer: 'Dense', grad_w: np.ndarray, grad_b: np.ndarray) -> None:
        layer_id = id(layer)
        
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'v_w': np.zeros_like(grad_w),
                'v_b': np.zeros_like(grad_b)
            }
        
        # Update velocities
        self.velocities[layer_id]['v_w'] = (self.momentum * self.velocities[layer_id]['v_w'] + 
                                           self.learning_rate * grad_w)
        self.velocities[layer_id]['v_b'] = (self.momentum * self.velocities[layer_id]['v_b'] + 
                                           self.learning_rate * grad_b)
        
        # Update parameters
        layer.weights -= self.velocities[layer_id]['v_w']
        layer.bias -= self.velocities[layer_id]['v_b']


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = {}
        self.t = 0  # Time step
    
    def update(self, layer: 'Dense', grad_w: np.ndarray, grad_b: np.ndarray) -> None:
        self.t += 1
        layer_id = id(layer)
        
        if layer_id not in self.moments:
            self.moments[layer_id] = {
                'm_w': np.zeros_like(grad_w),
                'v_w': np.zeros_like(grad_w),
                'm_b': np.zeros_like(grad_b),
                'v_b': np.zeros_like(grad_b)
            }
        
        # Update biased first moment estimate
        self.moments[layer_id]['m_w'] = (self.beta1 * self.moments[layer_id]['m_w'] + 
                                        (1 - self.beta1) * grad_w)
        self.moments[layer_id]['m_b'] = (self.beta1 * self.moments[layer_id]['m_b'] + 
                                        (1 - self.beta1) * grad_b)
        
        # Update biased second raw moment estimate
        self.moments[layer_id]['v_w'] = (self.beta2 * self.moments[layer_id]['v_w'] + 
                                        (1 - self.beta2) * grad_w ** 2)
        self.moments[layer_id]['v_b'] = (self.beta2 * self.moments[layer_id]['v_b'] + 
                                        (1 - self.beta2) * grad_b ** 2)
        
        # Compute bias-corrected first and second moment estimates
        m_w_corrected = self.moments[layer_id]['m_w'] / (1 - self.beta1 ** self.t)
        m_b_corrected = self.moments[layer_id]['m_b'] / (1 - self.beta1 ** self.t)
        v_w_corrected = self.moments[layer_id]['v_w'] / (1 - self.beta2 ** self.t)
        v_b_corrected = self.moments[layer_id]['v_b'] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        layer.weights -= (self.learning_rate * m_w_corrected / 
                         (np.sqrt(v_w_corrected) + self.epsilon))
        layer.bias -= (self.learning_rate * m_b_corrected / 
                      (np.sqrt(v_b_corrected) + self.epsilon))


class Dense:
    """Dense (fully connected) layer."""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: ActivationFunction, 
                 weight_init: str = 'xavier'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights
        if weight_init == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        elif weight_init == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:  # random
            self.weights = np.random.randn(input_size, output_size) * 0.01
        
        self.bias = np.zeros((1, output_size))
        
        # Cache for backpropagation
        self.last_input = None
        self.last_z = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        self.last_input = x
        self.last_z = x.dot(self.weights) + self.bias
        return self.activation.forward(self.last_z)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        # Gradient of activation function
        grad_activation = self.activation.backward(self.last_z)
        
        # Gradient with respect to pre-activation
        grad_z = grad_output * grad_activation
        
        # Gradients with respect to parameters
        grad_weights = self.last_input.T.dot(grad_z)
        grad_bias = np.sum(grad_z, axis=0, keepdims=True)
        
        # Gradient with respect to input
        grad_input = grad_z.dot(self.weights.T)
        
        return grad_input, grad_weights, grad_bias


class NeuralNetwork:
    """Multi-layer neural network."""
    
    def __init__(self, layers: List[Tuple[int, ActivationFunction]], 
                 loss_function: LossFunction, optimizer: Optimizer,
                 l1_reg: float = 0.0, l2_reg: float = 0.0):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        layers : list of tuples
            Each tuple contains (layer_size, activation_function)
        loss_function : LossFunction
            Loss function to use
        optimizer : Optimizer
            Optimizer for training
        l1_reg : float, default=0.0
            L1 regularization strength
        l2_reg : float, default=0.0
            L2 regularization strength
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # Build layers
        self.layers = []
        for i in range(len(layers) - 1):
            input_size = layers[i][0]
            output_size = layers[i + 1][0]
            activation = layers[i + 1][1]
            
            layer = Dense(input_size, output_size, activation)
            self.layers.append(layer)
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x: np.ndarray, y_true: np.ndarray) -> None:
        """Backward pass through the network."""
        # Forward pass to compute predictions
        y_pred = self.forward(x)
        
        # Compute loss gradient
        grad_output = self.loss_function.backward(y_true, y_pred)
        
        # Backward pass through layers
        for layer in reversed(self.layers):
            grad_input, grad_weights, grad_bias = layer.backward(grad_output)
            
            # Add regularization gradients
            if self.l1_reg > 0:
                grad_weights += self.l1_reg * np.sign(layer.weights)
            if self.l2_reg > 0:
                grad_weights += self.l2_reg * layer.weights
            
            # Update parameters
            self.optimizer.update(layer, grad_weights, grad_bias)
            
            grad_output = grad_input
    
    def compute_loss(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """Compute total loss including regularization."""
        y_pred = self.forward(x)
        loss = self.loss_function.forward(y_true, y_pred)
        
        # Add regularization terms
        reg_loss = 0
        for layer in self.layers:
            if self.l1_reg > 0:
                reg_loss += self.l1_reg * np.sum(np.abs(layer.weights))
            if self.l2_reg > 0:
                reg_loss += self.l2_reg * np.sum(layer.weights ** 2)
        
        return loss + reg_loss
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 100, batch_size: int = 32,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              verbose: bool = True) -> None:
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples, n_classes)
            Training targets
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Size of mini-batches
        validation_data : tuple, optional
            Validation data (X_val, y_val)
        verbose : bool, default=True
            Whether to print training progress
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Backward pass (includes forward pass)
                self.backward(X_batch, y_batch)
                
                # Compute batch loss
                batch_loss = self.compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss
            
            # Average loss for epoch
            avg_loss = epoch_loss / (n_samples // batch_size + 1)
            self.loss_history.append(avg_loss)
            
            # Compute accuracy
            train_accuracy = self.accuracy(X, y)
            self.accuracy_history.append(train_accuracy)
            
            # Validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.compute_loss(X_val, y_val)
                val_accuracy = self.accuracy(X_val, y_val)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {train_accuracy:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {train_accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.forward(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predictions = self.predict(X)
        if predictions.shape[1] == 1:
            # Binary classification
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(predictions, axis=1)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        y_pred = self.predict_classes(X)
        if y.ndim == 2:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y
        return np.mean(y_pred == y_true)
    
    def plot_training_history(self) -> None:
        """Plot training loss and accuracy history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'layers': [(layer.input_size, layer.output_size, 
                       layer.weights, layer.bias, layer.activation) 
                      for layer in self.layers],
            'loss_function': self.loss_function,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct layers
        self.layers = []
        for layer_data in model_data['layers']:
            input_size, output_size, weights, bias, activation = layer_data
            layer = Dense(input_size, output_size, activation)
            layer.weights = weights
            layer.bias = bias
            self.layers.append(layer)
        
        # Restore other attributes
        self.loss_function = model_data['loss_function']
        self.l1_reg = model_data['l1_reg']
        self.l2_reg = model_data['l2_reg']
        self.loss_history = model_data['loss_history']
        self.accuracy_history = model_data['accuracy_history']


# Utility functions
def to_categorical(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    categorical = np.zeros((len(y), num_classes))
    categorical[np.arange(len(y)), y] = 1
    return categorical


def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """Split data into training and testing sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# Example usage and demonstration
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_moons
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("Advanced Neural Network Demo")
    print("="*60)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_classes=3, 
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert labels to one-hot
    y_categorical = to_categorical(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Number of classes: {y_categorical.shape[1]}")
    
    # Define network architecture
    layers = [
        (20, None),  # Input layer (20 features)
        (64, ReLU()),  # Hidden layer 1
        (32, ReLU()),  # Hidden layer 2
        (16, ReLU()),  # Hidden layer 3
        (3, Softmax())  # Output layer (3 classes)
    ]
    
    # Create network with different optimizers
    print("\n1. Training with Adam optimizer")
    print("-" * 40)
    
    network_adam = NeuralNetwork(
        layers=layers,
        loss_function=CrossEntropyLoss(),
        optimizer=Adam(learning_rate=0.001),
        l2_reg=0.001
    )
    
    network_adam.train(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Evaluate
    train_accuracy = network_adam.accuracy(X_train, y_train)
    test_accuracy = network_adam.accuracy(X_test, y_test)
    
    print(f"\nFinal Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # Test different activation functions
    print("\n2. Training with different activation functions")
    print("-" * 40)
    
    activations = [
        ("ReLU", ReLU()),
        ("Sigmoid", Sigmoid()),
        ("Tanh", Tanh()),
        ("Leaky ReLU", LeakyReLU(alpha=0.01))
    ]
    
    for name, activation in activations:
        layers_test = [
            (20, None),
            (32, activation),
            (16, activation),
            (3, Softmax())
        ]
        
        network = NeuralNetwork(
            layers=layers_test,
            loss_function=CrossEntropyLoss(),
            optimizer=Adam(learning_rate=0.001)
        )
        
        network.train(X_train, y_train, epochs=50, verbose=False)
        accuracy = network.accuracy(X_test, y_test)
        
        print(f"{name:12} - Test Accuracy: {accuracy:.4f}")
    
    # Binary classification example
    print("\n3. Binary Classification Example")
    print("-" * 40)
    
    X_binary, y_binary = make_moons(n_samples=500, noise=0.1, random_state=42)
    X_binary = StandardScaler().fit_transform(X_binary)
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_binary, y_binary, test_size=0.2, random_state=42
    )
    
    # Reshape for binary classification
    y_train_bin = y_train_bin.reshape(-1, 1)
    y_test_bin = y_test_bin.reshape(-1, 1)
    
    binary_layers = [
        (2, None),
        (16, ReLU()),
        (8, ReLU()),
        (1, Sigmoid())
    ]
    
    binary_network = NeuralNetwork(
        layers=binary_layers,
        loss_function=CrossEntropyLoss(),
        optimizer=SGD(learning_rate=0.1, momentum=0.9)
    )
    
    binary_network.train(
        X_train_bin, y_train_bin,
        epochs=100,
        batch_size=16,
        validation_data=(X_test_bin, y_test_bin),
        verbose=False
    )
    
    binary_accuracy = binary_network.accuracy(X_test_bin, y_test_bin)
    print(f"Binary Classification Test Accuracy: {binary_accuracy:.4f}")
    
    print("\nDemo completed successfully!")
