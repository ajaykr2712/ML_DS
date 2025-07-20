"""
Advanced Neural Network Implementation from Scratch
Multi-layer perceptron with various activation functions, optimizers, and regularization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
import pickle


class ActivationFunction(ABC):
    """Abstract base class for activation functions"""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass


class ReLU(ActivationFunction):
    """ReLU activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class LeakyReLU(ActivationFunction):
    """Leaky ReLU activation function"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class Tanh(ActivationFunction):
    """Tanh activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Softmax(ActivationFunction):
    """Softmax activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        # For softmax, we'll compute this in the loss function
        # This is a placeholder
        return np.ones_like(x)


class LossFunction(ABC):
    """Abstract base class for loss functions"""
    
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss function"""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y_true - y_pred) / len(y_true)


class BinaryCrossEntropy(LossFunction):
    """Binary Cross Entropy loss function"""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / len(y_true)


class CategoricalCrossEntropy(LossFunction):
    """Categorical Cross Entropy loss function"""
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred_clipped / len(y_true)


class Optimizer(ABC):
    """Abstract base class for optimizers"""
    
    @abstractmethod
    def update(self, weights: np.ndarray, bias: np.ndarray, 
               dw: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vw = None
        self.vb = None
    
    def update(self, weights: np.ndarray, bias: np.ndarray, 
               dw: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.vw is None:
            self.vw = np.zeros_like(weights)
            self.vb = np.zeros_like(bias)
        
        self.vw = self.momentum * self.vw + self.learning_rate * dw
        self.vb = self.momentum * self.vb + self.learning_rate * db
        
        return weights - self.vw, bias - self.vb


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mw = None
        self.mb = None
        self.vw = None
        self.vb = None
        self.t = 0
    
    def update(self, weights: np.ndarray, bias: np.ndarray, 
               dw: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mw is None:
            self.mw = np.zeros_like(weights)
            self.mb = np.zeros_like(bias)
            self.vw = np.zeros_like(weights)
            self.vb = np.zeros_like(bias)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.mw = self.beta1 * self.mw + (1 - self.beta1) * dw
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        
        # Update biased second moment estimate
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * (dw ** 2)
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (db ** 2)
        
        # Bias correction
        mw_corrected = self.mw / (1 - self.beta1 ** self.t)
        mb_corrected = self.mb / (1 - self.beta1 ** self.t)
        vw_corrected = self.vw / (1 - self.beta2 ** self.t)
        vb_corrected = self.vb / (1 - self.beta2 ** self.t)
        
        # Update parameters
        weights_new = weights - self.learning_rate * mw_corrected / (np.sqrt(vw_corrected) + self.epsilon)
        bias_new = bias - self.learning_rate * mb_corrected / (np.sqrt(vb_corrected) + self.epsilon)
        
        return weights_new, bias_new


class Layer:
    """Neural network layer"""
    
    def __init__(self, n_inputs: int, n_neurons: int, activation: ActivationFunction,
                 weight_init: str = 'xavier', dropout_rate: float = 0.0):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        if weight_init == 'xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        elif weight_init == 'he':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        else:  # random
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        
        self.biases = np.zeros(n_neurons)
        
        # For backpropagation
        self.last_input = None
        self.last_output = None
        self.last_z = None
        self.dropout_mask = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass"""
        self.last_input = inputs
        
        # Linear transformation
        self.last_z = np.dot(inputs, self.weights) + self.biases
        
        # Activation
        self.last_output = self.activation.forward(self.last_z)
        
        # Dropout (only during training)
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                                 size=self.last_output.shape) / (1 - self.dropout_rate)
            self.last_output *= self.dropout_mask
        
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass"""
        # Apply dropout mask to gradients
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            grad_output *= self.dropout_mask
        
        # Gradient w.r.t. activation
        grad_activation = grad_output * self.activation.backward(self.last_z)
        
        # Gradients w.r.t. weights and biases
        grad_weights = np.dot(self.last_input.T, grad_activation)
        grad_biases = np.sum(grad_activation, axis=0)
        
        # Gradient w.r.t. input (for previous layer)
        grad_input = np.dot(grad_activation, self.weights.T)
        
        return grad_input, grad_weights, grad_biases


class NeuralNetwork:
    """
    Multi-layer Neural Network Implementation
    
    Features:
    - Multiple hidden layers
    - Various activation functions
    - Different optimizers (SGD, Adam)
    - Regularization (L1, L2, Dropout)
    - Batch processing
    - Early stopping
    - Learning rate scheduling
    - Model saving/loading
    """
    
    def __init__(
        self,
        hidden_layers: List[int],
        activation: str = 'relu',
        output_activation: str = 'linear',
        loss: str = 'mse',
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        dropout_rate: float = 0.0,
        weight_init: str = 'xavier',
        batch_size: int = 32,
        early_stopping: bool = False,
        patience: int = 10,
        validation_split: float = 0.2,
        random_state: Optional[int] = None
    ):
        """
        Initialize Neural Network
        
        Parameters:
        -----------
        hidden_layers : List[int]
            Number of neurons in each hidden layer
        activation : str, default='relu'
            Activation function for hidden layers
        output_activation : str, default='linear'
            Activation function for output layer
        loss : str, default='mse'
            Loss function to use
        optimizer : str, default='adam'
            Optimizer to use
        learning_rate : float, default=0.001
            Learning rate
        l1_reg : float, default=0.0
            L1 regularization strength
        l2_reg : float, default=0.0
            L2 regularization strength
        dropout_rate : float, default=0.0
            Dropout rate for hidden layers
        weight_init : str, default='xavier'
            Weight initialization method
        batch_size : int, default=32
            Batch size for training
        early_stopping : bool, default=False
            Whether to use early stopping
        patience : int, default=10
            Patience for early stopping
        validation_split : float, default=0.2
            Fraction of data to use for validation
        random_state : int, optional
            Random seed
        """
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.loss_name = loss
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize components
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        
        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        # Model state
        self.is_fitted = False
        self.n_features = None
        self.n_outputs = None
        
        # Initialize activation functions
        self.activations = {
            'relu': ReLU(),
            'leaky_relu': LeakyReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'softmax': Softmax(),
            'linear': None
        }
        
        # Initialize loss functions
        self.loss_functions = {
            'mse': MeanSquaredError(),
            'binary_crossentropy': BinaryCrossEntropy(),
            'categorical_crossentropy': CategoricalCrossEntropy()
        }
    
    def _get_activation(self, name: str) -> Optional[ActivationFunction]:
        """Get activation function by name"""
        if name == 'linear':
            return None
        return self.activations[name]
    
    def _build_network(self, n_features: int, n_outputs: int):
        """Build the neural network architecture"""
        self.layers = []
        self.n_features = n_features
        self.n_outputs = n_outputs
        
        # Input to first hidden layer
        if self.hidden_layers:
            layer = Layer(
                n_features, 
                self.hidden_layers[0], 
                self._get_activation(self.activation_name),
                weight_init=self.weight_init,
                dropout_rate=self.dropout_rate
            )
            self.layers.append(layer)
            
            # Hidden layers
            for i in range(1, len(self.hidden_layers)):
                layer = Layer(
                    self.hidden_layers[i-1], 
                    self.hidden_layers[i], 
                    self._get_activation(self.activation_name),
                    weight_init=self.weight_init,
                    dropout_rate=self.dropout_rate
                )
                self.layers.append(layer)
            
            # Last hidden to output layer
            output_layer = Layer(
                self.hidden_layers[-1], 
                n_outputs, 
                self._get_activation(self.output_activation_name),
                weight_init=self.weight_init,
                dropout_rate=0.0  # No dropout on output layer
            )
            self.layers.append(output_layer)
        else:
            # Direct input to output (no hidden layers)
            output_layer = Layer(
                n_features, 
                n_outputs, 
                self._get_activation(self.output_activation_name),
                weight_init=self.weight_init,
                dropout_rate=0.0
            )
            self.layers.append(output_layer)
        
        # Initialize loss function
        self.loss_function = self.loss_functions[self.loss_name]
        
        # Initialize optimizer for each layer
        if self.optimizer_name == 'sgd':
            self.optimizers = [SGD(self.learning_rate) for _ in self.layers]
        elif self.optimizer_name == 'adam':
            self.optimizers = [Adam(self.learning_rate) for _ in self.layers]
    
    def _forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network"""
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output
    
    def _backward_pass(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Backward pass through the network"""
        # Start with loss gradient
        grad = self.loss_function.backward(y_true, y_pred)
        
        gradients = []
        
        # Backpropagate through layers in reverse order
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            grad, grad_weights, grad_biases = layer.backward(grad)
            
            # Add regularization to weight gradients
            if self.l1_reg > 0:
                grad_weights += self.l1_reg * np.sign(layer.weights)
            if self.l2_reg > 0:
                grad_weights += self.l2_reg * layer.weights
            
            gradients.append((grad_weights, grad_biases))
        
        return list(reversed(gradients))
    
    def _update_weights(self, gradients: List[Tuple[np.ndarray, np.ndarray]]):
        """Update weights using optimizer"""
        for i, (grad_weights, grad_biases) in enumerate(gradients):
            layer = self.layers[i]
            optimizer = self.optimizers[i]
            
            layer.weights, layer.biases = optimizer.update(
                layer.weights, layer.biases, grad_weights, grad_biases
            )
    
    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy for classification tasks"""
        if self.loss_name == 'categorical_crossentropy':
            # Multiclass classification
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
            return np.mean(y_pred_classes == y_true_classes)
        elif self.loss_name == 'binary_crossentropy':
            # Binary classification
            y_pred_classes = (y_pred > 0.5).astype(int)
            return np.mean(y_pred_classes == y_true)
        else:
            # Regression - use R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            verbose: bool = True, X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the neural network
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples, n_outputs)
            Target values
        epochs : int, default=100
            Number of training epochs
        verbose : bool, default=True
            Whether to print training progress
        X_val : ndarray, optional
            Validation data
        y_val : ndarray, optional
            Validation targets
        
        Returns:
        --------
        history : Dict[str, List[float]]
            Training history
        """
        # Prepare data
        X = np.array(X)
        y = np.array(y)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        
        # Build network if not already built
        if not self.layers:
            self._build_network(n_features, n_outputs)
        
        # Split validation data if not provided
        if X_val is None and self.validation_split > 0:
            n_val = int(n_samples * self.validation_split)
            indices = np.random.permutation(n_samples)
            
            X_val = X[indices[:n_val]]
            y_val = y[indices[:n_val]]
            X = X[indices[n_val:]]
            y = y[indices[n_val:]]
            n_samples = len(X)
        
        # Reset history
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Training phase
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Forward pass
                y_pred = self._forward_pass(X_batch, training=True)
                
                # Compute loss
                batch_loss = self.loss_function.forward(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Compute accuracy
                batch_accuracy = self._compute_accuracy(y_batch, y_pred)
                epoch_accuracy += batch_accuracy
                
                # Backward pass
                gradients = self._backward_pass(y_batch, y_pred)
                
                # Update weights
                self._update_weights(gradients)
                
                n_batches += 1
            
            # Average metrics
            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_accuracy)
            
            # Validation phase
            if X_val is not None:
                val_pred = self._forward_pass(X_val, training=False)
                val_loss = self.loss_function.forward(y_val, val_pred)
                val_accuracy = self._compute_accuracy(y_val, val_pred)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                # Early stopping check
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best weights
                        best_weights = [layer.weights.copy() for layer in self.layers]
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        
                        # Restore best weights
                        if best_weights:
                            for i, weights in enumerate(best_weights):
                                self.layers[i].weights = weights
                        break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"loss: {epoch_loss:.4f} - "
                          f"accuracy: {epoch_accuracy:.4f} - "
                          f"val_loss: {val_loss:.4f} - "
                          f"val_accuracy: {val_accuracy:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"loss: {epoch_loss:.4f} - "
                          f"accuracy: {epoch_accuracy:.4f}")
        
        self.is_fitted = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._forward_pass(X, training=False)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy/R² score"""
        predictions = self.predict(X)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return self._compute_accuracy(y, predictions)
    
    def plot_history(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot training history"""
        if not self.history['loss']:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(self.history['loss'], 'b-', label='Training Loss', linewidth=2)
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if self.history['val_accuracy']:
            axes[1].plot(self.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        model_data = {
            'layers': [
                {
                    'weights': layer.weights,
                    'biases': layer.biases,
                    'n_inputs': layer.n_inputs,
                    'n_neurons': layer.n_neurons,
                    'dropout_rate': layer.dropout_rate
                }
                for layer in self.layers
            ],
            'config': {
                'hidden_layers': self.hidden_layers,
                'activation_name': self.activation_name,
                'output_activation_name': self.output_activation_name,
                'loss_name': self.loss_name,
                'optimizer_name': self.optimizer_name,
                'learning_rate': self.learning_rate,
                'l1_reg': self.l1_reg,
                'l2_reg': self.l2_reg,
                'dropout_rate': self.dropout_rate,
                'weight_init': self.weight_init
            },
            'history': self.history,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features,
            'n_outputs': self.n_outputs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore configuration
        config = model_data['config']
        for key, value in config.items():
            setattr(self, key, value)
        
        # Restore model state
        self.history = model_data['history']
        self.is_fitted = model_data['is_fitted']
        self.n_features = model_data['n_features']
        self.n_outputs = model_data['n_outputs']
        
        # Rebuild network structure
        if self.n_features and self.n_outputs:
            self._build_network(self.n_features, self.n_outputs)
            
            # Restore weights and biases
            for i, layer_data in enumerate(model_data['layers']):
                self.layers[i].weights = layer_data['weights']
                self.layers[i].biases = layer_data['biases']


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    
    print("Testing Neural Network Implementation")
    print("=" * 50)
    
    # Test regression
    print("\nRegression Test:")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_reg_scaled = scaler.fit_transform(X_reg)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_scaled, y_reg, test_size=0.3, random_state=42
    )
    
    # Create and train regression model
    reg_model = NeuralNetwork(
        hidden_layers=[64, 32],
        activation='relu',
        output_activation='linear',
        loss='mse',
        optimizer='adam',
        learning_rate=0.001,
        l2_reg=0.01,
        dropout_rate=0.1,
        early_stopping=True,
        patience=10,
        random_state=42
    )
    
    history = reg_model.fit(X_train_reg, y_train_reg, epochs=100, verbose=False)
    
    train_score = reg_model.score(X_train_reg, y_train_reg)
    test_score = reg_model.score(X_test_reg, y_test_reg)
    
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Test binary classification
    print("\nBinary Classification Test:")
    X_bin, y_bin = make_classification(
        n_samples=1000, n_features=20, n_redundant=0, n_informative=20,
        n_clusters_per_class=1, random_state=42
    )
    
    # Normalize features
    X_bin_scaled = scaler.fit_transform(X_bin)
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin_scaled, y_bin, test_size=0.3, random_state=42
    )
    
    # Create and train binary classification model
    bin_model = NeuralNetwork(
        hidden_layers=[32, 16],
        activation='relu',
        output_activation='sigmoid',
        loss='binary_crossentropy',
        optimizer='adam',
        learning_rate=0.001,
        dropout_rate=0.2,
        early_stopping=True,
        random_state=42
    )
    
    history = bin_model.fit(X_train_bin, y_train_bin, epochs=100, verbose=False)
    
    train_acc = bin_model.score(X_train_bin, y_train_bin)
    test_acc = bin_model.score(X_test_bin, y_test_bin)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Test multiclass classification
    print("\nMulticlass Classification Test:")
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=20, n_redundant=0, n_informative=20,
        n_clusters_per_class=1, n_classes=3, random_state=42
    )
    
    # Convert to one-hot encoding
    
    # Normalize features
    X_multi_scaled = scaler.fit_transform(X_multi)
    
    # One-hot encode targets
    onehot = OneHotEncoder(sparse_output=False)
    y_multi_onehot = onehot.fit_transform(y_multi.reshape(-1, 1))
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi_scaled, y_multi_onehot, test_size=0.3, random_state=42
    )
    
    # Create and train multiclass model
    multi_model = NeuralNetwork(
        hidden_layers=[64, 32],
        activation='relu',
        output_activation='softmax',
        loss='categorical_crossentropy',
        optimizer='adam',
        learning_rate=0.001,
        dropout_rate=0.1,
        early_stopping=True,
        random_state=42
    )
    
    history = multi_model.fit(X_train_multi, y_train_multi, epochs=100, verbose=False)
    
    train_acc = multi_model.score(X_train_multi, y_train_multi)
    test_acc = multi_model.score(X_test_multi, y_test_multi)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nAll tests completed successfully!")
