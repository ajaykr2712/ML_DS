"""
Advanced Linear Regression Implementation from Scratch
=====================================================

A comprehensive implementation of linear regression with multiple algorithms,
regularization techniques, and advanced features.

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from abc import ABC, abstractmethod
import warnings


class BaseRegressor(ABC):
    """Abstract base class for regression algorithms."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseRegressor':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass


class LinearRegression(BaseRegressor):
    """
    Advanced Linear Regression Implementation
    
    Supports multiple solving methods:
    - Normal Equation (closed-form solution)
    - Gradient Descent
    - Stochastic Gradient Descent
    - Mini-batch Gradient Descent
    """
    
    def __init__(
        self,
        method: str = 'normal_equation',
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        batch_size: Optional[int] = None,
        regularization: Optional[str] = None,
        alpha: float = 0.01,
        random_state: Optional[int] = None
    ):
        """
        Initialize Linear Regression model.
        
        Parameters:
        -----------
        method : str, default='normal_equation'
            Solving method: 'normal_equation', 'gradient_descent', 'sgd', 'mini_batch'
        learning_rate : float, default=0.01
            Learning rate for gradient-based methods
        max_iterations : int, default=1000
            Maximum number of iterations for gradient-based methods
        tolerance : float, default=1e-6
            Convergence tolerance
        batch_size : int, optional
            Batch size for mini-batch gradient descent
        regularization : str, optional
            Regularization type: 'l1', 'l2', 'elastic_net'
        alpha : float, default=0.01
            Regularization strength
        random_state : int, optional
            Random seed for reproducibility
        """
        self.method = method
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        
        # Model parameters
        self.weights_ = None
        self.bias_ = None
        self.cost_history_ = []
        self.n_features_ = None
        self.n_samples_ = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (intercept) to feature matrix."""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Compute cost function with optional regularization."""
        m = X.shape[0]
        predictions = X.dot(weights)
        mse_cost = np.sum((predictions - y) ** 2) / (2 * m)
        
        # Add regularization term
        regularization_cost = 0
        if self.regularization == 'l1':
            regularization_cost = self.alpha * np.sum(np.abs(weights[1:]))
        elif self.regularization == 'l2':
            regularization_cost = self.alpha * np.sum(weights[1:] ** 2)
        elif self.regularization == 'elastic_net':
            l1_term = 0.5 * self.alpha * np.sum(np.abs(weights[1:]))
            l2_term = 0.5 * self.alpha * np.sum(weights[1:] ** 2)
            regularization_cost = l1_term + l2_term
        
        return mse_cost + regularization_cost
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute gradients with optional regularization."""
        m = X.shape[0]
        predictions = X.dot(weights)
        gradients = X.T.dot(predictions - y) / m
        
        # Add regularization gradients
        if self.regularization == 'l1':
            gradients[1:] += self.alpha * np.sign(weights[1:])
        elif self.regularization == 'l2':
            gradients[1:] += 2 * self.alpha * weights[1:]
        elif self.regularization == 'elastic_net':
            gradients[1:] += self.alpha * (0.5 * np.sign(weights[1:]) + weights[1:])
        
        return gradients
    
    def _normal_equation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using normal equation (closed-form solution)."""
        try:
            # Add regularization to normal equation if specified
            if self.regularization == 'l2':
                regularization_matrix = self.alpha * np.eye(X.shape[1])
                regularization_matrix[0, 0] = 0  # Don't regularize bias term
                weights = np.linalg.inv(X.T.dot(X) + regularization_matrix).dot(X.T).dot(y)
            else:
                weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            return weights
        except np.linalg.LinAlgError:
            warnings.warn("Matrix is singular, using pseudo-inverse instead.")
            return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using batch gradient descent."""
        m, n = X.shape
        weights = np.random.normal(0, 0.01, n)
        
        for i in range(self.max_iterations):
            cost = self._compute_cost(X, y, weights)
            self.cost_history_.append(cost)
            
            gradients = self._compute_gradients(X, y, weights)
            weights -= self.learning_rate * gradients
            
            # Check for convergence
            if i > 0 and abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
        
        return weights
    
    def _stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using stochastic gradient descent."""
        m, n = X.shape
        weights = np.random.normal(0, 0.01, n)
        
        for epoch in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            for i in range(m):
                x_i = X_shuffled[i:i+1]
                y_i = y_shuffled[i:i+1]
                
                prediction = x_i.dot(weights)
                gradient = x_i.T.dot(prediction - y_i)
                
                # Add regularization gradient
                if self.regularization == 'l2':
                    gradient[1:] += 2 * self.alpha * weights[1:]
                
                weights -= self.learning_rate * gradient.flatten()
                epoch_cost += (prediction - y_i) ** 2
            
            self.cost_history_.append(epoch_cost[0] / (2 * m))
            
            # Check for convergence
            if (epoch > 0 and 
                abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance):
                print(f"Converged after {epoch+1} epochs")
                break
        
        return weights
    
    def _mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve using mini-batch gradient descent."""
        m, n = X.shape
        weights = np.random.normal(0, 0.01, n)
        
        if self.batch_size is None:
            self.batch_size = min(32, m)
        
        for epoch in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            num_batches = m // self.batch_size
            
            for i in range(0, m, self.batch_size):
                end_idx = min(i + self.batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                gradients = self._compute_gradients(X_batch, y_batch, weights)
                weights -= self.learning_rate * gradients
                
                batch_cost = self._compute_cost(X_batch, y_batch, weights)
                epoch_cost += batch_cost
            
            self.cost_history_.append(epoch_cost / num_batches)
            
            # Check for convergence
            if (epoch > 0 and 
                abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance):
                print(f"Converged after {epoch+1} epochs")
                break
        
        return weights
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : LinearRegression
            Fitted model
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store dataset info
        self.n_samples_, self.n_features_ = X.shape
        
        # Add bias term
        X_with_bias = self._add_bias_term(X)
        
        # Clear cost history
        self.cost_history_ = []
        
        # Choose solving method
        if self.method == 'normal_equation':
            if self.regularization in ['l1', 'elastic_net']:
                warnings.warn(f"Normal equation doesn't support {self.regularization} regularization. "
                             "Switching to gradient descent.")
                self.method = 'gradient_descent'
                weights = self._gradient_descent(X_with_bias, y)
            else:
                weights = self._normal_equation(X_with_bias, y)
        elif self.method == 'gradient_descent':
            weights = self._gradient_descent(X_with_bias, y)
        elif self.method == 'sgd':
            weights = self._stochastic_gradient_descent(X_with_bias, y)
        elif self.method == 'mini_batch':
            weights = self._mini_batch_gradient_descent(X_with_bias, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Extract bias and weights
        self.bias_ = weights[0]
        self.weights_ = weights[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features to predict on
        
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted values
        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.asarray(X)
        return X.dot(self.weights_) + self.bias_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared score.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test features
        y : array-like, shape (n_samples,)
            True target values
        
        Returns:
        --------
        score : float
            R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self) -> None:
        """Plot the cost function history during training."""
        if not self.cost_history_:
            print("No cost history available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history_)
        plt.title('Cost Function History')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'method': self.method,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'batch_size': self.batch_size,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'random_state': self.random_state
        }


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_weights = np.array([2.5, -1.3])
    true_bias = 0.8
    noise = np.random.randn(100) * 0.1
    y = X.dot(true_weights) + true_bias + noise
    
    # Test different methods
    methods = ['normal_equation', 'gradient_descent', 'sgd', 'mini_batch']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing method: {method}")
        print(f"{'='*50}")
        
        model = LinearRegression(method=method, random_state=42)
        model.fit(X, y)
        
        print(f"True weights: {true_weights}")
        print(f"Learned weights: {model.weights_}")
        print(f"True bias: {true_bias}")
        print(f"Learned bias: {model.bias_}")
        print(f"R-squared score: {model.score(X, y):.4f}")
        
        # Test predictions
        X_test = np.array([[1, 1], [-1, -1]])
        predictions = model.predict(X_test)
        print(f"Test predictions: {predictions}")
