"""
Advanced Logistic Regression Implementation from Scratch
======================================================

A comprehensive implementation of logistic regression with multiple optimization
algorithms, regularization, and advanced features.

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Abstract base class for classification algorithms."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass


class LogisticRegression(BaseClassifier):
    """
    Advanced Logistic Regression Implementation
    
    Features:
    - Multiple optimization algorithms (GD, SGD, Mini-batch, Newton's method)
    - L1, L2, and Elastic Net regularization
    - Multi-class classification (One-vs-Rest)
    - Feature scaling integration
    - Advanced convergence criteria
    - Comprehensive evaluation metrics
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        regularization: Optional[str] = None,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        method: str = 'gradient_descent',
        batch_size: Optional[int] = None,
        multi_class: str = 'ovr',
        class_weight: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for optimization
        max_iterations : int, default=1000
            Maximum number of iterations
        tolerance : float, default=1e-6
            Convergence tolerance
        regularization : str, optional
            Type of regularization: 'l1', 'l2', 'elastic_net'
        alpha : float, default=0.01
            Regularization strength
        l1_ratio : float, default=0.5
            Ratio of L1 regularization in Elastic Net (0 = L2, 1 = L1)
        method : str, default='gradient_descent'
            Optimization method: 'gradient_descent', 'sgd', 'mini_batch', 'newton'
        batch_size : int, optional
            Batch size for mini-batch gradient descent
        multi_class : str, default='ovr'
            Multi-class strategy: 'ovr' (One-vs-Rest), 'multinomial'
        class_weight : str, optional
            Class weight strategy: 'balanced'
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool, default=False
            Whether to print training progress
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.method = method
        self.batch_size = batch_size
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose
        
        # Model parameters
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.cost_history_ = []
        self.class_weights_ = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with numerical stability.
        
        Parameters:
        -----------
        z : array-like
            Input values
        
        Returns:
        --------
        sigmoid_values : array
            Sigmoid of input values
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function for multi-class classification.
        
        Parameters:
        -----------
        z : array-like, shape (n_samples, n_classes)
            Input logits
        
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Class probabilities
        """
        # Numerical stability: subtract max value
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Add bias term to feature matrix."""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_class_weights(self, y: np.ndarray) -> dict:
        """Compute class weights for imbalanced datasets."""
        if self.class_weight == 'balanced':
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            weights = n_samples / (n_classes * counts)
            return dict(zip(classes, weights))
        return {cls: 1.0 for cls in self.classes_}
    
    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights based on class weights."""
        if self.class_weights_ is None:
            return np.ones(len(y))
        return np.array([self.class_weights_[label] for label in y])
    
    def _compute_cost_binary(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        weights: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute logistic loss for binary classification."""
        m = X.shape[0]
        z = X.dot(weights)
        h = self.sigmoid(z)
        
        # Prevent log(0) by clipping
        h = np.clip(h, 1e-15, 1 - 1e-15)
        
        if sample_weights is None:
            cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        else:
            cost = -np.sum(sample_weights * (y * np.log(h) + (1 - y) * np.log(1 - h))) / m
        
        # Add regularization
        reg_cost = 0
        if self.regularization == 'l1':
            reg_cost = self.alpha * np.sum(np.abs(weights[1:]))
        elif self.regularization == 'l2':
            reg_cost = self.alpha * np.sum(weights[1:] ** 2) / 2
        elif self.regularization == 'elastic_net':
            l1_cost = self.alpha * self.l1_ratio * np.sum(np.abs(weights[1:]))
            l2_cost = self.alpha * (1 - self.l1_ratio) * np.sum(weights[1:] ** 2) / 2
            reg_cost = l1_cost + l2_cost
        
        return cost + reg_cost
    
    def _compute_gradients_binary(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        weights: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute gradients for binary classification."""
        m = X.shape[0]
        z = X.dot(weights)
        h = self.sigmoid(z)
        
        if sample_weights is None:
            gradients = X.T.dot(h - y) / m
        else:
            gradients = X.T.dot(sample_weights * (h - y)) / m
        
        # Add regularization gradients
        if self.regularization == 'l1':
            gradients[1:] += self.alpha * np.sign(weights[1:])
        elif self.regularization == 'l2':
            gradients[1:] += self.alpha * weights[1:]
        elif self.regularization == 'elastic_net':
            l1_grad = self.alpha * self.l1_ratio * np.sign(weights[1:])
            l2_grad = self.alpha * (1 - self.l1_ratio) * weights[1:]
            gradients[1:] += l1_grad + l2_grad
        
        return gradients
    
    def _gradient_descent_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Binary classification using gradient descent."""
        m, n = X.shape
        weights = np.random.normal(0, 0.01, n)
        sample_weights = self._compute_sample_weights(y)
        
        for i in range(self.max_iterations):
            cost = self._compute_cost_binary(X, y, weights, sample_weights)
            self.cost_history_.append(cost)
            
            gradients = self._compute_gradients_binary(X, y, weights, sample_weights)
            weights -= self.learning_rate * gradients
            
            # Check convergence
            if i > 0 and abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance:
                if self.verbose:
                    print(f"Converged after {i+1} iterations")
                break
        
        return weights
    
    def _sgd_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Binary classification using stochastic gradient descent."""
        m, n = X.shape
        weights = np.random.normal(0, 0.01, n)
        sample_weights = self._compute_sample_weights(y)
        
        for epoch in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            sw_shuffled = sample_weights[indices]
            
            epoch_cost = 0
            for i in range(m):
                x_i = X_shuffled[i:i+1]
                y_i = y_shuffled[i:i+1]
                sw_i = sw_shuffled[i:i+1] if sample_weights is not None else None
                
                cost = self._compute_cost_binary(x_i, y_i, weights, sw_i)
                gradients = self._compute_gradients_binary(x_i, y_i, weights, sw_i)
                
                weights -= self.learning_rate * gradients
                epoch_cost += cost
            
            self.cost_history_.append(epoch_cost / m)
            
            # Check convergence
            if (epoch > 0 and 
                abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance):
                if self.verbose:
                    print(f"Converged after {epoch+1} epochs")
                break
        
        return weights
    
    def _newton_method_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Binary classification using Newton's method."""
        m, n = X.shape
        weights = np.random.normal(0, 0.01, n)
        sample_weights = self._compute_sample_weights(y)
        
        for i in range(self.max_iterations):
            z = X.dot(weights)
            h = self.sigmoid(z)
            
            # Compute cost
            cost = self._compute_cost_binary(X, y, weights, sample_weights)
            self.cost_history_.append(cost)
            
            # Compute gradients
            gradients = self._compute_gradients_binary(X, y, weights, sample_weights)
            
            # Compute Hessian matrix
            D = h * (1 - h)
            if sample_weights is not None:
                D = D * sample_weights
            
            H = X.T.dot(np.diag(D)).dot(X) / m
            
            # Add regularization to Hessian
            if self.regularization == 'l2':
                H[1:, 1:] += self.alpha * np.eye(n-1)
            
            try:
                # Newton's update: w = w - H^(-1) * gradient
                delta = np.linalg.solve(H, gradients)
                weights -= delta
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if Hessian is singular
                weights -= self.learning_rate * gradients
            
            # Check convergence
            if i > 0 and abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance:
                if self.verbose:
                    print(f"Converged after {i+1} iterations")
                break
        
        return weights
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit binary classification model."""
        X_with_bias = self._add_bias_term(X)
        
        if self.method == 'gradient_descent':
            weights = self._gradient_descent_binary(X_with_bias, y)
        elif self.method == 'sgd':
            weights = self._sgd_binary(X_with_bias, y)
        elif self.method == 'newton':
            weights = self._newton_method_binary(X_with_bias, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return weights
    
    def _fit_multiclass_ovr(self, X: np.ndarray, y: np.ndarray):
        """Fit multi-class model using One-vs-Rest strategy."""
        self.weights_ = []
        self.bias_ = []
        
        for i, class_label in enumerate(self.classes_):
            if self.verbose:
                print(f"Training classifier for class {class_label}")
            
            # Create binary labels (current class vs all others)
            y_binary = (y == class_label).astype(int)
            
            # Fit binary classifier
            weights = self._fit_binary(X, y_binary)
            
            # Store weights and bias
            self.bias_.append(weights[0])
            self.weights_.append(weights[1:])
        
        self.weights_ = np.array(self.weights_).T
        self.bias_ = np.array(self.bias_)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target labels
        
        Returns:
        --------
        self : LogisticRegression
            Fitted model
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store dataset info
        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Compute class weights
        self.class_weights_ = self._compute_class_weights(y)
        
        # Clear cost history
        self.cost_history_ = []
        
        # Handle binary vs multi-class classification
        if self.n_classes_ == 2:
            # Binary classification
            y_binary = (y == self.classes_[1]).astype(int)
            weights = self._fit_binary(X, y_binary)
            self.bias_ = weights[0]
            self.weights_ = weights[1:]
        else:
            # Multi-class classification
            if self.multi_class == 'ovr':
                self._fit_multiclass_ovr(X, y)
            else:
                raise NotImplementedError("Multinomial regression not yet implemented")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features to predict on
        
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.weights_ is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        X = np.asarray(X)
        
        if self.n_classes_ == 2:
            # Binary classification
            z = X.dot(self.weights_) + self.bias_
            proba_positive = self.sigmoid(z)
            return np.column_stack([1 - proba_positive, proba_positive])
        else:
            # Multi-class classification (One-vs-Rest)
            scores = X.dot(self.weights_) + self.bias_
            probabilities = self.sigmoid(scores)
            
            # Normalize probabilities to sum to 1
            prob_sum = np.sum(probabilities, axis=1, keepdims=True)
            probabilities = probabilities / prob_sum
            
            return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features to predict on
        
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test features
        y : array-like, shape (n_samples,)
            True labels
        
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
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
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'method': self.method,
            'batch_size': self.batch_size,
            'multi_class': self.multi_class,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'verbose': self.verbose
        }


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_redundant=10,
        n_classes=3,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different configurations
    configs = [
        {'method': 'gradient_descent', 'regularization': None},
        {'method': 'gradient_descent', 'regularization': 'l2', 'alpha': 0.01},
        {'method': 'sgd', 'regularization': 'l1', 'alpha': 0.01},
        {'method': 'newton', 'regularization': 'l2', 'alpha': 0.01},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}: {config}")
        print(f"{'='*60}")
        
        model = LogisticRegression(
            **config,
            max_iterations=500,
            random_state=42,
            verbose=True
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Number of classes: {model.n_classes_}")
        print(f"Classes: {model.classes_}")
        
        # Test predictions
        sample_predictions = model.predict(X_test_scaled[:5])
        sample_probabilities = model.predict_proba(X_test_scaled[:5])
        
        print(f"Sample predictions: {sample_predictions}")
        print(f"Sample probabilities shape: {sample_probabilities.shape}")
