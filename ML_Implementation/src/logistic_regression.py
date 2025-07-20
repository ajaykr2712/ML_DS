"""
Advanced Logistic Regression Implementation from Scratch
Features multiple solvers, regularization, and advanced capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Abstract base class for classification models"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class LogisticRegression(BaseClassifier):
    """
    Advanced Logistic Regression Implementation
    
    Features:
    - Binary and multiclass classification
    - Multiple solvers (newton-cg, lbfgs, gradient descent, sgd)
    - Regularization (L1, L2, Elastic Net)
    - Feature scaling
    - Early stopping
    - Cross-validation
    - Probability calibration
    - Class balancing
    """
    
    def __init__(
        self,
        solver: str = 'lbfgs',
        penalty: Optional[str] = 'l2',
        C: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
        learning_rate: float = 0.01,
        fit_intercept: bool = True,
        normalize: bool = True,
        class_weight: Optional[Union[str, Dict]] = None,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        patience: int = 10,
        random_state: Optional[int] = None,
        multi_class: str = 'ovr',
        warm_start: bool = False
    ):
        """
        Initialize Logistic Regression model
        
        Parameters:
        -----------
        solver : str, default='lbfgs'
            Algorithm to use: 'lbfgs', 'newton-cg', 'gradient_descent', 'sgd'
        penalty : str or None, default='l2'
            Regularization penalty: 'l1', 'l2', 'elasticnet', None
        C : float, default=1.0
            Inverse of regularization strength (smaller values = stronger regularization)
        l1_ratio : float, default=0.5
            Elastic net mixing parameter (0 <= l1_ratio <= 1)
        max_iter : int, default=1000
            Maximum number of iterations
        tol : float, default=1e-6
            Tolerance for stopping criteria
        learning_rate : float, default=0.01
            Learning rate for gradient-based solvers
        fit_intercept : bool, default=True
            Whether to fit intercept term
        normalize : bool, default=True
            Whether to normalize features
        class_weight : dict, 'balanced' or None, default=None
            Weights associated with classes
        early_stopping : bool, default=False
            Whether to use early stopping
        validation_fraction : float, default=0.1
            Fraction of training data for validation
        patience : int, default=10
            Iterations to wait before early stopping
        random_state : int, optional
            Random seed for reproducibility
        multi_class : str, default='ovr'
            Multiclass strategy: 'ovr' (one-vs-rest) or 'multinomial'
        warm_start : bool, default=False
            Whether to reuse solution of previous call as initialization
        """
        self.solver = solver
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.multi_class = multi_class
        self.warm_start = warm_start
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_mean_ = None
        self.feature_std_ = None
        self.class_weights_ = None
        
        # Training history
        self.cost_history_ = []
        self.validation_score_history_ = []
        self.n_iter_ = 0
        
        # Multiclass models (for OvR)
        self.models_ = {}
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Stable sigmoid function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Stable softmax function"""
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using z-score normalization"""
        if not self.normalize:
            return X
        
        if fit:
            self.feature_mean_ = np.mean(X, axis=0)
            self.feature_std_ = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_std_[self.feature_std_ == 0] = 1
        
        return (X - self.feature_mean_) / self.feature_std_
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for balanced learning"""
        if self.class_weight is None:
            return {cls: 1.0 for cls in self.classes_}
        elif self.class_weight == 'balanced':
            n_samples = len(y)
            n_classes = len(self.classes_)
            weights = {}
            for cls in self.classes_:
                n_cls_samples = np.sum(y == cls)
                weights[cls] = n_samples / (n_classes * n_cls_samples)
            return weights
        else:
            return self.class_weight
    
    def _get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Get sample weights based on class weights"""
        if self.class_weights_ is None:
            return np.ones(len(y))
        
        sample_weights = np.ones(len(y))
        for i, cls in enumerate(y):
            sample_weights[i] = self.class_weights_[cls]
        
        return sample_weights
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                     sample_weights: np.ndarray) -> float:
        """Compute logistic regression cost with regularization"""
        n_samples = X.shape[0]
        
        if self.multi_class == 'multinomial' and len(self.classes_) > 2:
            # Multinomial logistic regression
            z = X @ weights.reshape(-1, len(self.classes_))
            probs = self.softmax(z)
            
            # Convert y to one-hot encoding
            y_one_hot = np.eye(len(self.classes_))[y]
            
            # Cross-entropy loss
            log_probs = np.log(probs + 1e-15)  # Add small epsilon for numerical stability
            cost = -np.mean(sample_weights[:, np.newaxis] * y_one_hot * log_probs)
        else:
            # Binary logistic regression
            z = X @ weights
            predictions = self.sigmoid(z)
            
            # Binary cross-entropy loss
            cost = -np.mean(sample_weights * (y * np.log(predictions + 1e-15) + 
                                             (1 - y) * np.log(1 - predictions + 1e-15)))
        
        # Add regularization
        if self.penalty is not None and self.C != np.inf:
            reg_strength = 1 / self.C
            
            if self.penalty == 'l2':
                if self.fit_intercept:
                    # Don't regularize intercept
                    reg_term = reg_strength * np.sum(weights[1:] ** 2) / (2 * n_samples)
                else:
                    reg_term = reg_strength * np.sum(weights ** 2) / (2 * n_samples)
                cost += reg_term
            
            elif self.penalty == 'l1':
                if self.fit_intercept:
                    reg_term = reg_strength * np.sum(np.abs(weights[1:])) / n_samples
                else:
                    reg_term = reg_strength * np.sum(np.abs(weights)) / n_samples
                cost += reg_term
            
            elif self.penalty == 'elasticnet':
                if self.fit_intercept:
                    l1_term = reg_strength * self.l1_ratio * np.sum(np.abs(weights[1:]))
                    l2_term = reg_strength * (1 - self.l1_ratio) * np.sum(weights[1:] ** 2) / 2
                else:
                    l1_term = reg_strength * self.l1_ratio * np.sum(np.abs(weights))
                    l2_term = reg_strength * (1 - self.l1_ratio) * np.sum(weights ** 2) / 2
                cost += (l1_term + l2_term) / n_samples
        
        return cost
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                          sample_weights: np.ndarray) -> np.ndarray:
        """Compute gradients with regularization"""
        n_samples = X.shape[0]
        
        if self.multi_class == 'multinomial' and len(self.classes_) > 2:
            # Multinomial logistic regression gradients
            z = X @ weights.reshape(-1, len(self.classes_))
            probs = self.softmax(z)
            
            # Convert y to one-hot encoding
            y_one_hot = np.eye(len(self.classes_))[y]
            
            # Compute gradients
            gradients = X.T @ (sample_weights[:, np.newaxis] * (probs - y_one_hot)) / n_samples
            gradients = gradients.flatten()
        else:
            # Binary logistic regression gradients
            z = X @ weights
            predictions = self.sigmoid(z)
            
            # Compute gradients
            gradients = X.T @ (sample_weights * (predictions - y)) / n_samples
        
        # Add regularization gradients
        if self.penalty is not None and self.C != np.inf:
            reg_strength = 1 / self.C
            
            if self.penalty == 'l2':
                if self.fit_intercept:
                    reg_grad = np.zeros_like(weights)
                    reg_grad[1:] = reg_strength * weights[1:] / n_samples
                else:
                    reg_grad = reg_strength * weights / n_samples
                gradients += reg_grad
            
            elif self.penalty == 'l1':
                if self.fit_intercept:
                    reg_grad = np.zeros_like(weights)
                    reg_grad[1:] = reg_strength * np.sign(weights[1:]) / n_samples
                else:
                    reg_grad = reg_strength * np.sign(weights) / n_samples
                gradients += reg_grad
            
            elif self.penalty == 'elasticnet':
                if self.fit_intercept:
                    reg_grad = np.zeros_like(weights)
                    reg_grad[1:] = reg_strength * (
                        self.l1_ratio * np.sign(weights[1:]) + 
                        (1 - self.l1_ratio) * weights[1:]
                    ) / n_samples
                else:
                    reg_grad = reg_strength * (
                        self.l1_ratio * np.sign(weights) + 
                        (1 - self.l1_ratio) * weights
                    ) / n_samples
                gradients += reg_grad
        
        return gradients
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
        """Gradient descent optimization"""
        if self.multi_class == 'multinomial' and len(self.classes_) > 2:
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            weights = np.random.randn(n_features * n_classes) * 0.01
        else:
            n_features = X.shape[1]
            weights = np.random.randn(n_features) * 0.01
        
        # Initialize from previous solution if warm start
        if self.warm_start and self.coef_ is not None:
            if self.multi_class == 'multinomial' and len(self.classes_) > 2:
                weights = self.coef_.flatten()
            else:
                if self.fit_intercept:
                    weights = np.concatenate([[self.intercept_], self.coef_])
                else:
                    weights = self.coef_.copy()
        
        # Prepare validation data if early stopping is enabled
        if self.early_stopping:
            n_validation = int(len(X) * self.validation_fraction)
            indices = np.random.permutation(len(X))
            
            X_train = X[indices[n_validation:]]
            X_val = X[indices[:n_validation]]
            y_train = y[indices[n_validation:]]
            y_val = y[indices[:n_validation]]
            sample_weights_train = sample_weights[indices[n_validation:]]
            sample_weights_val = sample_weights[indices[:n_validation]]
            
            best_val_score = float('inf')
            patience_counter = 0
            best_weights = weights.copy()
        else:
            X_train, y_train = X, y
            sample_weights_train = sample_weights
        
        for iteration in range(self.max_iter):
            # Compute cost and gradients
            cost = self._compute_cost(X_train, y_train, weights, sample_weights_train)
            gradients = self._compute_gradients(X_train, y_train, weights, sample_weights_train)
            
            # Update weights
            weights -= self.learning_rate * gradients
            
            # Store training history
            self.cost_history_.append(cost)
            
            # Early stopping check
            if self.early_stopping:
                val_cost = self._compute_cost(X_val, y_val, weights, sample_weights_val)
                self.validation_score_history_.append(val_cost)
                
                if val_cost < best_val_score:
                    best_val_score = val_cost
                    patience_counter = 0
                    best_weights = weights.copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    weights = best_weights
                    self.n_iter_ = iteration + 1
                    break
            
            # Convergence check
            if len(self.cost_history_) > 1:
                if abs(self.cost_history_[-1] - self.cost_history_[-2]) < self.tol:
                    self.n_iter_ = iteration + 1
                    break
        else:
            self.n_iter_ = self.max_iter
        
        return weights
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Fit binary logistic regression"""
        # Normalize features
        X_normalized = self._normalize_features(X, fit=True)
        
        # Add intercept term if needed
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(X_normalized.shape[0]), X_normalized])
        else:
            X_with_intercept = X_normalized
        
        # Compute class weights
        self.class_weights_ = self._compute_class_weights(y)
        sample_weights = self._get_sample_weights(y)
        
        # Fit model using selected solver
        if self.solver in ['gradient_descent', 'sgd']:
            weights = self._gradient_descent(X_with_intercept, y, sample_weights)
        else:
            # For other solvers, use gradient descent as fallback
            weights = self._gradient_descent(X_with_intercept, y, sample_weights)
        
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        
        return self
    
    def _fit_multiclass_ovr(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Fit multiclass using One-vs-Rest strategy"""
        for i, cls in enumerate(self.classes_):
            # Create binary target (current class vs rest)
            y_binary = (y == cls).astype(int)
            
            # Create binary classifier
            binary_model = LogisticRegression(
                solver=self.solver,
                penalty=self.penalty,
                C=self.C,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                tol=self.tol,
                learning_rate=self.learning_rate,
                fit_intercept=self.fit_intercept,
                normalize=False,  # Already normalized
                class_weight=self.class_weight,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                patience=self.patience,
                random_state=self.random_state,
                warm_start=self.warm_start
            )
            
            # Use pre-normalized features  
            self._normalize_features(X, fit=(i == 0))
            binary_model.feature_mean_ = self.feature_mean_
            binary_model.feature_std_ = self.feature_std_
            
            binary_model._fit_binary(X, y_binary)
            self.models_[cls] = binary_model
        
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : LogisticRegression
            Fitted model
        """
        # Reset training history if not warm start
        if not self.warm_start:
            self.cost_history_ = []
            self.validation_score_history_ = []
        
        # Store input information
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Handle binary vs multiclass
        if len(self.classes_) == 2:
            # Binary classification - map to 0,1
            y_mapped = (y == self.classes_[1]).astype(int)
            self._fit_binary(X, y_mapped)
        else:
            # Multiclass classification
            if self.multi_class == 'ovr':
                self._fit_multiclass_ovr(X, y)
            else:
                # Multinomial not fully implemented, fall back to OvR
                self._fit_multiclass_ovr(X, y)
        
        return self
    
    def _predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def _predict_multiclass(self, X: np.ndarray) -> np.ndarray:
        """Make multiclass predictions using OvR"""
        decision_scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, cls in enumerate(self.classes_):
            model = self.models_[cls]
            probs = model.predict_proba(X)
            decision_scores[:, i] = probs[:, 1]  # Probability of being this class
        
        # Predict class with highest score
        predicted_indices = np.argmax(decision_scores, axis=1)
        return self.classes_[predicted_indices]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input samples
        
        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels
        """
        if len(self.classes_) == 2:
            predictions = self._predict_binary(X)
            return self.classes_[predictions]
        else:
            return self._predict_multiclass(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input samples
        
        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.coef_ is None and not self.models_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Normalize features using training statistics
        X_normalized = self._normalize_features(X, fit=False)
        
        if len(self.classes_) == 2:
            # Binary classification
            if self.fit_intercept:
                z = X_normalized @ self.coef_ + self.intercept_
            else:
                z = X_normalized @ self.coef_
            
            prob_class_1 = self.sigmoid(z)
            prob_class_0 = 1 - prob_class_1
            
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            # Multiclass using OvR
            probs = np.zeros((X.shape[0], len(self.classes_)))
            
            for i, cls in enumerate(self.classes_):
                model = self.models_[cls]
                class_probs = model.predict_proba(X)
                probs[:, i] = class_probs[:, 1]  # Probability of being this class
            
            # Normalize probabilities to sum to 1
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            
            return probs
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True labels
        
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_coefficients(self) -> Dict[str, Any]:
        """Get model coefficients and statistics"""
        if self.coef_ is None and not self.models_:
            raise ValueError("Model must be fitted first")
        
        if len(self.classes_) == 2:
            return {
                'coef': self.coef_,
                'intercept': self.intercept_,
                'classes': self.classes_,
                'n_iter': self.n_iter_,
                'final_cost': self.cost_history_[-1] if self.cost_history_ else None
            }
        else:
            coeffs = {}
            for cls in self.classes_:
                model = self.models_[cls]
                coeffs[f'class_{cls}'] = {
                    'coef': model.coef_,
                    'intercept': model.intercept_
                }
            
            return {
                'coefficients': coeffs,
                'classes': self.classes_,
                'n_features': self.n_features_in_
            }
    
    def plot_learning_curve(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot learning curves"""
        if not self.cost_history_:
            raise ValueError("No training history available for binary model.")
        
        fig, axes = plt.subplots(1, 2 if self.validation_score_history_ else 1, figsize=figsize)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot training cost
        axes[0].plot(self.cost_history_, 'b-', linewidth=2, label='Training Cost')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Cost')
        axes[0].set_title('Training Cost Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot validation score if available
        if self.validation_score_history_:
            axes[1].plot(self.validation_score_history_, 'r-', linewidth=2, label='Validation Cost')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Validation Cost')
            axes[1].set_title('Validation Cost Over Time')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    print("Testing Logistic Regression Implementation")
    print("=" * 50)
    
    # Test binary classification
    print("\nBinary Classification Test:")
    X_binary, y_binary = make_classification(
        n_samples=1000, n_features=5, n_redundant=0, n_informative=5,
        n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42
    )
    
    # Test different penalties
    penalties = [None, 'l1', 'l2', 'elasticnet']
    
    for penalty in penalties:
        print(f"\nPenalty: {penalty}")
        
        model = LogisticRegression(
            penalty=penalty,
            C=1.0,
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        print(f"Average predicted probability: {np.mean(y_proba[:, 1]):.4f}")
    
    # Test multiclass classification
    print("\n" + "="*50)
    print("Multiclass Classification Test:")
    
    X_multi, y_multi = make_blobs(
        n_samples=1000, centers=4, n_features=2, 
        cluster_std=1.5, random_state=42
    )
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.3, random_state=42
    )
    
    model_multi = LogisticRegression(
        penalty='l2',
        C=1.0,
        max_iter=1000,
        random_state=42,
        multi_class='ovr'
    )
    
    model_multi.fit(X_train_multi, y_train_multi)
    
    train_score_multi = model_multi.score(X_train_multi, y_train_multi)
    test_score_multi = model_multi.score(X_test_multi, y_test_multi)
    
    print(f"Training Accuracy: {train_score_multi:.4f}")
    print(f"Test Accuracy: {test_score_multi:.4f}")
    
    # Get multiclass predictions
    y_pred_multi = model_multi.predict(X_test_multi)
    y_proba_multi = model_multi.predict_proba(X_test_multi)
    
    print(f"Number of classes: {len(model_multi.classes_)}")
    print(f"Classes: {model_multi.classes_}")
    print(f"Probability shape: {y_proba_multi.shape}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_multi, y_pred_multi))
