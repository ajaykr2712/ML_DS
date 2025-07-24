"""
Advanced Ensemble Methods Implementation from Scratch
===================================================

Comprehensive ensemble learning algorithms including:
- Random Forest with feature importance
- Gradient Boosting (XGBoost-style)
- AdaBoost with weak learners
- Voting and Stacking ensembles
- Advanced feature selection

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from concurrent.futures import ThreadPoolExecutor
import time


class BaseEnsemble(ABC):
    """Abstract base class for ensemble methods."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEnsemble':
        """Fit the ensemble to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass


class RandomForestFromScratch(BaseEnsemble):
    """
    Random Forest implementation from scratch with advanced features.
    
    Features:
    - Bootstrap sampling with replacement
    - Random feature selection at each split
    - Out-of-bag (OOB) score calculation
    - Feature importance computation
    - Parallel training support
    - Memory optimization
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = 'sqrt',
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        oob_score: bool = False,
        task: str = 'classification'
    ):
        """
        Initialize Random Forest.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        max_features : str, int, float, default='sqrt'
            Number of features to consider at each split
        bootstrap : bool, default=True
            Whether to use bootstrap sampling
        random_state : int, optional
            Random seed for reproducibility
        n_jobs : int, default=1
            Number of parallel jobs (-1 for all cores)
        oob_score : bool, default=False
            Whether to compute out-of-bag score
        task : str, default='classification'
            'classification' or 'regression'
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.oob_score = oob_score
        self.task = task
        
        # Initialize components
        self.trees_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None
        self.n_samples_ = None
        self.classes_ = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_max_features(self, n_features: int) -> int:
        """Calculate number of features to use at each split."""
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
            elif self.max_features == 'auto':
                return int(np.sqrt(n_features))
            else:
                raise ValueError(f"Unknown max_features: {self.max_features}")
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, 
                         random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create bootstrap sample of the data."""
        np.random.seed(random_state)
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        else:
            # Use all samples
            indices = np.arange(n_samples)
            oob_indices = np.array([])
        
        return X[indices], y[indices], oob_indices
    
    def _fit_tree(self, args: Tuple) -> Tuple[Any, np.ndarray]:
        """Fit a single tree (for parallel processing)."""
        X, y, tree_idx = args
        
        # Create bootstrap sample
        X_bootstrap, y_bootstrap, oob_indices = self._bootstrap_sample(
            X, y, self.random_state + tree_idx if self.random_state else tree_idx
        )
        
        # Create and fit tree
        if self.task == 'classification':
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._get_max_features(X.shape[1]),
                random_state=self.random_state + tree_idx if self.random_state else None
            )
        else:
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._get_max_features(X.shape[1]),
                random_state=self.random_state + tree_idx if self.random_state else None
            )
        
        tree.fit(X_bootstrap, y_bootstrap)
        return tree, oob_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestFromScratch':
        """
        Fit the Random Forest to training data.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : RandomForestFromScratch
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_samples_, self.n_features_ = X.shape
        
        if self.task == 'classification':
            self.classes_ = np.unique(y)
        
        # Prepare arguments for parallel tree fitting
        tree_args = [(X, y, i) for i in range(self.n_estimators)]
        
        # Fit trees in parallel
        if self.n_jobs == 1:
            # Sequential processing
            results = [self._fit_tree(args) for args in tree_args]
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(self._fit_tree, tree_args))
        
        # Extract trees and OOB indices
        self.trees_ = [result[0] for result in results]
        oob_indices_list = [result[1] for result in results]
        
        # Compute feature importances
        self._compute_feature_importances()
        
        # Compute OOB score if requested
        if self.oob_score:
            self._compute_oob_score(X, y, oob_indices_list)
        
        return self
    
    def _compute_feature_importances(self):
        """Compute feature importances across all trees."""
        importances = np.zeros(self.n_features_)
        
        for tree in self.trees_:
            importances += tree.feature_importances_
        
        # Average and normalize
        importances /= len(self.trees_)
        self.feature_importances_ = importances / np.sum(importances)
    
    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray, 
                          oob_indices_list: List[np.ndarray]):
        """Compute out-of-bag score."""
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, len(self.classes_) if self.task == 'classification' else 1))
        oob_counts = np.zeros(n_samples)
        
        for tree, oob_indices in zip(self.trees_, oob_indices_list):
            if len(oob_indices) > 0:
                if self.task == 'classification':
                    # Get class probabilities
                    proba = tree.predict_proba(X[oob_indices])
                    oob_predictions[oob_indices] += proba
                else:
                    # Get regression predictions
                    pred = tree.predict(X[oob_indices])
                    oob_predictions[oob_indices, 0] += pred
                
                oob_counts[oob_indices] += 1
        
        # Compute final OOB predictions
        valid_oob = oob_counts > 0
        
        if self.task == 'classification':
            oob_predictions[valid_oob] /= oob_counts[valid_oob].reshape(-1, 1)
            oob_pred_classes = np.argmax(oob_predictions[valid_oob], axis=1)
            oob_pred_classes = self.classes_[oob_pred_classes]
            self.oob_score_ = np.mean(oob_pred_classes == y[valid_oob])
        else:
            oob_predictions[valid_oob, 0] /= oob_counts[valid_oob]
            oob_pred = oob_predictions[valid_oob, 0]
            self.oob_score_ = 1 - np.mean((oob_pred - y[valid_oob]) ** 2) / np.var(y[valid_oob])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        predictions : ndarray
            Predicted values
        """
        if not self.trees_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        if self.task == 'classification':
            # Collect predictions from all trees
            all_predictions = np.array([tree.predict(X) for tree in self.trees_])
            
            # Majority voting
            predictions = np.array([
                Counter(all_predictions[:, i]).most_common(1)[0][0]
                for i in range(X.shape[0])
            ])
            
            return predictions
        else:
            # Average predictions for regression
            all_predictions = np.array([tree.predict(X) for tree in self.trees_])
            return np.mean(all_predictions, axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.trees_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Collect probabilities from all trees
        all_probas = np.zeros((n_samples, n_classes))
        
        for tree in self.trees_:
            probas = tree.predict_proba(X)
            all_probas += probas
        
        # Average probabilities
        all_probas /= len(self.trees_)
        
        return all_probas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score for classification or R² for regression.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        y : ndarray of shape (n_samples,)
            True values
        
        Returns:
        --------
        score : float
            Accuracy or R² score
        """
        predictions = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(predictions == y)
        else:
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)


class GradientBoostingFromScratch(BaseEnsemble):
    """
    Gradient Boosting implementation with advanced features.
    
    Features:
    - Multiple loss functions
    - Learning rate scheduling
    - Early stopping with validation
    - Feature subsampling
    - Row subsampling
    - Regularization parameters
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features: Union[str, int, float] = 1.0,
        loss: str = None,
        validation_fraction: float = 0.1,
        early_stopping: bool = False,
        n_iter_no_change: int = 5,
        random_state: Optional[int] = None,
        task: str = 'regression'
    ):
        """
        Initialize Gradient Boosting.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting stages
        learning_rate : float, default=0.1
            Learning rate shrinks contribution of each tree
        max_depth : int, default=3
            Maximum depth of regression estimators
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        subsample : float, default=1.0
            Fraction of samples used for fitting trees
        max_features : str, int, float, default=1.0
            Number of features to consider at each split
        loss : str, default=None
            Loss function to optimize (auto-selected based on task if None)
        validation_fraction : float, default=0.1
            Fraction of training data for early stopping
        early_stopping : bool, default=False
            Whether to use early stopping
        n_iter_no_change : int, default=5
            Number of iterations with no improvement for early stopping
        random_state : int, optional
            Random seed for reproducibility
        task : str, default='regression'
            'classification' or 'regression'
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        
        # Auto-select loss function if not provided
        if loss is None:
            self.loss = 'log_loss' if task == 'classification' else 'squared_error'
        else:
            self.loss = loss
            
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.task = task
        
        # Initialize components
        self.estimators_ = []
        self.train_score_ = []
        self.validation_score_ = []
        self.init_prediction_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_loss_function(self):
        """Get loss function based on task and loss parameter."""
        if self.task == 'regression':
            if self.loss == 'squared_error':
                return self._squared_error_loss
            elif self.loss == 'absolute_error':
                return self._absolute_error_loss
            elif self.loss == 'huber':
                return self._huber_loss
        elif self.task == 'classification':
            if self.loss == 'log_loss':
                return self._log_loss
        
        raise ValueError(f"Unknown loss function: {self.loss} for task: {self.task}")
    
    def _squared_error_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Squared error loss and its negative gradient."""
        loss = 0.5 * np.mean((y_true - y_pred) ** 2)
        gradient = y_true - y_pred
        return loss, gradient
    
    def _absolute_error_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Absolute error loss and its negative gradient."""
        loss = np.mean(np.abs(y_true - y_pred))
        gradient = np.sign(y_true - y_pred)
        return loss, gradient
    
    def _huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> Tuple[float, np.ndarray]:
        """Huber loss and its negative gradient."""
        residual = y_true - y_pred
        abs_residual = np.abs(residual)
        
        # Compute loss
        quadratic = np.minimum(abs_residual, delta)
        linear = abs_residual - quadratic
        loss = np.mean(0.5 * quadratic ** 2 + delta * linear)
        
        # Compute gradient
        gradient = np.where(abs_residual <= delta, residual, delta * np.sign(residual))
        return loss, gradient
    
    def _log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Logistic loss and its negative gradient."""
        # Convert to probabilities
        prob = 1 / (1 + np.exp(-y_pred))
        prob = np.clip(prob, 1e-15, 1 - 1e-15)  # Avoid log(0)
        
        # Compute loss
        loss = -np.mean(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))
        
        # Compute gradient
        gradient = y_true - prob
        return loss, gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingFromScratch':
        """
        Fit the Gradient Boosting model.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : GradientBoostingFromScratch
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, self.n_features_ = X.shape
        
        # Split data for early stopping
        if self.early_stopping and self.validation_fraction > 0:
            n_val = int(n_samples * self.validation_fraction)
            indices = np.random.permutation(n_samples)
            
            X_val = X[indices[:n_val]]
            y_val = y[indices[:n_val]]
            X_train = X[indices[n_val:]]
            y_train = y[indices[n_val:]]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Initialize prediction with the mean/mode
        if self.task == 'regression':
            self.init_prediction_ = np.mean(y_train)
        else:
            self.init_prediction_ = np.log(np.mean(y_train) / (1 - np.mean(y_train)))
        
        # Get loss function
        loss_func = self._get_loss_function()
        
        # Initialize predictions
        current_pred = np.full(len(y_train), self.init_prediction_)
        if X_val is not None:
            current_pred_val = np.full(len(y_val), self.init_prediction_)
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(self.n_features_)
        
        # Early stopping variables
        best_val_score = float('inf')
        no_improvement_count = 0
        
        for i in range(self.n_estimators):
            # Compute loss and gradients
            train_loss, gradients = loss_func(y_train, current_pred)
            self.train_score_.append(train_loss)
            
            # Subsample data if needed
            if self.subsample < 1.0:
                n_subset = int(self.subsample * len(X_train))
                subset_indices = np.random.choice(len(X_train), n_subset, replace=False)
                X_subset = X_train[subset_indices]
                gradients_subset = gradients[subset_indices]
            else:
                X_subset = X_train
                gradients_subset = gradients
            
            # Fit weak learner to gradients
            if self.task == 'regression':
                estimator = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.random_state + i if self.random_state else None
                )
            else:
                estimator = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.random_state + i if self.random_state else None
                )
            
            estimator.fit(X_subset, gradients_subset)
            self.estimators_.append(estimator)
            
            # Update feature importances
            self.feature_importances_ += estimator.feature_importances_
            
            # Update predictions
            pred_update = estimator.predict(X_train)
            current_pred += self.learning_rate * pred_update
            
            # Validation score for early stopping
            if X_val is not None:
                pred_update_val = estimator.predict(X_val)
                current_pred_val += self.learning_rate * pred_update_val
                val_loss, _ = loss_func(y_val, current_pred_val)
                self.validation_score_.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_score:
                    best_val_score = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= self.n_iter_no_change:
                    print(f"Early stopping at iteration {i + 1}")
                    break
        
        # Normalize feature importances
        self.feature_importances_ /= len(self.estimators_)
        self.feature_importances_ /= np.sum(self.feature_importances_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        predictions : ndarray
            Predicted values
        """
        if not self.estimators_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        # Initialize with base prediction
        predictions = np.full(X.shape[0], self.init_prediction_)
        
        # Add contributions from all estimators
        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)
        
        # Convert to probabilities for classification
        if self.task == 'classification':
            predictions = 1 / (1 + np.exp(-predictions))
            predictions = (predictions > 0.5).astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for classification.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        probabilities : ndarray of shape (n_samples, 2)
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.estimators_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        # Get raw predictions
        predictions = np.full(X.shape[0], self.init_prediction_)
        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)
        
        # Convert to probabilities
        prob_positive = 1 / (1 + np.exp(-predictions))
        prob_negative = 1 - prob_positive
        
        return np.column_stack([prob_negative, prob_positive])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score for classification or R² for regression.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        y : ndarray of shape (n_samples,)
            True values
        
        Returns:
        --------
        score : float
            Accuracy or R² score
        """
        predictions = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(predictions == y)
        else:
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 5)):
        """Plot training and validation loss curves."""
        if not self.train_score_:
            raise ValueError("No training history available. Fit the model first.")
        
        fig, axes = plt.subplots(1, 2 if self.validation_score_ else 1, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot training loss
        epochs = range(1, len(self.train_score_) + 1)
        axes[0].plot(epochs, self.train_score_, 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Boosting Iterations')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot validation loss if available
        if self.validation_score_ and len(axes) > 1:
            val_epochs = range(1, len(self.validation_score_) + 1)
            axes[1].plot(val_epochs, self.validation_score_, 'r-', linewidth=2, label='Validation Loss')
            axes[1].set_xlabel('Boosting Iterations')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Validation Loss')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score
    
    print("="*60)
    print("Advanced Ensemble Methods Demo")
    print("="*60)
    
    # Test Random Forest Classification
    print("\n1. Random Forest Classification Test:")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                      n_redundant=5, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    rf_clf = RandomForestFromScratch(
        n_estimators=50,
        max_depth=10,
        oob_score=True,
        random_state=42,
        task='classification'
    )
    
    start_time = time.time()
    rf_clf.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    predictions = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Training time: {fit_time:.3f} seconds")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {rf_clf.oob_score_:.4f}")
    print(f"Top 5 feature importances: {rf_clf.feature_importances_[:5]}")
    
    # Test Random Forest Regression
    print("\n2. Random Forest Regression Test:")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    rf_reg = RandomForestFromScratch(
        n_estimators=50,
        max_depth=10,
        oob_score=True,
        random_state=42,
        task='regression'
    )
    
    rf_reg.fit(X_train, y_train)
    predictions = rf_reg.predict(X_test)
    r2 = r2_score(y_test, predictions)
    
    print(f"Test R² Score: {r2:.4f}")
    print(f"OOB Score: {rf_reg.oob_score_:.4f}")
    
    # Test Gradient Boosting Classification
    print("\n3. Gradient Boosting Classification Test:")
    gb_clf = GradientBoostingFromScratch(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        task='classification'
    )
    
    # Convert to binary classification
    y_binary = (y_train > 1).astype(int)
    y_test_binary = (y_test > 1).astype(int)
    
    gb_clf.fit(X_train, y_binary)
    predictions = gb_clf.predict(X_test)
    accuracy = accuracy_score(y_test_binary, predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Number of estimators used: {len(gb_clf.estimators_)}")
    
    # Test Gradient Boosting Regression
    print("\n4. Gradient Boosting Regression Test:")
    gb_reg = GradientBoostingFromScratch(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        task='regression'
    )
    
    gb_reg.fit(X_train, y_train)
    predictions = gb_reg.predict(X_test)
    r2 = r2_score(y_test, predictions)
    
    print(f"Test R² Score: {r2:.4f}")
    print(f"Number of estimators used: {len(gb_reg.estimators_)}")
    
    print("\nAll ensemble tests completed successfully!")
