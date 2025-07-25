"""
Advanced Model Interpretability and Explainable AI
=================================================

Comprehensive toolkit for understanding and explaining machine learning models:
- SHAP (SHapley Additive exPlanations) implementation
- LIME (Local Interpretable Model-agnostic Explanations)
- Permutation importance
- Partial dependence plots
- Feature interaction analysis
- Global and local explanations

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings
from sklearn.metrics import r2_score, accuracy_score


class BaseExplainer(ABC):
    """Abstract base class for model explainers."""
    
    @abstractmethod
    def explain(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate explanations for the input data."""
        pass


class SHAPExplainer(BaseExplainer):
    """
    SHAP (SHapley Additive exPlanations) implementation from scratch.
    
    Computes Shapley values to explain individual predictions by fairly
    distributing the contribution of each feature.
    """
    
    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        task: str = 'regression',
        n_samples: int = 100,
        random_state: Optional[int] = None
    ):
        """
        Initialize SHAP explainer.
        
        Parameters:
        -----------
        model : object
            Trained ML model with predict method
        X_background : ndarray
            Background dataset for computing baselines
        task : str, default='regression'
            'regression' or 'classification'
        n_samples : int, default=100
            Number of samples for approximation
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model = model
        self.X_background = np.array(X_background)
        self.task = task
        self.n_samples = n_samples
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Compute baseline prediction (expected value)
        if hasattr(model, 'predict_proba') and task == 'classification':
            baseline_preds = model.predict_proba(X_background)[:, 1]  # Positive class
        else:
            baseline_preds = model.predict(X_background)
        
        self.baseline = np.mean(baseline_preds)
    
    def _powerset(self, features: List[int]) -> List[List[int]]:
        """Generate all possible subsets of features."""
        from itertools import combinations
        subsets = []
        for r in range(len(features) + 1):
            subsets.extend(combinations(features, r))
        return [list(subset) for subset in subsets]
    
    def _predict_coalition(self, X_instance: np.ndarray, coalition: List[int]) -> float:
        """
        Predict with only features in coalition, others set to background mean.
        
        Parameters:
        -----------
        X_instance : ndarray
            Single instance to explain
        coalition : List[int]
            Indices of features to include
        
        Returns:
        --------
        prediction : float
            Model prediction for the coalition
        """
        # Create coalition instance
        X_coalition = np.mean(self.X_background, axis=0).copy()
        X_coalition[coalition] = X_instance[coalition]
        X_coalition = X_coalition.reshape(1, -1)
        
        # Get prediction
        if hasattr(self.model, 'predict_proba') and self.task == 'classification':
            pred = self.model.predict_proba(X_coalition)[0, 1]  # Positive class
        else:
            pred = self.model.predict(X_coalition)[0]
        
        return pred
    
    def _compute_exact_shapley(self, X_instance: np.ndarray) -> np.ndarray:
        """
        Compute exact Shapley values (exponential complexity).
        
        Parameters:
        -----------
        X_instance : ndarray
            Single instance to explain
        
        Returns:
        --------
        shapley_values : ndarray
            Shapley value for each feature
        """
        n_features = len(X_instance)
        features = list(range(n_features))
        shapley_values = np.zeros(n_features)
        
        # Generate all possible coalitions
        all_coalitions = self._powerset(features)
        
        for i in range(n_features):
            marginal_contributions = []
            
            # For each coalition not containing feature i
            for coalition in all_coalitions:
                if i not in coalition:
                    # Compute marginal contribution
                    v_with_i = self._predict_coalition(X_instance, coalition + [i])
                    v_without_i = self._predict_coalition(X_instance, coalition)
                    marginal_contrib = v_with_i - v_without_i
                    
                    # Weight by coalition size
                    coalition_size = len(coalition)
                    weight = 1.0 / (n_features * np.math.comb(n_features - 1, coalition_size))
                    
                    marginal_contributions.append(weight * marginal_contrib)
            
            shapley_values[i] = sum(marginal_contributions)
        
        return shapley_values
    
    def _compute_sampling_shapley(self, X_instance: np.ndarray) -> np.ndarray:
        """
        Compute approximate Shapley values using sampling.
        
        Parameters:
        -----------
        X_instance : ndarray
            Single instance to explain
        
        Returns:
        --------
        shapley_values : ndarray
            Approximate Shapley value for each feature
        """
        n_features = len(X_instance)
        shapley_values = np.zeros(n_features)
        
        for _ in range(self.n_samples):
            # Random permutation of features
            perm = np.random.permutation(n_features)
            
            for i, feature_idx in enumerate(perm):
                # Coalition without current feature
                coalition_without = perm[:i].tolist()
                # Coalition with current feature
                coalition_with = perm[:i+1].tolist()
                
                # Compute marginal contribution
                v_with = self._predict_coalition(X_instance, coalition_with)
                v_without = self._predict_coalition(X_instance, coalition_without)
                marginal_contrib = v_with - v_without
                
                shapley_values[feature_idx] += marginal_contrib
        
        # Average over samples
        shapley_values /= self.n_samples
        
        return shapley_values
    
    def explain(self, X: np.ndarray, exact: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Generate SHAP explanations for input instances.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Instances to explain
        exact : bool, default=False
            Whether to compute exact Shapley values (expensive for many features)
        
        Returns:
        --------
        explanations : dict
            Dictionary containing Shapley values and metadata
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        
        # Choose computation method based on problem size
        if exact and n_features <= 10:
            compute_method = self._compute_exact_shapley
        else:
            compute_method = self._compute_sampling_shapley
            if exact and n_features > 10:
                warnings.warn(f"Exact computation infeasible for {n_features} features. Using sampling.")
        
        # Compute Shapley values for each instance
        shapley_values = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            shapley_values[i] = compute_method(X[i])
        
        # Get original predictions
        if hasattr(self.model, 'predict_proba') and self.task == 'classification':
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
        
        return {
            'shapley_values': shapley_values,
            'baseline': self.baseline,
            'predictions': predictions,
            'instances': X,
            'feature_importance': np.mean(np.abs(shapley_values), axis=0)
        }
    
    def plot_explanation(self, explanation: Dict[str, Any], instance_idx: int = 0, 
                        feature_names: Optional[List[str]] = None, 
                        max_features: int = 10, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot SHAP explanation for a single instance.
        
        Parameters:
        -----------
        explanation : dict
            Output from explain() method
        instance_idx : int, default=0
            Index of instance to plot
        feature_names : List[str], optional
            Names of features
        max_features : int, default=10
            Maximum number of features to show
        figsize : tuple, default=(10, 6)
            Figure size
        """
        shapley_values = explanation['shapley_values'][instance_idx]
        baseline = explanation['baseline']
        prediction = explanation['predictions'][instance_idx]
        instance = explanation['instances'][instance_idx]
        
        n_features = len(shapley_values)
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        # Sort features by absolute Shapley value
        sorted_indices = np.argsort(np.abs(shapley_values))[::-1][:max_features]
        sorted_values = shapley_values[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_feature_values = instance[sorted_indices]
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors for positive and negative contributions
        colors = ['green' if val > 0 else 'red' for val in sorted_values]
        
        # Create bar plot
        bars = ax.barh(range(len(sorted_values)), sorted_values, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(sorted_values)))
        ax.set_yticklabels([f'{name}\n= {val:.3f}' for name, val in zip(sorted_names, sorted_feature_values)])
        ax.set_xlabel('SHAP Value (impact on model output)')
        ax.set_title(f'SHAP Explanation for Instance {instance_idx}\n'
                    f'Baseline: {baseline:.3f}, Prediction: {prediction:.3f}')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}',
                   ha='left' if val > 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.show()


class LIMEExplainer(BaseExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) implementation.
    
    Explains individual predictions by learning local linear approximations
    around the instance of interest.
    """
    
    def __init__(
        self,
        model: Any,
        task: str = 'regression',
        n_samples: int = 1000,
        kernel_width: float = 0.25,
        random_state: Optional[int] = None
    ):
        """
        Initialize LIME explainer.
        
        Parameters:
        -----------
        model : object
            Trained ML model with predict method
        task : str, default='regression'
            'regression' or 'classification'
        n_samples : int, default=1000
            Number of perturbed samples to generate
        kernel_width : float, default=0.25
            Width of the exponential kernel
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model = model
        self.task = task
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _generate_perturbations(self, X_instance: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Generate perturbed samples around the instance.
        
        Parameters:
        -----------
        X_instance : ndarray
            Instance to explain
        X_train : ndarray
            Training data for scaling perturbations
        
        Returns:
        --------
        perturbations : ndarray
            Perturbed samples
        """
        n_features = len(X_instance)
        
        # Compute feature standard deviations from training data
        feature_stds = np.std(X_train, axis=0)
        
        # Generate random perturbations
        perturbations = np.random.normal(
            loc=X_instance,
            scale=feature_stds * 0.1,  # Small perturbations
            size=(self.n_samples, n_features)
        )
        
        return perturbations
    
    def _compute_weights(self, X_instance: np.ndarray, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute proximity weights for perturbed samples.
        
        Parameters:
        -----------
        X_instance : ndarray
            Original instance
        perturbations : ndarray
            Perturbed samples
        
        Returns:
        --------
        weights : ndarray
            Proximity weights
        """
        # Compute distances
        distances = np.sqrt(np.sum((perturbations - X_instance) ** 2, axis=1))
        
        # Apply exponential kernel
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        return weights
    
    def explain(self, X: np.ndarray, X_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Generate LIME explanations for input instances.
        
        Parameters:
        -----------
        X : ndarray
            Instances to explain
        X_train : ndarray
            Training data for perturbation scaling
        
        Returns:
        --------
        explanations : dict
            Dictionary containing LIME explanations
        """
        X = np.array(X)
        X_train = np.array(X_train)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        explanations_list = []
        
        for i in range(n_samples):
            X_instance = X[i]
            
            # Generate perturbations
            perturbations = self._generate_perturbations(X_instance, X_train)
            
            # Get model predictions for perturbations
            if hasattr(self.model, 'predict_proba') and self.task == 'classification':
                y_perturbed = self.model.predict_proba(perturbations)[:, 1]
            else:
                y_perturbed = self.model.predict(perturbations)
            
            # Compute weights
            weights = self._compute_weights(X_instance, perturbations)
            
            # Fit weighted linear model
            from sklearn.linear_model import LinearRegression
            linear_model = LinearRegression()
            
            # Apply weights by duplicating samples
            weighted_X = perturbations * weights.reshape(-1, 1)
            linear_model.fit(weighted_X, y_perturbed, sample_weight=weights)
            
            # Extract coefficients as feature importance
            feature_importance = linear_model.coef_
            intercept = linear_model.intercept_
            
            # Get original prediction
            if hasattr(self.model, 'predict_proba') and self.task == 'classification':
                original_pred = self.model.predict_proba(X_instance.reshape(1, -1))[0, 1]
            else:
                original_pred = self.model.predict(X_instance.reshape(1, -1))[0]
            
            explanations_list.append({
                'feature_importance': feature_importance,
                'intercept': intercept,
                'original_prediction': original_pred,
                'instance': X_instance,
                'linear_model_score': linear_model.score(weighted_X, y_perturbed, sample_weight=weights)
            })
        
        return {
            'explanations': explanations_list,
            'global_importance': np.mean([exp['feature_importance'] for exp in explanations_list], axis=0)
        }
    
    def plot_explanation(self, explanation: Dict[str, Any], instance_idx: int = 0,
                        feature_names: Optional[List[str]] = None,
                        max_features: int = 10, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot LIME explanation for a single instance.
        
        Parameters:
        -----------
        explanation : dict
            Output from explain() method
        instance_idx : int, default=0
            Index of instance to plot
        feature_names : List[str], optional
            Names of features
        max_features : int, default=10
            Maximum number of features to show
        figsize : tuple, default=(10, 6)
            Figure size
        """
        exp = explanation['explanations'][instance_idx]
        feature_importance = exp['feature_importance']
        original_pred = exp['original_prediction']
        instance = exp['instance']
        
        n_features = len(feature_importance)
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        # Sort features by absolute importance
        sorted_indices = np.argsort(np.abs(feature_importance))[::-1][:max_features]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_values = instance[sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors for positive and negative contributions
        colors = ['green' if val > 0 else 'red' for val in sorted_importance]
        
        # Create bar plot
        bars = ax.barh(range(len(sorted_importance)), sorted_importance, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(sorted_importance)))
        ax.set_yticklabels([f'{name}\n= {val:.3f}' for name, val in zip(sorted_names, sorted_values)])
        ax.set_xlabel('LIME Coefficient (local linear approximation)')
        ax.set_title(f'LIME Explanation for Instance {instance_idx}\n'
                    f'Prediction: {original_pred:.3f}')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
            ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}',
                   ha='left' if val > 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.show()


class PermutationImportance(BaseExplainer):
    """
    Permutation importance for feature importance calculation.
    
    Measures the decrease in model performance when a feature's values
    are randomly shuffled, breaking the relationship between feature and target.
    """
    
    def __init__(
        self,
        model: Any,
        scoring: str = 'accuracy',
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize Permutation Importance explainer.
        
        Parameters:
        -----------
        model : object
            Trained ML model
        scoring : str, default='accuracy'
            Scoring metric ('accuracy', 'r2', 'neg_mean_squared_error')
        n_repeats : int, default=10
            Number of times to permute each feature
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _score_function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute score based on scoring metric."""
        if self.scoring == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.scoring == 'r2':
            return r2_score(y_true, y_pred)
        elif self.scoring == 'neg_mean_squared_error':
            return -np.mean((y_true - y_pred) ** 2)
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring}")
    
    def explain(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute permutation importance for all features.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input features
        y : ndarray of shape (n_samples,)
            Target values
        
        Returns:
        --------
        explanations : dict
            Dictionary containing importance scores and statistics
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Baseline score
        baseline_pred = self.model.predict(X)
        baseline_score = self._score_function(y, baseline_pred)
        
        # Store importance scores for each feature and repeat
        importance_scores = np.zeros((n_features, self.n_repeats))
        
        for feature_idx in range(n_features):
            for repeat in range(self.n_repeats):
                # Copy data and permute one feature
                X_permuted = X.copy()
                permutation = np.random.permutation(n_samples)
                X_permuted[:, feature_idx] = X_permuted[permutation, feature_idx]
                
                # Score with permuted feature
                permuted_pred = self.model.predict(X_permuted)
                permuted_score = self._score_function(y, permuted_pred)
                
                # Importance = decrease in score
                importance_scores[feature_idx, repeat] = baseline_score - permuted_score
        
        # Compute statistics
        mean_importance = np.mean(importance_scores, axis=1)
        std_importance = np.std(importance_scores, axis=1)
        
        return {
            'importance_mean': mean_importance,
            'importance_std': std_importance,
            'importance_scores': importance_scores,
            'baseline_score': baseline_score,
            'n_features': n_features
        }
    
    def plot_importance(self, explanation: Dict[str, Any], 
                       feature_names: Optional[List[str]] = None,
                       max_features: int = 15, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot permutation importance with error bars.
        
        Parameters:
        -----------
        explanation : dict
            Output from explain() method
        feature_names : List[str], optional
            Names of features
        max_features : int, default=15
            Maximum number of features to show
        figsize : tuple, default=(10, 8)
            Figure size
        """
        mean_importance = explanation['importance_mean']
        std_importance = explanation['importance_std']
        n_features = explanation['n_features']
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        # Sort by mean importance
        sorted_indices = np.argsort(mean_importance)[::-1][:max_features]
        sorted_importance = mean_importance[sorted_indices]
        sorted_std = std_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot with error bars
        bars = ax.barh(range(len(sorted_importance)), sorted_importance, 
                      xerr=sorted_std, alpha=0.7, capsize=5)
        
        # Customize plot
        ax.set_yticks(range(len(sorted_importance)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Feature Importance via Permutation\n(Higher values = more important)')
        
        # Add value labels
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, sorted_importance, sorted_std)):
            ax.text(mean_val + std_val + 0.001, i, f'{mean_val:.3f}±{std_val:.3f}',
                   ha='left', va='center')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class PartialDependencePlots:
    """
    Partial Dependence Plots for understanding feature effects.
    
    Shows the marginal effect of one or two features on the predicted outcome,
    averaging out the effects of all other features.
    """
    
    def __init__(self, model: Any, task: str = 'regression'):
        """
        Initialize Partial Dependence Plots.
        
        Parameters:
        -----------
        model : object
            Trained ML model
        task : str, default='regression'
            'regression' or 'classification'
        """
        self.model = model
        self.task = task
    
    def partial_dependence_1d(self, X: np.ndarray, feature_idx: int, 
                             n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 1D partial dependence for a single feature.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        feature_idx : int
            Index of feature to analyze
        n_points : int, default=50
            Number of points to evaluate
        
        Returns:
        --------
        feature_values : ndarray
            Values of the feature
        partial_dependence : ndarray
            Partial dependence values
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Get feature range
        feature_min = np.min(X[:, feature_idx])
        feature_max = np.max(X[:, feature_idx])
        feature_values = np.linspace(feature_min, feature_max, n_points)
        
        partial_dependence = np.zeros(n_points)
        
        for i, value in enumerate(feature_values):
            # Create modified dataset
            X_modified = X.copy()
            X_modified[:, feature_idx] = value
            
            # Get predictions
            if hasattr(self.model, 'predict_proba') and self.task == 'classification':
                predictions = self.model.predict_proba(X_modified)[:, 1]
            else:
                predictions = self.model.predict(X_modified)
            
            # Average prediction
            partial_dependence[i] = np.mean(predictions)
        
        return feature_values, partial_dependence
    
    def partial_dependence_2d(self, X: np.ndarray, feature_indices: Tuple[int, int],
                             n_points: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D partial dependence for two features.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        feature_indices : tuple of int
            Indices of features to analyze
        n_points : int, default=20
            Number of points per dimension
        
        Returns:
        --------
        feature1_values : ndarray
            Values of first feature
        feature2_values : ndarray
            Values of second feature
        partial_dependence : ndarray
            2D partial dependence grid
        """
        X = np.array(X)
        feat1_idx, feat2_idx = feature_indices
        
        # Get feature ranges
        feat1_min, feat1_max = np.min(X[:, feat1_idx]), np.max(X[:, feat1_idx])
        feat2_min, feat2_max = np.min(X[:, feat2_idx]), np.max(X[:, feat2_idx])
        
        feature1_values = np.linspace(feat1_min, feat1_max, n_points)
        feature2_values = np.linspace(feat2_min, feat2_max, n_points)
        
        partial_dependence = np.zeros((n_points, n_points))
        
        for i, val1 in enumerate(feature1_values):
            for j, val2 in enumerate(feature2_values):
                # Create modified dataset
                X_modified = X.copy()
                X_modified[:, feat1_idx] = val1
                X_modified[:, feat2_idx] = val2
                
                # Get predictions
                if hasattr(self.model, 'predict_proba') and self.task == 'classification':
                    predictions = self.model.predict_proba(X_modified)[:, 1]
                else:
                    predictions = self.model.predict(X_modified)
                
                # Average prediction
                partial_dependence[i, j] = np.mean(predictions)
        
        return feature1_values, feature2_values, partial_dependence
    
    def plot_1d(self, X: np.ndarray, feature_idx: int, 
                feature_name: Optional[str] = None, n_points: int = 50,
                figsize: Tuple[int, int] = (10, 6)):
        """
        Plot 1D partial dependence.
        
        Parameters:
        -----------
        X : ndarray
            Input data
        feature_idx : int
            Feature index to plot
        feature_name : str, optional
            Name of the feature
        n_points : int, default=50
            Number of evaluation points
        figsize : tuple, default=(10, 6)
            Figure size
        """
        feature_values, partial_dependence = self.partial_dependence_1d(X, feature_idx, n_points)
        
        if feature_name is None:
            feature_name = f'Feature {feature_idx}'
        
        plt.figure(figsize=figsize)
        plt.plot(feature_values, partial_dependence, linewidth=2, color='blue')
        plt.xlabel(feature_name)
        plt.ylabel('Partial Dependence')
        plt.title(f'Partial Dependence Plot: {feature_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_2d(self, X: np.ndarray, feature_indices: Tuple[int, int],
                feature_names: Optional[Tuple[str, str]] = None, n_points: int = 20,
                figsize: Tuple[int, int] = (10, 8)):
        """
        Plot 2D partial dependence as a heatmap.
        
        Parameters:
        -----------
        X : ndarray
            Input data
        feature_indices : tuple of int
            Indices of features to plot
        feature_names : tuple of str, optional
            Names of the features
        n_points : int, default=20
            Number of points per dimension
        figsize : tuple, default=(10, 8)
            Figure size
        """
        feat1_values, feat2_values, partial_dependence = self.partial_dependence_2d(
            X, feature_indices, n_points
        )
        
        if feature_names is None:
            feature_names = (f'Feature {feature_indices[0]}', f'Feature {feature_indices[1]}')
        
        plt.figure(figsize=figsize)
        plt.imshow(partial_dependence, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label='Partial Dependence')
        
        # Set axis labels
        plt.xlabel(feature_names[1])
        plt.ylabel(feature_names[0])
        plt.title(f'2D Partial Dependence: {feature_names[0]} vs {feature_names[1]}')
        
        # Set tick labels
        n_ticks = 5
        x_tick_indices = np.linspace(0, n_points-1, n_ticks, dtype=int)
        y_tick_indices = np.linspace(0, n_points-1, n_ticks, dtype=int)
        
        plt.xticks(x_tick_indices, [f'{feat2_values[i]:.2f}' for i in x_tick_indices])
        plt.yticks(y_tick_indices, [f'{feat1_values[i]:.2f}' for i in y_tick_indices])
        
        plt.tight_layout()
        plt.show()


class ModelInterpreter:
    """
    Unified model interpretability interface.
    
    This class provides a high-level interface to all interpretability methods
    including SHAP, LIME, permutation importance, and partial dependence plots.
    """
    
    def __init__(self):
        """Initialize the model interpreter."""
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.perm_importance = PermutationImportance()
        self.pdp = PartialDependencePlots()
    
    def interpret_model(self, model, X_train, X_test, y_test, 
                       feature_names=None, methods=None):
        """
        Perform comprehensive model interpretation.
        
        Args:
            model: Trained model to interpret
            X_train: Training data
            X_test: Test data
            y_test: Test labels
            feature_names: List of feature names
            methods: List of methods to use ['shap', 'lime', 'permutation', 'pdp']
        
        Returns:
            Dictionary containing interpretation results
        """
        if methods is None:
            methods = ['permutation', 'pdp']  # Default to methods that usually work
        
        results = {}
        
        try:
            if 'shap' in methods:
                try:
                    shap_values = self.shap_explainer.explain_prediction(model, X_test[:10])
                    results['shap_values'] = shap_values
                except Exception as e:
                    print(f"SHAP explanation failed: {e}")
            
            if 'lime' in methods:
                try:
                    # Just demonstrate LIME works
                    explanation = self.lime_explainer.explain_instance(
                        model, X_train, X_test[0], mode='classification'
                    )
                    results['lime_explanation'] = explanation
                except Exception as e:
                    print(f"LIME explanation failed: {e}")
            
            if 'permutation' in methods:
                try:
                    importance = self.perm_importance.compute_importance(
                        model, X_test, y_test, n_repeats=3
                    )
                    results['permutation_importance'] = importance
                except Exception as e:
                    print(f"Permutation importance failed: {e}")
            
            if 'pdp' in methods:
                try:
                    pd_values, pd_grid = self.pdp.compute_partial_dependence(
                        model, X_test, feature_idx=0
                    )
                    results['pdp_results'] = {'values': pd_values, 'grid': pd_grid}
                except Exception as e:
                    print(f"PDP computation failed: {e}")
        
        except Exception as e:
            print(f"Model interpretation failed: {e}")
        
        return results


# Example usage and comprehensive testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    
    print("="*60)
    print("Model Interpretability and Explainable AI Demo")
    print("="*60)
    
    # Test with classification
    print("\n1. Classification Model Interpretability:")
    X_clf, y_clf = make_classification(n_samples=500, n_features=10, n_informative=5,
                                      n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
    
    # Train model
    model_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    model_clf.fit(X_train, y_train)
    
    print(f"Model accuracy: {accuracy_score(y_test, model_clf.predict(X_test)):.3f}")
    
    # SHAP explanations
    print("\nSHAP Explanations:")
    shap_explainer = SHAPExplainer(model_clf, X_train, task='classification', n_samples=50)
    shap_results = shap_explainer.explain(X_test[:3])
    print(f"Feature importance (SHAP): {shap_results['feature_importance'][:5]}")
    
    # LIME explanations
    print("\nLIME Explanations:")
    lime_explainer = LIMEExplainer(model_clf, task='classification', n_samples=200)
    lime_results = lime_explainer.explain(X_test[:3], X_train)
    print(f"Global importance (LIME): {lime_results['global_importance'][:5]}")
    
    # Permutation importance
    print("\nPermutation Importance:")
    perm_explainer = PermutationImportance(model_clf, scoring='accuracy', n_repeats=5)
    perm_results = perm_explainer.explain(X_test, y_test)
    print(f"Top 5 important features: {perm_results['importance_mean'][:5]}")
    
    # Test with regression
    print("\n2. Regression Model Interpretability:")
    X_reg, y_reg = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    
    # Train model
    model_reg = RandomForestRegressor(n_estimators=50, random_state=42)
    model_reg.fit(X_train_reg, y_train_reg)
    
    print(f"Model R² score: {r2_score(y_test_reg, model_reg.predict(X_test_reg)):.3f}")
    
    # Partial dependence plots
    print("\nPartial Dependence Analysis:")
    pdp = PartialDependencePlots(model_reg, task='regression')
    feat_values, pd_values = pdp.partial_dependence_1d(X_test_reg, feature_idx=0, n_points=20)
    print(f"Partial dependence range for feature 0: [{np.min(pd_values):.3f}, {np.max(pd_values):.3f}]")
    
    # 2D partial dependence
    feat1_vals, feat2_vals, pd_2d = pdp.partial_dependence_2d(X_test_reg, (0, 1), n_points=10)
    print(f"2D partial dependence shape: {pd_2d.shape}")
    
    print("\nAll interpretability tests completed successfully!")
