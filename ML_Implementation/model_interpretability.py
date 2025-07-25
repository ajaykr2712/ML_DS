"""
Enhanced Model Interpretability Suite
====================================

Comprehensive tools for explaining machine learning models including:
- SHAP (SHapley Additive exPlanations) implementation
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Permutation importance
- Partial dependence plots

Author: ML Interpretability Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import itertools

class SHAPExplainer:
    """SHAP-based model explainer."""
    
    def __init__(self, model: BaseEstimator, background_data: np.ndarray):
        self.model = model
        self.background_data = background_data
        self.baseline_value = self._compute_baseline()
    
    def _compute_baseline(self) -> float:
        """Compute baseline prediction on background data."""
        if hasattr(self.model, 'predict_proba'):
            baseline_preds = self.model.predict_proba(self.background_data)
            return np.mean(baseline_preds[:, 1])  # For binary classification
        else:
            baseline_preds = self.model.predict(self.background_data)
            return np.mean(baseline_preds)
    
    def explain_instance(self, instance: np.ndarray, num_features: Optional[int] = None) -> Dict:
        """Explain a single prediction using SHAP values."""
        if num_features is None:
            num_features = len(instance)
        
        shap_values = self._compute_shap_values(instance, num_features)
        
        return {
            'shap_values': shap_values,
            'baseline_value': self.baseline_value,
            'instance_prediction': self._predict_instance(instance),
            'feature_contributions': dict(enumerate(shap_values))
        }
    
    def _compute_shap_values(self, instance: np.ndarray, num_features: int) -> np.ndarray:
        """Compute SHAP values using sampling approach."""
        shap_values = np.zeros(num_features)
        
        # Sample coalitions to estimate Shapley values
        n_samples = 100
        
        for i in range(num_features):
            marginal_contributions = []
            
            for _ in range(n_samples):
                # Random coalition without feature i
                coalition = np.random.choice(num_features, 
                                           size=np.random.randint(0, num_features),
                                           replace=False)
                coalition = coalition[coalition != i]
                
                # Create masked instances
                instance_without = instance.copy()
                instance_with = instance.copy()
                
                # Mask features not in coalition
                for j in range(num_features):
                    if j not in coalition and j != i:
                        # Use background average
                        bg_value = np.mean(self.background_data[:, j])
                        instance_without[j] = bg_value
                        instance_with[j] = bg_value
                
                # Mask feature i in 'without' version
                instance_without[i] = np.mean(self.background_data[:, i])
                
                # Compute marginal contribution
                pred_with = self._predict_instance(instance_with)
                pred_without = self._predict_instance(instance_without)
                marginal_contributions.append(pred_with - pred_without)
            
            shap_values[i] = np.mean(marginal_contributions)
        
        return shap_values
    
    def _predict_instance(self, instance: np.ndarray) -> float:
        """Make prediction for a single instance."""
        instance_2d = instance.reshape(1, -1)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(instance_2d)[0, 1]
        else:
            return self.model.predict(instance_2d)[0]

class LIMEExplainer:
    """LIME-based local explanation."""
    
    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
    
    def explain_instance(self, instance: np.ndarray, num_features: int = 5) -> Dict:
        """Explain prediction using LIME approach."""
        # Generate perturbations around the instance
        perturbations = self._generate_perturbations(instance, n_samples=1000)
        
        # Get predictions for perturbations
        predictions = self._get_predictions(perturbations)
        
        # Fit linear model to local neighborhood
        weights = self._compute_weights(instance, perturbations)
        coefficients = self._fit_linear_model(perturbations, predictions, weights)
        
        # Get top contributing features
        top_features = np.argsort(np.abs(coefficients))[-num_features:]
        
        explanation = {
            'coefficients': coefficients,
            'top_features': top_features,
            'feature_importance': {i: coefficients[i] for i in top_features},
            'intercept': self._predict_instance(instance),
            'local_r2': self._compute_local_r2(perturbations, predictions, coefficients, weights)
        }
        
        return explanation
    
    def _generate_perturbations(self, instance: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate perturbations around the instance."""
        perturbations = []
        
        for _ in range(n_samples):
            perturbation = instance.copy()
            
            # Randomly perturb features
            for i in range(len(instance)):
                if np.random.random() < 0.5:  # 50% chance to perturb each feature
                    # Add Gaussian noise
                    perturbation[i] += np.random.normal(0, 0.1 * np.std(instance))
            
            perturbations.append(perturbation)
        
        return np.array(perturbations)
    
    def _get_predictions(self, perturbations: np.ndarray) -> np.ndarray:
        """Get model predictions for perturbations."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(perturbations)[:, 1]
        else:
            return self.model.predict(perturbations)
    
    def _compute_weights(self, instance: np.ndarray, perturbations: np.ndarray) -> np.ndarray:
        """Compute weights based on distance to original instance."""
        distances = np.linalg.norm(perturbations - instance, axis=1)
        # Exponential kernel
        weights = np.exp(-distances**2 / (2 * 0.25**2))
        return weights
    
    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Fit weighted linear regression."""
        # Weighted least squares
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        
        # Add regularization for stability
        regularization = 1e-6 * np.eye(X.shape[1])
        coefficients = np.linalg.solve(XtWX + regularization, XtWy)
        
        return coefficients
    
    def _predict_instance(self, instance: np.ndarray) -> float:
        """Predict single instance."""
        instance_2d = instance.reshape(1, -1)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(instance_2d)[0, 1]
        else:
            return self.model.predict(instance_2d)[0]
    
    def _compute_local_r2(self, X: np.ndarray, y: np.ndarray, 
                         coefficients: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted R-squared for local model."""
        y_pred = X @ coefficients
        ss_res = np.sum(weights * (y - y_pred)**2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
        return 1 - (ss_res / ss_tot)

class ModelInterpreter:
    """Comprehensive model interpretation toolkit."""
    
    def __init__(self, model: BaseEstimator, X_train: np.ndarray, 
                 feature_names: Optional[List[str]] = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.shap_explainer = SHAPExplainer(model, X_train)
        self.lime_explainer = LIMEExplainer(model, feature_names)
    
    def global_feature_importance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Compute global feature importance."""
        # Permutation importance
        perm_importance = permutation_importance(
            self.model, X_test, y_test, n_repeats=10, random_state=42
        )
        
        # SHAP-based global importance (average of absolute SHAP values)
        shap_importances = []
        sample_size = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        
        for idx in sample_indices:
            explanation = self.shap_explainer.explain_instance(X_test[idx])
            shap_importances.append(np.abs(explanation['shap_values']))
        
        avg_shap_importance = np.mean(shap_importances, axis=0)
        
        importance_dict = {
            'permutation_importance': {
                self.feature_names[i]: perm_importance.importances_mean[i] 
                for i in range(len(self.feature_names))
            },
            'shap_importance': {
                self.feature_names[i]: avg_shap_importance[i] 
                for i in range(len(self.feature_names))
            }
        }
        
        return importance_dict
    
    def explain_prediction(self, instance: np.ndarray, method: str = 'both') -> Dict:
        """Explain a single prediction using specified method(s)."""
        explanations = {}
        
        if method in ['shap', 'both']:
            explanations['shap'] = self.shap_explainer.explain_instance(instance)
        
        if method in ['lime', 'both']:
            explanations['lime'] = self.lime_explainer.explain_instance(instance)
        
        return explanations
    
    def plot_feature_importance(self, importance_dict: Dict, method: str = 'shap'):
        """Plot feature importance."""
        importance_data = importance_dict[f'{method}_importance']
        
        features = list(importance_data.keys())
        values = list(importance_data.values())
        
        # Sort by importance
        sorted_indices = np.argsort(values)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_features)), sorted_values)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel(f'{method.upper()} Importance')
        plt.title(f'Feature Importance ({method.upper()})')
        plt.tight_layout()
        plt.show()
    
    def plot_explanation(self, explanation: Dict, method: str = 'shap'):
        """Plot individual prediction explanation."""
        if method == 'shap':
            shap_data = explanation['shap']
            shap_values = shap_data['shap_values']
            
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(self.feature_names))
            
            colors = ['red' if val < 0 else 'blue' for val in shap_values]
            plt.barh(y_pos, shap_values, color=colors, alpha=0.7)
            plt.yticks(y_pos, self.feature_names)
            plt.xlabel('SHAP Value')
            plt.title('SHAP Feature Contributions')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
        elif method == 'lime':
            lime_data = explanation['lime']
            top_features = lime_data['top_features']
            coefficients = lime_data['coefficients']
            
            feature_names = [self.feature_names[i] for i in top_features]
            feature_values = [coefficients[i] for i in top_features]
            
            plt.figure(figsize=(10, 6))
            colors = ['red' if val < 0 else 'blue' for val in feature_values]
            plt.barh(range(len(feature_names)), feature_values, color=colors, alpha=0.7)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('LIME Coefficient')
            plt.title('LIME Feature Contributions')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model Interpretability Suite Demo")
    print("=" * 50)
    print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    
    # Create interpreter
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    interpreter = ModelInterpreter(model, X_train, feature_names)
    
    # Global feature importance
    print("\n1. Computing Global Feature Importance...")
    importance = interpreter.global_feature_importance(X_test, y_test)
    
    print("\nTop 5 Features (SHAP):")
    shap_importance = importance['shap_importance']
    sorted_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance_val in sorted_features[:5]:
        print(f"  {feature}: {importance_val:.4f}")
    
    # Individual prediction explanation
    print("\n2. Explaining Individual Prediction...")
    test_instance = X_test[0]
    explanation = interpreter.explain_prediction(test_instance, method='both')
    
    if 'shap' in explanation:
        shap_values = explanation['shap']['shap_values']
        print(f"\nSHAP Explanation (sum: {np.sum(shap_values):.4f}):")
        for i, val in enumerate(shap_values[:5]):
            print(f"  {feature_names[i]}: {val:.4f}")
    
    if 'lime' in explanation:
        lime_r2 = explanation['lime']['local_r2']
        print(f"\nLIME Local Model R²: {lime_r2:.4f}")
    
    print("\nInterpretability demo completed!")
