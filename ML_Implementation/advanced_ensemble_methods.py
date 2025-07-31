"""
Advanced Ensemble Methods Implementation
State-of-the-art ensemble techniques for improved model performance

Features:
- Dynamic ensemble weighting with performance tracking
- Stacking with cross-validation and meta-learning
- Advanced voting strategies (soft, hard, weighted, adaptive)
- Bayesian model averaging for uncertainty quantification
- Multi-level ensemble hierarchies
- Real-time ensemble adaptation
- Ensemble pruning and selection algorithms
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from typing import List, Dict, Union, Tuple, Optional, Callable
import joblib
import warnings
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import os

warnings.filterwarnings('ignore')

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    # Cross-validation settings
    cv_folds: int = 5
    random_state: int = 42
    
    # Stacking configuration
    use_probabilities: bool = True
    stack_method: str = 'cv'  # 'cv', 'holdout', 'full'
    
    # Voting configuration
    voting_strategy: str = 'soft'  # 'hard', 'soft', 'weighted', 'adaptive'
    
    # Dynamic weighting
    use_dynamic_weights: bool = True
    weight_update_frequency: int = 100  # samples
    performance_window: int = 1000  # samples for rolling performance
    
    # Bayesian model averaging
    use_bayesian_averaging: bool = False
    prior_strength: float = 1.0
    
    # Ensemble pruning
    enable_pruning: bool = True
    min_ensemble_size: int = 3
    pruning_threshold: float = 0.01  # minimum improvement required
    
    # Parallel processing
    n_jobs: int = -1
    
    # Logging
    verbose: bool = True

class DynamicWeightCalculator:
    """Dynamic weight calculator for ensemble members"""
    
    def __init__(self, window_size: int = 1000, decay_factor: float = 0.95):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.performance_history = {}
        self.weights = {}
        
    def update_performance(self, model_name: str, predictions: np.ndarray, 
                          true_labels: np.ndarray, metric: str = 'accuracy'):
        """Update performance history for a model"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
            
        # Calculate performance
        if metric == 'accuracy':
            score = accuracy_score(true_labels, predictions)
        elif metric == 'mse':
            score = -mean_squared_error(true_labels, predictions)  # Negative for maximization
        elif metric == 'log_loss':
            score = -log_loss(true_labels, predictions)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        # Add to history with time decay
        self.performance_history[model_name].append(score)
        
        # Keep only recent history
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
    
    def calculate_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        if not self.performance_history:
            return {}
            
        # Calculate weighted average performance for each model
        weighted_performances = {}
        
        for model_name, scores in self.performance_history.items():
            if not scores:
                continue
                
            # Apply exponential decay to recent scores
            weights = np.array([self.decay_factor ** (len(scores) - i - 1) for i in range(len(scores))])
            weighted_avg = np.average(scores, weights=weights)
            weighted_performances[model_name] = weighted_avg
        
        # Convert to ensemble weights (softmax)
        if weighted_performances:
            performances = np.array(list(weighted_performances.values()))
            # Apply softmax for weight normalization
            exp_performances = np.exp(performances - np.max(performances))
            weights = exp_performances / np.sum(exp_performances)
            
            self.weights = dict(zip(weighted_performances.keys(), weights))
        
        return self.weights

class AdvancedStackingEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced stacking ensemble with multiple meta-learners and cross-validation"""
    
    def __init__(self, base_learners: List[Tuple[str, BaseEstimator]], 
                 meta_learner: BaseEstimator = None, config: EnsembleConfig = None):
        self.base_learners = base_learners
        self.meta_learner = meta_learner or LogisticRegression(random_state=42)
        self.config = config or EnsembleConfig()
        self.fitted_base_learners = []
        self.meta_features = None
        self.is_fitted = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def _create_meta_features(self, X, y):
        """Create meta-features using cross-validation"""
        n_samples = X.shape[0]
        n_base_learners = len(self.base_learners)
        
        if hasattr(y[0], '__len__'):  # Multi-class
            n_classes = len(np.unique(y))
            meta_features = np.zeros((n_samples, n_base_learners * n_classes))
        else:
            meta_features = np.zeros((n_samples, n_base_learners))
        
        # Use stratified K-fold for classification
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for i, (name, base_learner) in enumerate(self.base_learners):
                # Clone and fit base learner
                learner_clone = joblib.load(joblib.dump(base_learner, None))
                learner_clone.fit(X_train_fold, y_train_fold)
                
                # Generate predictions for validation set
                if self.config.use_probabilities and hasattr(learner_clone, 'predict_proba'):
                    pred_proba = learner_clone.predict_proba(X_val_fold)
                    if pred_proba.shape[1] > 1:  # Multi-class
                        start_idx = i * pred_proba.shape[1]
                        end_idx = start_idx + pred_proba.shape[1]
                        meta_features[val_idx, start_idx:end_idx] = pred_proba
                    else:  # Binary classification
                        meta_features[val_idx, i] = pred_proba[:, 1]
                else:
                    meta_features[val_idx, i] = learner_clone.predict(X_val_fold)
        
        return meta_features
    
    def fit(self, X, y):
        """Fit the stacking ensemble"""
        self.logger.info("Training Advanced Stacking Ensemble...")
        
        # Create meta-features using cross-validation
        self.meta_features = self._create_meta_features(X, y)
        
        # Fit all base learners on full training data
        self.fitted_base_learners = []
        
        with ThreadPoolExecutor(max_workers=self.config.n_jobs if self.config.n_jobs > 0 else None) as executor:
            futures = []
            
            for name, base_learner in self.base_learners:
                future = executor.submit(self._fit_base_learner, base_learner, X, y, name)
                futures.append(future)
            
            for future in futures:
                fitted_learner, name = future.result()
                self.fitted_base_learners.append((name, fitted_learner))
        
        # Fit meta-learner
        self.meta_learner.fit(self.meta_features, y)
        self.is_fitted = True
        
        self.logger.info("Stacking ensemble training completed")
        return self
    
    def _fit_base_learner(self, base_learner, X, y, name):
        """Fit a single base learner"""
        learner_clone = joblib.load(joblib.dump(base_learner, None))
        learner_clone.fit(X, y)
        return learner_clone, name
    
    def _generate_meta_features_predict(self, X):
        """Generate meta-features for prediction"""
        n_samples = X.shape[0]
        n_base_learners = len(self.fitted_base_learners)
        
        # Determine number of features per base learner
        sample_pred = self.fitted_base_learners[0][1].predict_proba(X[:1]) if hasattr(self.fitted_base_learners[0][1], 'predict_proba') else None
        
        if self.config.use_probabilities and sample_pred is not None:
            n_features_per_learner = sample_pred.shape[1]
            meta_features = np.zeros((n_samples, n_base_learners * n_features_per_learner))
        else:
            meta_features = np.zeros((n_samples, n_base_learners))
        
        # Generate predictions from each base learner
        for i, (name, learner) in enumerate(self.fitted_base_learners):
            if self.config.use_probabilities and hasattr(learner, 'predict_proba'):
                pred_proba = learner.predict_proba(X)
                if pred_proba.shape[1] > 1:
                    start_idx = i * pred_proba.shape[1]
                    end_idx = start_idx + pred_proba.shape[1]
                    meta_features[:, start_idx:end_idx] = pred_proba
                else:
                    meta_features[:, i] = pred_proba[:, 1]
            else:
                meta_features[:, i] = learner.predict(X)
        
        return meta_features
    
    def predict(self, X):
        """Make predictions using the stacking ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Generate meta-features
        meta_features = self._generate_meta_features_predict(X)
        
        # Make final prediction using meta-learner
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X):
        """Predict class probabilities using the stacking ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if not hasattr(self.meta_learner, 'predict_proba'):
            raise ValueError("Meta-learner does not support probability prediction")
        
        # Generate meta-features
        meta_features = self._generate_meta_features_predict(X)
        
        # Make final prediction using meta-learner
        return self.meta_learner.predict_proba(meta_features)

class AdaptiveVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Adaptive voting ensemble with dynamic weight adjustment"""
    
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], 
                 config: EnsembleConfig = None):
        self.estimators = estimators
        self.config = config or EnsembleConfig()
        self.fitted_estimators = []
        self.weight_calculator = DynamicWeightCalculator()
        self.is_fitted = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y):
        """Fit the adaptive voting ensemble"""
        self.logger.info("Training Adaptive Voting Ensemble...")
        
        # Fit all estimators
        self.fitted_estimators = []
        
        for name, estimator in self.estimators:
            estimator_clone = joblib.load(joblib.dump(estimator, None))
            estimator_clone.fit(X, y)
            self.fitted_estimators.append((name, estimator_clone))
        
        # Initialize weights equally
        n_estimators = len(self.fitted_estimators)
        initial_weight = 1.0 / n_estimators
        
        for name, _ in self.fitted_estimators:
            self.weight_calculator.weights[name] = initial_weight
        
        self.is_fitted = True
        self.logger.info("Adaptive voting ensemble training completed")
        return self
    
    def predict(self, X):
        """Make predictions using adaptive voting"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = {}
        
        # Get predictions from all estimators
        for name, estimator in self.fitted_estimators:
            predictions[name] = estimator.predict(X)
        
        # Apply voting strategy
        if self.config.voting_strategy == 'hard':
            return self._hard_voting(predictions)
        elif self.config.voting_strategy == 'soft':
            return self._soft_voting(X)
        elif self.config.voting_strategy == 'weighted':
            return self._weighted_voting(predictions)
        elif self.config.voting_strategy == 'adaptive':
            return self._adaptive_voting(X)
        else:
            raise ValueError(f"Unknown voting strategy: {self.config.voting_strategy}")
    
    def _hard_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Hard voting strategy"""
        pred_array = np.array(list(predictions.values())).T
        # Use mode for hard voting
        from scipy import stats
        final_predictions, _ = stats.mode(pred_array, axis=1)
        return final_predictions.flatten()
    
    def _soft_voting(self, X) -> np.ndarray:
        """Soft voting using probability averaging"""
        prob_sum = None
        n_estimators = 0
        
        for name, estimator in self.fitted_estimators:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
                if prob_sum is None:
                    prob_sum = proba
                else:
                    prob_sum += proba
                n_estimators += 1
        
        if prob_sum is not None:
            avg_proba = prob_sum / n_estimators
            return np.argmax(avg_proba, axis=1)
        else:
            # Fallback to hard voting if no probability estimates available
            predictions = {name: est.predict(X) for name, est in self.fitted_estimators}
            return self._hard_voting(predictions)
    
    def _weighted_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted voting using current weights"""
        weights = self.weight_calculator.weights
        weighted_preds = np.zeros(len(list(predictions.values())[0]))
        
        for name, pred in predictions.items():
            weight = weights.get(name, 0.0)
            weighted_preds += weight * pred
        
        return np.round(weighted_preds).astype(int)
    
    def _adaptive_voting(self, X) -> np.ndarray:
        """Adaptive voting with dynamic weight updates"""
        # Get current weights
        weights = self.weight_calculator.calculate_weights()
        
        if not weights:
            # Fall back to equal weighting
            return self._soft_voting(X)
        
        # Weighted probability averaging
        prob_sum = None
        total_weight = 0
        
        for name, estimator in self.fitted_estimators:
            if hasattr(estimator, 'predict_proba') and name in weights:
                proba = estimator.predict_proba(X)
                weight = weights[name]
                
                if prob_sum is None:
                    prob_sum = weight * proba
                else:
                    prob_sum += weight * proba
                total_weight += weight
        
        if prob_sum is not None and total_weight > 0:
            avg_proba = prob_sum / total_weight
            return np.argmax(avg_proba, axis=1)
        else:
            return self._soft_voting(X)
    
    def update_weights(self, X, y_true):
        """Update ensemble weights based on recent performance"""
        for name, estimator in self.fitted_estimators:
            predictions = estimator.predict(X)
            self.weight_calculator.update_performance(name, predictions, y_true)
        
        # Recalculate weights
        self.weight_calculator.calculate_weights()

class BayesianModelAveraging(BaseEstimator, ClassifierMixin):
    """Bayesian Model Averaging for uncertainty quantification"""
    
    def __init__(self, models: List[Tuple[str, BaseEstimator]], 
                 prior_strength: float = 1.0, config: EnsembleConfig = None):
        self.models = models
        self.prior_strength = prior_strength
        self.config = config or EnsembleConfig()
        self.fitted_models = []
        self.model_weights = {}
        self.model_evidence = {}
        self.is_fitted = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y):
        """Fit BMA ensemble with evidence calculation"""
        self.logger.info("Training Bayesian Model Averaging Ensemble...")
        
        # Fit all models and calculate evidence
        for name, model in self.models:
            model_clone = joblib.load(joblib.dump(model, None))
            
            # Calculate model evidence using cross-validation log-likelihood
            evidence = self._calculate_evidence(model_clone, X, y)
            
            # Fit on full data
            model_clone.fit(X, y)
            
            self.fitted_models.append((name, model_clone))
            self.model_evidence[name] = evidence
        
        # Calculate Bayesian weights
        self._calculate_bayesian_weights()
        
        self.is_fitted = True
        self.logger.info("BMA ensemble training completed")
        return self
    
    def _calculate_evidence(self, model, X, y) -> float:
        """Calculate model evidence using cross-validation"""
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        log_likelihoods = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model_clone = joblib.load(joblib.dump(model, None))
            model_clone.fit(X_train, y_train)
            
            if hasattr(model_clone, 'predict_proba'):
                proba = model_clone.predict_proba(X_val)
                # Calculate log-likelihood
                ll = np.sum(np.log(proba[np.arange(len(y_val)), y_val] + 1e-15))
                log_likelihoods.append(ll)
        
        return np.mean(log_likelihoods)
    
    def _calculate_bayesian_weights(self):
        """Calculate Bayesian model weights using evidence"""
        if not self.model_evidence:
            return
        
        # Convert log evidence to weights using softmax
        evidence_values = np.array(list(self.model_evidence.values()))
        
        # Apply prior
        evidence_values += np.log(self.prior_strength)
        
        # Softmax for normalization
        exp_evidence = np.exp(evidence_values - np.max(evidence_values))
        weights = exp_evidence / np.sum(exp_evidence)
        
        # Store weights
        for i, (name, _) in enumerate(self.fitted_models):
            self.model_weights[name] = weights[i]
    
    def predict_proba(self, X):
        """Predict with uncertainty quantification"""
        if not self.is_fitted:
            raise ValueError("BMA ensemble must be fitted before making predictions")
        
        weighted_proba = None
        
        for name, model in self.fitted_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                weight = self.model_weights[name]
                
                if weighted_proba is None:
                    weighted_proba = weight * proba
                else:
                    weighted_proba += weight * proba
        
        return weighted_proba
    
    def predict(self, X):
        """Make predictions using BMA"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty estimates"""
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        
        # Calculate prediction uncertainty (entropy)
        uncertainty = -np.sum(proba * np.log(proba + 1e-15), axis=1)
        
        # Calculate model uncertainty (variance across models)
        model_predictions = []
        for name, model in self.fitted_models:
            if hasattr(model, 'predict_proba'):
                model_predictions.append(model.predict_proba(X))
        
        if model_predictions:
            model_var = np.var(model_predictions, axis=0)
            model_uncertainty = np.mean(model_var, axis=1)
        else:
            model_uncertainty = np.zeros(len(predictions))
        
        return {
            'predictions': predictions,
            'probabilities': proba,
            'aleatoric_uncertainty': uncertainty,  # Data uncertainty
            'epistemic_uncertainty': model_uncertainty,  # Model uncertainty
            'total_uncertainty': uncertainty + model_uncertainty
        }

class EnsemblePruner:
    """Ensemble pruning and selection algorithms"""
    
    def __init__(self, min_ensemble_size: int = 3, threshold: float = 0.01):
        self.min_ensemble_size = min_ensemble_size
        self.threshold = threshold
    
    def prune_ensemble(self, ensemble_models: List[Tuple[str, BaseEstimator]], 
                      X_val, y_val) -> List[Tuple[str, BaseEstimator]]:
        """Prune ensemble using forward selection"""
        if len(ensemble_models) <= self.min_ensemble_size:
            return ensemble_models
        
        # Calculate individual model performances
        model_scores = {}
        for name, model in ensemble_models:
            pred = model.predict(X_val)
            score = accuracy_score(y_val, pred)
            model_scores[name] = score
        
        # Start with best performing model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        selected_models = [best_model[0]]
        best_ensemble_score = best_model[1]
        
        remaining_models = [name for name, _ in ensemble_models if name != best_model[0]]
        
        # Forward selection
        while len(remaining_models) > 0 and len(selected_models) < len(ensemble_models):
            best_addition = None
            best_score = best_ensemble_score
            
            for candidate in remaining_models:
                # Test ensemble with candidate added
                test_ensemble = selected_models + [candidate]
                ensemble_pred = self._ensemble_predict(test_ensemble, ensemble_models, X_val)
                score = accuracy_score(y_val, ensemble_pred)
                
                if score > best_score + self.threshold:
                    best_score = score
                    best_addition = candidate
            
            if best_addition is not None:
                selected_models.append(best_addition)
                remaining_models.remove(best_addition)
                best_ensemble_score = best_score
            else:
                break
        
        # Return selected models
        return [(name, model) for name, model in ensemble_models if name in selected_models]
    
    def _ensemble_predict(self, selected_names: List[str], 
                         all_models: List[Tuple[str, BaseEstimator]], X):
        """Make ensemble prediction with selected models"""
        predictions = []
        
        for name, model in all_models:
            if name in selected_names:
                pred = model.predict(X)
                predictions.append(pred)
        
        if predictions:
            # Simple majority voting
            pred_array = np.array(predictions).T
            from scipy import stats
            final_pred, _ = stats.mode(pred_array, axis=1)
            return final_pred.flatten()
        else:
            return np.array([])

def create_advanced_ensemble(ensemble_type: str = 'stacking', 
                           base_models: List[BaseEstimator] = None,
                           config: EnsembleConfig = None) -> BaseEstimator:
    """Factory function to create advanced ensemble models"""
    
    if config is None:
        config = EnsembleConfig()
    
    if base_models is None:
        # Default base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100,), random_state=42))
        ]
    else:
        # Convert to named tuples if not already
        if isinstance(base_models[0], BaseEstimator):
            base_models = [(f'model_{i}', model) for i, model in enumerate(base_models)]
    
    if ensemble_type == 'stacking':
        return AdvancedStackingEnsemble(base_models, config=config)
    elif ensemble_type == 'adaptive_voting':
        return AdaptiveVotingEnsemble(base_models, config=config)
    elif ensemble_type == 'bayesian':
        return BayesianModelAveraging(base_models, config=config)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                             n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different ensemble methods
    ensemble_types = ['stacking', 'adaptive_voting', 'bayesian']
    
    for ens_type in ensemble_types:
        print(f"\n{'='*50}")
        print(f"Testing {ens_type.upper()} Ensemble")
        print(f"{'='*50}")
        
        # Create and train ensemble
        ensemble = create_advanced_ensemble(ens_type)
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Additional analysis for Bayesian ensemble
        if ens_type == 'bayesian':
            results = ensemble.predict_with_uncertainty(X_test)
            print(f"Mean Aleatoric Uncertainty: {np.mean(results['aleatoric_uncertainty']):.4f}")
            print(f"Mean Epistemic Uncertainty: {np.mean(results['epistemic_uncertainty']):.4f}")
