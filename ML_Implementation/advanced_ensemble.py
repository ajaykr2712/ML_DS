"""
Advanced Ensemble Methods Implementation
=======================================

This module implements state-of-the-art ensemble methods including:
- Stacked Generalization (Stacking)
- Bayesian Model Averaging
- Dynamic Ensemble Selection
- Multi-level Ensemble Architecture

Author: Ensemble Learning Team
Date: July 2025
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import logging

class AdvancedStackingEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced stacking ensemble with multiple layers."""
    
    def __init__(self, base_estimators=None, meta_estimator=None, use_probas=True, cv=5):
        self.base_estimators = base_estimators or [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            SVC(probability=True, random_state=42)
        ]
        self.meta_estimator = meta_estimator or LogisticRegression()
        self.use_probas = use_probas
        self.cv = cv
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X, y):
        """Fit the stacking ensemble."""
        self.logger.info("Training stacking ensemble...")
        
        # Train base estimators and create meta-features
        meta_features = []
        
        for i, estimator in enumerate(self.base_estimators):
            self.logger.info(f"Training base estimator {i+1}/{len(self.base_estimators)}")
            
            if self.use_probas and hasattr(estimator, 'predict_proba'):
                # Use cross-validation to get out-of-fold predictions
                cv_preds = cross_val_predict(estimator, X, y, cv=self.cv, method='predict_proba')
                meta_features.append(cv_preds)
            else:
                cv_preds = cross_val_predict(estimator, X, y, cv=self.cv)
                meta_features.append(cv_preds.reshape(-1, 1))
            
            # Fit estimator on full dataset
            estimator.fit(X, y)
        
        # Concatenate meta-features
        X_meta = np.concatenate(meta_features, axis=1)
        
        # Train meta-estimator
        self.meta_estimator.fit(X_meta, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble."""
        X_meta = self._get_meta_features(X)
        return self.meta_estimator.predict(X_meta)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        X_meta = self._get_meta_features(X)
        return self.meta_estimator.predict_proba(X_meta)
    
    def _get_meta_features(self, X):
        """Generate meta-features from base estimators."""
        meta_features = []
        
        for estimator in self.base_estimators:
            if self.use_probas and hasattr(estimator, 'predict_proba'):
                preds = estimator.predict_proba(X)
            else:
                preds = estimator.predict(X).reshape(-1, 1)
            meta_features.append(preds)
        
        return np.concatenate(meta_features, axis=1)

class BayesianModelAveraging:
    """Bayesian Model Averaging for ensemble predictions."""
    
    def __init__(self, models=None, prior_weights=None):
        self.models = models or []
        self.prior_weights = prior_weights
        self.posterior_weights = None
        self.model_likelihoods = None
        
    def fit(self, X, y):
        """Fit BMA ensemble."""
        if not self.models:
            raise ValueError("No models provided for BMA")
        
        # Calculate model likelihoods using cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        likelihoods = []
        
        for model in self.models:
            fold_likelihoods = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_val)
                    likelihood = np.mean([probs[i, y_val[i]] for i in range(len(y_val))])
                else:
                    preds = model.predict(X_val)
                    likelihood = np.mean(preds == y_val)
                
                fold_likelihoods.append(likelihood)
            
            avg_likelihood = np.mean(fold_likelihoods)
            likelihoods.append(avg_likelihood)
        
        self.model_likelihoods = np.array(likelihoods)
        
        # Calculate posterior weights using Bayes' theorem
        if self.prior_weights is None:
            self.prior_weights = np.ones(len(self.models)) / len(self.models)
        
        # Posterior ∝ Likelihood × Prior
        unnormalized_posterior = self.model_likelihoods * self.prior_weights
        self.posterior_weights = unnormalized_posterior / np.sum(unnormalized_posterior)
        
        # Fit all models on full dataset
        for model in self.models:
            model.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Weighted average of model predictions."""
        if self.posterior_weights is None:
            raise ValueError("BMA not fitted yet")
        
        weighted_probs = None
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
            else:
                # Convert hard predictions to probabilities
                preds = model.predict(X)
                n_classes = len(np.unique(preds))
                probs = np.zeros((len(preds), n_classes))
                for j, pred in enumerate(preds):
                    probs[j, pred] = 1.0
            
            if weighted_probs is None:
                weighted_probs = self.posterior_weights[i] * probs
            else:
                weighted_probs += self.posterior_weights[i] * probs
        
        return weighted_probs
    
    def predict(self, X):
        """Make predictions using BMA."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Advanced Ensemble Methods Demo")
    print("=" * 50)
    
    # Test Stacking Ensemble
    print("\n1. Testing Advanced Stacking Ensemble:")
    stacking_ensemble = AdvancedStackingEnsemble()
    stacking_ensemble.fit(X_train, y_train)
    
    stacking_preds = stacking_ensemble.predict(X_test)
    stacking_accuracy = accuracy_score(y_test, stacking_preds)
    print(f"Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")
    
    # Test Bayesian Model Averaging
    print("\n2. Testing Bayesian Model Averaging:")
    models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    bma = BayesianModelAveraging(models)
    bma.fit(X_train, y_train)
    
    bma_preds = bma.predict(X_test)
    bma_accuracy = accuracy_score(y_test, bma_preds)
    print(f"BMA Accuracy: {bma_accuracy:.4f}")
    print(f"Model Weights: {bma.posterior_weights}")
    
    print("\nEnsemble methods demo completed!")
