"""
Machine Learning Evaluation & Model Selection Examples
Author: [Your Name]
Date: [Current Date]
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    Lasso
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ModelEvaluation:
    """Class for implementing various model evaluation techniques"""
    
    def __init__(self, X, y):
        """Initialize with dataset"""
        self.X = X
        self.y = y
        
    def perform_kfold_cv(self, model, n_splits=5):
        """Implement k-fold cross validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
            
        return np.mean(scores), np.std(scores)
    
    def perform_stratified_kfold(self, model, n_splits=5):
        """Implement stratified k-fold cross validation"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X, self.y, cv=skf)
        return np.mean(scores), np.std(scores)
    
    def perform_loocv(self, model):
        """Implement Leave-One-Out Cross Validation"""
        loo = LeaveOneOut()
        scores = cross_val_score(model, self.X, self.y, cv=loo)
        return np.mean(scores), np.std(scores)

class ModelMetrics:
    """Class for calculating various evaluation metrics"""
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_prob=None):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

class ModelSelection:
    """Class for model selection techniques"""
    
    def __init__(self):
        """Initialize with common models"""
        self.models = {
            'logistic': LogisticRegression(),
            'rf': RandomForestClassifier(),
            'gb': GradientBoostingClassifier()
        }
        
    def create_regularized_model(self, alpha=1.0, regularization='l2'):
        """Create regularized models"""
        if regularization == 'l1':
            return Lasso(alpha=alpha)
        return Ridge(alpha=alpha)
    
    def create_pipeline(self, model):
        """Create a pipeline with preprocessing"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

def example_usage():
    """Example usage of the classes"""
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Initialize classes
    evaluator = ModelEvaluation(X, y)
    model_selection = ModelSelection()
    
    # Perform cross-validation
    model = model_selection.models['rf']
    kfold_score = evaluator.perform_kfold_cv(model)
    print(f"K-Fold CV Score: {kfold_score}")
    
    # Calculate metrics
    y_pred = model.fit(X, y).predict(X)
    metrics = ModelMetrics.classification_metrics(y, y_pred)
    print("\nClassification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    example_usage()
