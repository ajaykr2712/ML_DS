"""
Advanced Model Training System with Hyperparameter Optimization
Implementing Contributions #5, #6, #7, #8, #9, #10
"""

import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str
    hyperparameters: Dict[str, Any]
    cross_validation_folds: int = 5
    optimization_metric: str = 'f1_weighted'
    early_stopping: bool = True
    early_stopping_patience: int = 10


@dataclass
class TrainingResult:
    """Results from model training."""
    model: BaseEstimator
    best_score: float
    best_params: Dict[str, Any]
    cv_scores: List[float]
    training_history: Dict[str, List[float]]
    model_config: ModelConfig


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train the model."""
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        pass


class RandomForestTrainer(BaseTrainer):
    """Trainer for Random Forest models."""
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get Random Forest hyperparameter space."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train Random Forest model."""
        self.logger.info("Training Random Forest model")
        
        # Create base model
        base_model = RandomForestClassifier(
            random_state=42,
            **self.config.hyperparameters
        )
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            base_model, X, y, 
            cv=cv, 
            scoring=self.config.optimization_metric
        )
        
        # Train final model
        final_model = base_model.fit(X, y)
        
        # Create training history
        training_history = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': [cv_scores.mean()],
            'std_cv_score': [cv_scores.std()]
        }
        
        return TrainingResult(
            model=final_model,
            best_score=cv_scores.mean(),
            best_params=self.config.hyperparameters,
            cv_scores=cv_scores.tolist(),
            training_history=training_history,
            model_config=self.config
        )


class GradientBoostingTrainer(BaseTrainer):
    """Trainer for Gradient Boosting models."""
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get Gradient Boosting hyperparameter space."""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train Gradient Boosting model."""
        self.logger.info("Training Gradient Boosting model")
        
        base_model = GradientBoostingClassifier(
            random_state=42,
            **self.config.hyperparameters
        )
        
        cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            base_model, X, y,
            cv=cv,
            scoring=self.config.optimization_metric
        )
        
        final_model = base_model.fit(X, y)
        
        training_history = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': [cv_scores.mean()],
            'std_cv_score': [cv_scores.std()]
        }
        
        return TrainingResult(
            model=final_model,
            best_score=cv_scores.mean(),
            best_params=self.config.hyperparameters,
            cv_scores=cv_scores.tolist(),
            training_history=training_history,
            model_config=self.config
        )


class SVMTrainer(BaseTrainer):
    """Trainer for Support Vector Machine models."""
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get SVM hyperparameter space."""
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]  # Only for poly kernel
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train SVM model."""
        self.logger.info("Training SVM model")
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        base_model = SVC(
            random_state=42,
            probability=True,
            **self.config.hyperparameters
        )
        
        cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            base_model, X_scaled, y,
            cv=cv,
            scoring=self.config.optimization_metric
        )
        
        final_model = base_model.fit(X_scaled, y)
        
        training_history = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': [cv_scores.mean()],
            'std_cv_score': [cv_scores.std()]
        }
        
        return TrainingResult(
            model=final_model,
            best_score=cv_scores.mean(),
            best_params=self.config.hyperparameters,
            cv_scores=cv_scores.tolist(),
            training_history=training_history,
            model_config=self.config
        )


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization system."""
    
    def __init__(self, optimization_method: str = 'grid_search'):
        self.optimization_method = optimization_method
        self.logger = logging.getLogger(__name__)
    
    def optimize(
        self, 
        trainer: BaseTrainer, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_trials: int = 50
    ) -> TrainingResult:
        """Optimize hyperparameters for the given trainer."""
        
        if self.optimization_method == 'grid_search':
            return self._grid_search(trainer, X, y)
        elif self.optimization_method == 'random_search':
            return self._random_search(trainer, X, y, n_trials)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(trainer, X, y, n_trials)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _grid_search(self, trainer: BaseTrainer, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Perform grid search optimization."""
        self.logger.info("Performing grid search optimization")
        
        param_space = trainer.get_hyperparameter_space()
        best_score = -np.inf
        best_params = None
        best_result = None
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_space)
        
        for params in param_combinations:
            config = ModelConfig(
                model_type=trainer.config.model_type,
                hyperparameters=params,
                cross_validation_folds=trainer.config.cross_validation_folds,
                optimization_metric=trainer.config.optimization_metric
            )
            
            temp_trainer = type(trainer)(config)
            result = temp_trainer.train(X, y)
            
            if result.best_score > best_score:
                best_score = result.best_score
                best_params = params
                best_result = result
        
        self.logger.info(f"Best score: {best_score:.4f}, Best params: {best_params}")
        return best_result
    
    def _random_search(self, trainer: BaseTrainer, X: pd.DataFrame, y: pd.Series, n_trials: int) -> TrainingResult:
        """Perform random search optimization."""
        self.logger.info(f"Performing random search optimization with {n_trials} trials")
        
        param_space = trainer.get_hyperparameter_space()
        best_score = -np.inf
        best_result = None
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = np.random.choice(param_values)
            
            config = ModelConfig(
                model_type=trainer.config.model_type,
                hyperparameters=params,
                cross_validation_folds=trainer.config.cross_validation_folds,
                optimization_metric=trainer.config.optimization_metric
            )
            
            temp_trainer = type(trainer)(config)
            result = temp_trainer.train(X, y)
            
            if result.best_score > best_score:
                best_score = result.best_score
                best_result = result
                
            self.logger.info(f"Trial {trial + 1}/{n_trials}: Score = {result.best_score:.4f}")
        
        return best_result
    
    def _bayesian_optimization(self, trainer: BaseTrainer, X: pd.DataFrame, y: pd.Series, n_trials: int) -> TrainingResult:
        """Perform Bayesian optimization (simplified version)."""
        self.logger.info(f"Performing Bayesian optimization with {n_trials} trials")
        
        # For now, fall back to random search
        # In a full implementation, would use libraries like optuna or scikit-optimize
        return self._random_search(trainer, X, y, n_trials)
    
    def _generate_param_combinations(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations


class ModelTrainingSystem:
    """Main system for training and optimizing models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results_history = []
    
    def create_trainer(self, model_type: str, hyperparameters: Dict[str, Any]) -> BaseTrainer:
        """Create appropriate trainer based on model type."""
        config = ModelConfig(
            model_type=model_type,
            hyperparameters=hyperparameters,
            cross_validation_folds=self.config.get('cv_folds', 5),
            optimization_metric=self.config.get('metric', 'f1_weighted')
        )
        
        if model_type == 'random_forest':
            return RandomForestTrainer(config)
        elif model_type == 'gradient_boosting':
            return GradientBoostingTrainer(config)
        elif model_type == 'svm':
            return SVMTrainer(config)
        elif model_type == 'logistic_regression':
            return LogisticRegressionTrainer(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_with_optimization(
        self, 
        model_type: str, 
        X: pd.DataFrame, 
        y: pd.Series,
        optimization_method: str = 'random_search',
        n_trials: int = 50
    ) -> TrainingResult:
        """Train model with hyperparameter optimization."""
        
        # Create trainer with default parameters
        trainer = self.create_trainer(model_type, {})
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(optimization_method)
        
        # Optimize and train
        result = optimizer.optimize(trainer, X, y, n_trials)
        
        # Store result
        self.results_history.append(result)
        
        return result
    
    def save_model(self, result: TrainingResult, model_path: str):
        """Save trained model and metadata."""
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(result.model, model_dir / 'model.pkl')
        
        # Save metadata
        metadata = {
            'model_config': asdict(result.model_config),
            'best_score': result.best_score,
            'best_params': result.best_params,
            'cv_scores': result.cv_scores,
            'training_history': result.training_history
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Load trained model and metadata."""
        model_dir = Path(model_path)
        
        # Load model
        model = joblib.load(model_dir / 'model.pkl')
        
        # Load metadata
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    
    def compare_models(self, results: List[TrainingResult]) -> pd.DataFrame:
        """Compare multiple model results."""
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'model_type': result.model_config.model_type,
                'best_score': result.best_score,
                'cv_mean': np.mean(result.cv_scores),
                'cv_std': np.std(result.cv_scores),
                'best_params': str(result.best_params)
            })
        
        return pd.DataFrame(comparison_data).sort_values('best_score', ascending=False)


class LogisticRegressionTrainer(BaseTrainer):
    """Trainer for Logistic Regression models."""
    
    def get_hyperparameter_space(self) -> Dict[str, List[Any]]:
        """Get Logistic Regression hyperparameter space."""
        return {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            'max_iter': [100, 200, 500, 1000]
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train Logistic Regression model."""
        self.logger.info("Training Logistic Regression model")
        
        base_model = LogisticRegression(
            random_state=42,
            **self.config.hyperparameters
        )
        
        cv = StratifiedKFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            base_model, X, y,
            cv=cv,
            scoring=self.config.optimization_metric
        )
        
        final_model = base_model.fit(X, y)
        
        training_history = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': [cv_scores.mean()],
            'std_cv_score': [cv_scores.std()]
        }
        
        return TrainingResult(
            model=final_model,
            best_score=cv_scores.mean(),
            best_params=self.config.hyperparameters,
            cv_scores=cv_scores.tolist(),
            training_history=training_history,
            model_config=self.config
        )


if __name__ == "__main__":
    # Example usage
    config = {
        'cv_folds': 5,
        'metric': 'f1_weighted'
    }
    
    training_system = ModelTrainingSystem(config)
    print("Model training system created successfully!")
