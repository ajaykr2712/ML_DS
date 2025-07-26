"""
Advanced Hyperparameter Optimization Framework
Comprehensive hyperparameter tuning with multiple optimization strategies
"""

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import logging
from typing import Dict, List
import json
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    n_trials: int = 100
    cv_folds: int = 5
    scoring: str = 'accuracy'
    timeout: int = 3600  # seconds
    n_jobs: int = -1
    random_state: int = 42
    save_results: bool = True
    early_stopping: bool = True
    early_stopping_rounds: int = 20

class BaseOptimizer:
    """Base class for hyperparameter optimizers"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def objective(self, params: Dict, X, y, model_class) -> float:
        """Objective function to optimize"""
        try:
            # Create model with suggested parameters
            model = model_class(**params)
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=self.config.cv_folds, 
                               shuffle=True, 
                               random_state=self.config.random_state)
            
            scores = cross_val_score(model, X, y, 
                                   cv=cv, 
                                   scoring=self.config.scoring,
                                   n_jobs=self.config.n_jobs)
            
            score = np.mean(scores)
            
            # Track optimization history
            self.optimization_history.append({
                'params': params,
                'score': score,
                'std': np.std(scores),
                'timestamp': time.time()
            })
            
            # Update best score
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.logger.info(f"New best score: {score:.4f} with params: {params}")
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Evaluation failed with params {params}: {e}")
            return -np.inf
    
    def optimize(self, X, y, model_class, param_space: Dict) -> Dict:
        """Optimize hyperparameters"""
        raise NotImplementedError("Subclasses must implement optimize method")
    
    def save_results(self, filename: str):
        """Save optimization results"""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'config': self.config.__dict__
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filename}")

class OptunaOptimizer(BaseOptimizer):
    """Optuna-based hyperparameter optimizer"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.study = None
    
    def _create_trial_params(self, trial, param_space: Dict) -> Dict:
        """Create parameters for a trial"""
        params = {}
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, 
                    param_config['choices']
                )
        
        return params
    
    def optimize(self, X, y, model_class, param_space: Dict) -> Dict:
        """Optimize using Optuna"""
        def objective_wrapper(trial):
            params = self._create_trial_params(trial, param_space)
            score = self.objective(params, X, y, model_class)
            return score
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Add early stopping callback
        if self.config.early_stopping:
            early_stopping = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=self.config.early_stopping_rounds
            )
            self.study.pruner = early_stopping
        
        # Optimize
        self.study.optimize(
            objective_wrapper,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        return {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials)
        }

class HyperoptOptimizer(BaseOptimizer):
    """Hyperopt-based optimizer"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.trials = Trials()
    
    def _convert_param_space(self, param_space: Dict) -> Dict:
        """Convert parameter space to hyperopt format"""
        hp_space = {}
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'int':
                hp_space[param_name] = hp.randint(
                    param_name, 
                    param_config['low'], 
                    param_config['high'] + 1
                )
            elif param_config['type'] == 'float':
                if param_config.get('log', False):
                    hp_space[param_name] = hp.loguniform(
                        param_name,
                        np.log(param_config['low']),
                        np.log(param_config['high'])
                    )
                else:
                    hp_space[param_name] = hp.uniform(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            elif param_config['type'] == 'categorical':
                hp_space[param_name] = hp.choice(
                    param_name,
                    param_config['choices']
                )
        
        return hp_space
    
    def optimize(self, X, y, model_class, param_space: Dict) -> Dict:
        """Optimize using Hyperopt"""
        hp_space = self._convert_param_space(param_space)
        
        def objective_wrapper(params):
            score = self.objective(params, X, y, model_class)
            return {'loss': -score, 'status': STATUS_OK}
        
        best = fmin(
            fn=objective_wrapper,
            space=hp_space,
            algo=tpe.suggest,
            max_evals=self.config.n_trials,
            trials=self.trials,
            rstate=np.random.RandomState(self.config.random_state)
        )
        
        return {
            'best_params': best,
            'best_score': -self.trials.best_trial['result']['loss'],
            'n_trials': len(self.trials.trials)
        }

class ScikitOptimizeOptimizer(BaseOptimizer):
    """Scikit-optimize based optimizer"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.result = None
    
    def _convert_param_space(self, param_space: Dict) -> List:
        """Convert parameter space to skopt format"""
        dimensions = []
        param_names = []
        
        for param_name, param_config in param_space.items():
            param_names.append(param_name)
            
            if param_config['type'] == 'int':
                dimensions.append(Integer(param_config['low'], param_config['high']))
            elif param_config['type'] == 'float':
                prior = 'log-uniform' if param_config.get('log', False) else 'uniform'
                dimensions.append(Real(param_config['low'], param_config['high'], prior=prior))
            elif param_config['type'] == 'categorical':
                dimensions.append(Categorical(param_config['choices']))
        
        return dimensions, param_names
    
    def optimize(self, X, y, model_class, param_space: Dict) -> Dict:
        """Optimize using scikit-optimize"""
        dimensions, param_names = self._convert_param_space(param_space)
        
        def objective_wrapper(params_list):
            params = dict(zip(param_names, params_list))
            score = self.objective(params, X, y, model_class)
            return -score  # Minimize negative score
        
        self.result = gp_minimize(
            func=objective_wrapper,
            dimensions=dimensions,
            n_calls=self.config.n_trials,
            random_state=self.config.random_state
        )
        
        best_params = dict(zip(param_names, self.result.x))
        
        return {
            'best_params': best_params,
            'best_score': -self.result.fun,
            'n_trials': len(self.result.func_vals)
        }

class EnsembleOptimizer:
    """Ensemble of multiple optimization strategies"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizers = [
            OptunaOptimizer(config),
            HyperoptOptimizer(config),
            ScikitOptimizeOptimizer(config)
        ]
        self.results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, X, y, model_class, param_space: Dict) -> Dict:
        """Run multiple optimizers and combine results"""
        self.logger.info("Starting ensemble optimization...")
        
        # Run each optimizer
        for i, optimizer in enumerate(self.optimizers):
            optimizer_name = optimizer.__class__.__name__
            self.logger.info(f"Running {optimizer_name}...")
            
            try:
                result = optimizer.optimize(X, y, model_class, param_space)
                self.results[optimizer_name] = result
                self.logger.info(f"{optimizer_name} completed. Best score: {result['best_score']:.4f}")
            except Exception as e:
                self.logger.error(f"{optimizer_name} failed: {e}")
                continue
        
        # Find overall best result
        best_optimizer = max(self.results.keys(), 
                           key=lambda k: self.results[k]['best_score'])
        best_result = self.results[best_optimizer]
        
        self.logger.info(f"Best optimizer: {best_optimizer}")
        self.logger.info(f"Best score: {best_result['best_score']:.4f}")
        
        return {
            'best_params': best_result['best_params'],
            'best_score': best_result['best_score'],
            'best_optimizer': best_optimizer,
            'all_results': self.results
        }

class AutoMLOptimizer:
    """Automated machine learning with hyperparameter optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.model_configs = self._get_model_configs()
        self.results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_model_configs(self) -> Dict:
        """Get model configurations with parameter spaces"""
        return {
            'RandomForest': {
                'class': RandomForestClassifier,
                'param_space': {
                    'n_estimators': {'type': 'int', 'low': 10, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
                }
            },
            'GradientBoosting': {
                'class': GradientBoostingClassifier,
                'param_space': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
                }
            },
            'SVM': {
                'class': SVC,
                'param_space': {
                    'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
                    'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
                    'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']}
                }
            },
            'MLPClassifier': {
                'class': MLPClassifier,
                'param_space': {
                    'hidden_layer_sizes': {'type': 'categorical', 'choices': [(100,), (50, 50), (100, 50), (200,)]},
                    'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
                    'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.01, 'log': True}
                }
            }
        }
    
    def optimize_all_models(self, X, y) -> Dict:
        """Optimize all models and return best"""
        self.logger.info("Starting AutoML optimization...")
        
        for model_name, model_config in self.model_configs.items():
            self.logger.info(f"Optimizing {model_name}...")
            
            # Use ensemble optimizer for each model
            optimizer = EnsembleOptimizer(self.config)
            result = optimizer.optimize(
                X, y, 
                model_config['class'], 
                model_config['param_space']
            )
            
            self.results[model_name] = result
            self.logger.info(f"{model_name} optimization completed. Best score: {result['best_score']:.4f}")
        
        # Find overall best model
        best_model = max(self.results.keys(), 
                        key=lambda k: self.results[k]['best_score'])
        
        self.logger.info(f"Best model: {best_model}")
        self.logger.info(f"Best score: {self.results[best_model]['best_score']:.4f}")
        
        return {
            'best_model': best_model,
            'best_params': self.results[best_model]['best_params'],
            'best_score': self.results[best_model]['best_score'],
            'all_results': self.results
        }
    
    def save_results(self, filename: str):
        """Save all optimization results"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Configuration
    config = OptimizationConfig(
        n_trials=50,
        cv_folds=3,
        scoring='accuracy',
        timeout=1800
    )
    
    # Run AutoML optimization
    automl = AutoMLOptimizer(config)
    results = automl.optimize_all_models(X, y)
    
    # Save results
    automl.save_results('automl_results.json')
    
    print(f"Best model: {results['best_model']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
