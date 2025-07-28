"""
Advanced Feature Engineering Automation
Automated feature engineering with genetic algorithms and ML-based selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class FeatureTransformer:
    """Base class for feature transformations."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features."""
        raise NotImplementedError
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted parameters."""
        raise NotImplementedError

class NumericTransformer(FeatureTransformer):
    """Transforms numeric features."""
    
    def __init__(self, method: str = 'standard'):
        super().__init__(f"numeric_{method}")
        self.method = method
        self.scaler = None
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return X
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X[numeric_cols])
            
        elif self.method == 'log':
            # Log transform positive values
            X_scaled = np.log1p(X[numeric_cols].clip(lower=0))
            
        elif self.method == 'sqrt':
            # Square root transform
            X_scaled = np.sqrt(X[numeric_cols].clip(lower=0))
        
        result = X.copy()
        for i, col in enumerate(numeric_cols):
            if isinstance(X_scaled, np.ndarray):
                result[f"{col}_{self.method}"] = X_scaled[:, i]
            else:
                result[f"{col}_{self.method}"] = X_scaled[col]
        
        self.is_fitted = True
        return result

class CategoricalTransformer(FeatureTransformer):
    """Transforms categorical features."""
    
    def __init__(self, method: str = 'onehot', max_categories: int = 10):
        super().__init__(f"categorical_{method}")
        self.method = method
        self.max_categories = max_categories
        self.encoders = {}
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        result = X.copy()
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            
            if unique_count > self.max_categories:
                # High cardinality: use target encoding or frequency encoding
                if y is not None and self.method == 'target':
                    # Target encoding
                    target_mean = y.mean()
                    target_encoding = X.groupby(col)[y.name].mean()
                    result[f"{col}_target_encoded"] = X[col].map(target_encoding).fillna(target_mean)
                else:
                    # Frequency encoding
                    freq_encoding = X[col].value_counts()
                    result[f"{col}_freq_encoded"] = X[col].map(freq_encoding)
            
            elif self.method == 'onehot':
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(X[col], prefix=col)
                result = pd.concat([result, dummies], axis=1)
            
            elif self.method == 'label':
                # Label encoding
                le = LabelEncoder()
                result[f"{col}_label_encoded"] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
        
        self.is_fitted = True
        return result

class PolynomialTransformer(FeatureTransformer):
    """Creates polynomial features."""
    
    def __init__(self, degree: int = 2, interaction_only: bool = False):
        super().__init__(f"polynomial_deg{degree}")
        self.degree = degree
        self.interaction_only = interaction_only
        self.poly_features = None
        self.feature_names = None
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return X
        
        # Limit to prevent explosion of features
        if len(numeric_cols) > 10:
            numeric_cols = numeric_cols[:10]
        
        self.poly_features = PolynomialFeatures(
            degree=self.degree, 
            interaction_only=self.interaction_only,
            include_bias=False
        )
        
        X_poly = self.poly_features.fit_transform(X[numeric_cols])
        self.feature_names = self.poly_features.get_feature_names_out(numeric_cols)
        
        # Create result dataframe
        result = X.copy()
        for i, name in enumerate(self.feature_names):
            if name not in numeric_cols:  # Skip original features
                result[f"poly_{name}"] = X_poly[:, i]
        
        self.is_fitted = True
        return result

class TimeBasedTransformer(FeatureTransformer):
    """Extracts time-based features from datetime columns."""
    
    def __init__(self):
        super().__init__("time_based")
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        result = X.copy()
        
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                dt_col = pd.to_datetime(X[col])
                
                # Extract time components
                result[f"{col}_year"] = dt_col.dt.year
                result[f"{col}_month"] = dt_col.dt.month
                result[f"{col}_day"] = dt_col.dt.day
                result[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                result[f"{col}_hour"] = dt_col.dt.hour
                result[f"{col}_quarter"] = dt_col.dt.quarter
                
                # Cyclical encoding for periodic features
                result[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12)
                result[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12)
                result[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7)
                result[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7)
        
        self.is_fitted = True
        return result

class FeatureSelector:
    """Selects best features using various methods."""
    
    def __init__(self, method: str = 'mutual_info', k: int = 'all'):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
    
    def fit_select(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and select features."""
        if self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        elif self.method == 'f_classif':
            self.selector = SelectKBest(score_func=f_classif, k=self.k)
        elif self.method == 'random_forest':
            return self._random_forest_selection(X, y)
        
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = X.columns[self.selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def _random_forest_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features using Random Forest importance."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top k features
        if isinstance(self.k, int):
            n_features = min(self.k, len(indices))
        else:
            n_features = len(indices)
        
        selected_indices = indices[:n_features]
        self.selected_features = X.columns[selected_indices]
        
        return X[self.selected_features]

class GeneticFeatureSelector:
    """Genetic algorithm for feature selection."""
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_features = None
        self.best_score = 0
    
    def evolve_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Evolve the best feature subset using genetic algorithm."""
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            # Random binary vector representing feature selection
            individual = np.random.rand(n_features) > 0.5
            # Ensure at least one feature is selected
            if not individual.any():
                individual[np.random.randint(n_features)] = True
            population.append(individual)
        
        for generation in range(self.generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                score = self._evaluate_fitness(X.iloc[:, individual], y)
                fitness_scores.append(score)
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_score:
                self.best_score = fitness_scores[best_idx]
                self.best_features = [feature_names[i] for i, selected in enumerate(population[best_idx]) if selected]
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Keep best individuals (elitism)
            elite_size = self.population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return self.best_features
    
    def _evaluate_fitness(self, X_subset: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate fitness of a feature subset."""
        if X_subset.empty:
            return 0
        
        try:
            # Use cross-validation score as fitness
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(clf, X_subset, y, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return 0
    
    def _tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Tournament selection for genetic algorithm."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Bit-flip mutation."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_rate:
                mutated[i] = not mutated[i]
        
        # Ensure at least one feature is selected
        if not mutated.any():
            mutated[np.random.randint(len(mutated))] = True
        
        return mutated

class AutoFeatureEngineer:
    """Main automated feature engineering pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.transformers = []
        self.feature_selector = None
        self.is_fitted = False
        self.original_features = None
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Automatically engineer features."""
        self.original_features = X.columns.tolist()
        result = X.copy()
        
        # Apply transformations
        transformers = self._get_transformers()
        
        for transformer in transformers:
            print(f"Applying {transformer.name}...")
            try:
                result = transformer.fit_transform(result, y)
                self.transformers.append(transformer)
            except Exception as e:
                print(f"Error in {transformer.name}: {e}")
        
        # Feature selection
        if y is not None and self.config.get('feature_selection', True):
            print("Performing feature selection...")
            selection_method = self.config.get('selection_method', 'mutual_info')
            max_features = self.config.get('max_features', 'all')
            
            if selection_method == 'genetic':
                genetic_selector = GeneticFeatureSelector()
                selected_features = genetic_selector.evolve_features(result, y)
                result = result[selected_features]
            else:
                self.feature_selector = FeatureSelector(method=selection_method, k=max_features)
                result = self.feature_selector.fit_select(result, y)
        
        self.is_fitted = True
        print(f"Feature engineering complete. Features: {X.shape[1]} -> {result.shape[1]}")
        return result
    
    def _get_transformers(self) -> List[FeatureTransformer]:
        """Get list of transformers to apply."""
        transformers = []
        
        if self.config.get('numeric_transforms', True):
            transformers.extend([
                NumericTransformer('standard'),
                NumericTransformer('log'),
                NumericTransformer('sqrt')
            ])
        
        if self.config.get('categorical_transforms', True):
            transformers.extend([
                CategoricalTransformer('onehot'),
                CategoricalTransformer('target'),
                CategoricalTransformer('label')
            ])
        
        if self.config.get('polynomial_features', True):
            transformers.append(PolynomialTransformer(degree=2, interaction_only=True))
        
        if self.config.get('time_features', True):
            transformers.append(TimeBasedTransformer())
        
        return transformers
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """Generate a report of feature engineering results."""
        report = {
            "original_features": len(self.original_features),
            "final_features": 0,
            "transformations_applied": [t.name for t in self.transformers],
            "feature_selection_method": None
        }
        
        if self.feature_selector:
            report["feature_selection_method"] = self.feature_selector.method
            if hasattr(self.feature_selector, 'selected_features'):
                report["final_features"] = len(self.feature_selector.selected_features)
                report["selected_features"] = list(self.feature_selector.selected_features)
        
        return report

# Example usage
if __name__ == "__main__":
    print("Testing Advanced Feature Engineering Automation...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'numeric1': np.random.normal(0, 1, n_samples),
        'numeric2': np.random.exponential(2, n_samples),
        'category1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
        'date_col': pd.date_range('2020-01-01', periods=n_samples, freq='H')
    }
    
    X = pd.DataFrame(data)
    y = pd.Series(np.random.choice([0, 1], n_samples))
    
    print(f"Original data shape: {X.shape}")
    print(f"Original columns: {list(X.columns)}")
    
    # Configure auto feature engineering
    config = {
        'numeric_transforms': True,
        'categorical_transforms': True,
        'polynomial_features': True,
        'time_features': True,
        'feature_selection': True,
        'selection_method': 'mutual_info',
        'max_features': 20
    }
    
    # Initialize and run auto feature engineering
    auto_fe = AutoFeatureEngineer(config)
    X_engineered = auto_fe.fit_transform(X, y)
    
    print(f"Engineered data shape: {X_engineered.shape}")
    print(f"Feature reduction: {X.shape[1]} -> {X_engineered.shape[1]}")
    
    # Get report
    report = auto_fe.get_feature_importance_report()
    print(f"\nFeature Engineering Report:")
    print(f"Transformations applied: {report['transformations_applied']}")
    print(f"Feature selection method: {report['feature_selection_method']}")
    
    print("\nAdvanced Feature Engineering Automation implemented successfully! 🚀")
