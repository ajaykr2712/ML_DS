"""
Advanced Feature Engineering Toolkit
Comprehensive feature engineering with automated pipelines
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA, FastICA
from typing import List, Dict, Optional
import logging
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline with multiple techniques
    """
    
    def __init__(self, 
                 numerical_features: List[str] = None,
                 categorical_features: List[str] = None,
                 datetime_features: List[str] = None,
                 text_features: List[str] = None,
                 target_column: str = None):
        """
        Initialize the feature engineer
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            datetime_features: List of datetime feature names
            text_features: List of text feature names
            target_column: Target variable name
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.datetime_features = datetime_features or []
        self.text_features = text_features or []
        self.target_column = target_column
        
        # Feature engineering components
        self.scalers = {}
        self.encoders = {}
        self.selectors = {}
        self.transformers = {}
        
        # Generated features tracking
        self.generated_features = []
        self.feature_importance = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineering pipeline"""
        self.logger.info("Fitting feature engineering pipeline...")
        
        # Auto-detect feature types if not provided
        if not any([self.numerical_features, self.categorical_features, 
                   self.datetime_features, self.text_features]):
            self._auto_detect_features(X)
        
        # Fit numerical feature transformers
        self._fit_numerical_transformers(X)
        
        # Fit categorical feature transformers
        self._fit_categorical_transformers(X, y)
        
        # Fit datetime feature transformers
        self._fit_datetime_transformers(X)
        
        # Fit text feature transformers
        self._fit_text_transformers(X)
        
        # Fit feature selection
        if y is not None:
            self._fit_feature_selection(X, y)
        
        self.logger.info("Feature engineering pipeline fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted pipeline"""
        self.logger.info("Applying feature transformations...")
        
        X_transformed = X.copy()
        
        # Apply numerical transformations
        X_transformed = self._transform_numerical_features(X_transformed)
        
        # Apply categorical transformations
        X_transformed = self._transform_categorical_features(X_transformed)
        
        # Apply datetime transformations
        X_transformed = self._transform_datetime_features(X_transformed)
        
        # Apply text transformations
        X_transformed = self._transform_text_features(X_transformed)
        
        # Generate interaction features
        X_transformed = self._generate_interaction_features(X_transformed)
        
        # Generate polynomial features
        X_transformed = self._generate_polynomial_features(X_transformed)
        
        # Generate statistical features
        X_transformed = self._generate_statistical_features(X_transformed)
        
        # Apply feature selection
        X_transformed = self._apply_feature_selection(X_transformed)
        
        self.logger.info(f"Feature engineering completed. Shape: {X_transformed.shape}")
        return X_transformed
    
    def _auto_detect_features(self, X: pd.DataFrame):
        """Automatically detect feature types"""
        self.logger.info("Auto-detecting feature types...")
        
        for col in X.columns:
            if col == self.target_column:
                continue
                
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numerical_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(X[col]):
                self.datetime_features.append(col)
            elif X[col].dtype == 'object':
                # Check if it's text or categorical
                avg_length = X[col].astype(str).str.len().mean()
                if avg_length > 50:  # Assume text if average length > 50
                    self.text_features.append(col)
                else:
                    self.categorical_features.append(col)
        
        self.logger.info(f"Detected {len(self.numerical_features)} numerical, "
                        f"{len(self.categorical_features)} categorical, "
                        f"{len(self.datetime_features)} datetime, "
                        f"{len(self.text_features)} text features")
    
    def _fit_numerical_transformers(self, X: pd.DataFrame):
        """Fit numerical feature transformers"""
        if not self.numerical_features:
            return
            
        self.logger.info("Fitting numerical transformers...")
        
        # Standard scaling
        self.scalers['standard'] = StandardScaler()
        self.scalers['standard'].fit(X[self.numerical_features])
        
        # Robust scaling
        self.scalers['robust'] = RobustScaler()
        self.scalers['robust'].fit(X[self.numerical_features])
        
        # Min-Max scaling
        self.scalers['minmax'] = MinMaxScaler()
        self.scalers['minmax'].fit(X[self.numerical_features])
        
        # PCA for dimensionality reduction
        self.transformers['pca'] = PCA(n_components=0.95)
        self.transformers['pca'].fit(X[self.numerical_features])
        
        # ICA for independent components
        n_components = min(len(self.numerical_features), 10)
        self.transformers['ica'] = FastICA(n_components=n_components, random_state=42)
        self.transformers['ica'].fit(X[self.numerical_features])
    
    def _transform_numerical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features"""
        if not self.numerical_features:
            return X
            
        # Apply robust scaling (often works best)
        scaled_features = self.scalers['robust'].transform(X[self.numerical_features])
        scaled_df = pd.DataFrame(
            scaled_features, 
            columns=[f"{col}_scaled" for col in self.numerical_features],
            index=X.index
        )
        
        # Add PCA components
        pca_features = self.transformers['pca'].transform(X[self.numerical_features])
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f"pca_{i}" for i in range(pca_features.shape[1])],
            index=X.index
        )
        
        # Add ICA components
        ica_features = self.transformers['ica'].transform(X[self.numerical_features])
        ica_df = pd.DataFrame(
            ica_features,
            columns=[f"ica_{i}" for i in range(ica_features.shape[1])],
            index=X.index
        )
        
        # Generate binned features
        binned_df = self._generate_binned_features(X[self.numerical_features])
        
        # Combine all features
        result = pd.concat([X, scaled_df, pca_df, ica_df, binned_df], axis=1)
        
        self.generated_features.extend(scaled_df.columns.tolist())
        self.generated_features.extend(pca_df.columns.tolist())
        self.generated_features.extend(ica_df.columns.tolist())
        self.generated_features.extend(binned_df.columns.tolist())
        
        return result
    
    def _generate_binned_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate binned features from numerical data"""
        binned_features = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            # Quantile-based binning
            try:
                binned_features[f"{col}_quartile"] = pd.qcut(
                    X[col], q=4, labels=False, duplicates='drop'
                )
            except ValueError:
                # Handle case where values are not unique enough for quartiles
                binned_features[f"{col}_quartile"] = 0
            
            # Equal-width binning
            binned_features[f"{col}_binned"] = pd.cut(
                X[col], bins=5, labels=False
            )
        
        return binned_features
    
    def _fit_categorical_transformers(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """Fit categorical feature transformers"""
        if not self.categorical_features:
            return
            
        self.logger.info("Fitting categorical transformers...")
        
        # Target encoding if target is provided
        if y is not None:
            self.encoders['target'] = {}
            for col in self.categorical_features:
                target_means = X.groupby(col)[y.name].mean() if y.name in X.columns else \
                             pd.concat([X[col], y], axis=1).groupby(col)[y.name].mean()
                self.encoders['target'][col] = target_means
        
        # Frequency encoding
        self.encoders['frequency'] = {}
        for col in self.categorical_features:
            self.encoders['frequency'][col] = X[col].value_counts()
    
    def _transform_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features"""
        if not self.categorical_features:
            return X
            
        result = X.copy()
        
        # One-hot encoding for low cardinality features
        for col in self.categorical_features:
            if X[col].nunique() <= 10:  # Low cardinality
                dummies = pd.get_dummies(X[col], prefix=col)
                result = pd.concat([result, dummies], axis=1)
                self.generated_features.extend(dummies.columns.tolist())
        
        # Frequency encoding
        for col in self.categorical_features:
            freq_col = f"{col}_frequency"
            result[freq_col] = X[col].map(self.encoders['frequency'][col]).fillna(0)
            self.generated_features.append(freq_col)
        
        # Target encoding (if available)
        if 'target' in self.encoders:
            for col in self.categorical_features:
                target_col = f"{col}_target_encoded"
                result[target_col] = X[col].map(self.encoders['target'][col]).fillna(
                    self.encoders['target'][col].mean()
                )
                self.generated_features.append(target_col)
        
        return result
    
    def _fit_datetime_transformers(self, X: pd.DataFrame):
        """Fit datetime feature transformers"""
        if not self.datetime_features:
            return
            
        self.logger.info("Fitting datetime transformers...")
        # No fitting required for datetime features
        pass
    
    def _transform_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime features"""
        if not self.datetime_features:
            return X
            
        result = X.copy()
        
        for col in self.datetime_features:
            dt_col = pd.to_datetime(X[col])
            
            # Extract time components
            result[f"{col}_year"] = dt_col.dt.year
            result[f"{col}_month"] = dt_col.dt.month
            result[f"{col}_day"] = dt_col.dt.day
            result[f"{col}_dayofweek"] = dt_col.dt.dayofweek
            result[f"{col}_hour"] = dt_col.dt.hour
            result[f"{col}_minute"] = dt_col.dt.minute
            
            # Generate cyclical features
            result[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12)
            result[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12)
            result[f"{col}_day_sin"] = np.sin(2 * np.pi * dt_col.dt.day / 31)
            result[f"{col}_day_cos"] = np.cos(2 * np.pi * dt_col.dt.day / 31)
            result[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
            result[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
            
            # Time-based features
            result[f"{col}_is_weekend"] = (dt_col.dt.dayofweek >= 5).astype(int)
            result[f"{col}_is_month_start"] = dt_col.dt.is_month_start.astype(int)
            result[f"{col}_is_month_end"] = dt_col.dt.is_month_end.astype(int)
            
            # Add generated features to tracking
            new_features = [col for col in result.columns if col.startswith(f"{col}_")]
            self.generated_features.extend(new_features)
        
        return result
    
    def _fit_text_transformers(self, X: pd.DataFrame):
        """Fit text feature transformers"""
        if not self.text_features:
            return
            
        self.logger.info("Fitting text transformers...")
        # TF-IDF and other text transformers would be implemented here
        pass
    
    def _transform_text_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text features"""
        if not self.text_features:
            return X
            
        result = X.copy()
        
        for col in self.text_features:
            text_col = X[col].astype(str)
            
            # Basic text statistics
            result[f"{col}_length"] = text_col.str.len()
            result[f"{col}_word_count"] = text_col.str.split().str.len()
            result[f"{col}_char_count"] = text_col.str.len()
            result[f"{col}_sentence_count"] = text_col.str.count(r'[.!?]+')
            
            # Advanced text features
            result[f"{col}_avg_word_length"] = (
                text_col.str.replace(r'[^\w\s]', '').str.split().apply(
                    lambda x: np.mean([len(word) for word in x]) if x else 0
                )
            )
            
            result[f"{col}_punctuation_count"] = text_col.str.count(r'[^\w\s]')
            result[f"{col}_uppercase_count"] = text_col.str.count(r'[A-Z]')
            result[f"{col}_digit_count"] = text_col.str.count(r'\d')
            
            # Add generated features to tracking
            new_features = [col for col in result.columns if col.startswith(f"{col}_")]
            self.generated_features.extend(new_features)
        
        return result
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between numerical variables"""
        if len(self.numerical_features) < 2:
            return X
            
        result = X.copy()
        
        # Generate pairwise interactions for top features
        top_features = self.numerical_features[:10]  # Limit to avoid explosion
        
        for feat1, feat2 in combinations(top_features, 2):
            if feat1 in X.columns and feat2 in X.columns:
                # Multiplication
                interaction_col = f"{feat1}_x_{feat2}"
                result[interaction_col] = X[feat1] * X[feat2]
                self.generated_features.append(interaction_col)
                
                # Division (with safety check)
                division_col = f"{feat1}_div_{feat2}"
                result[division_col] = X[feat1] / (X[feat2] + 1e-8)
                self.generated_features.append(division_col)
        
        return result
    
    def _generate_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features"""
        if not self.numerical_features:
            return X
            
        result = X.copy()
        
        # Generate squared and cubed features for selected numerical features
        top_features = self.numerical_features[:5]  # Limit to avoid explosion
        
        for feat in top_features:
            if feat in X.columns:
                result[f"{feat}_squared"] = X[feat] ** 2
                result[f"{feat}_cubed"] = X[feat] ** 3
                result[f"{feat}_sqrt"] = np.sqrt(np.abs(X[feat]))
                result[f"{feat}_log"] = np.log1p(np.abs(X[feat]))
                
                self.generated_features.extend([
                    f"{feat}_squared", f"{feat}_cubed", 
                    f"{feat}_sqrt", f"{feat}_log"
                ])
        
        return result
    
    def _generate_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features"""
        if len(self.numerical_features) < 2:
            return X
            
        result = X.copy()
        
        # Statistical aggregations across numerical features
        numerical_data = X[self.numerical_features]
        
        result['numerical_mean'] = numerical_data.mean(axis=1)
        result['numerical_std'] = numerical_data.std(axis=1)
        result['numerical_median'] = numerical_data.median(axis=1)
        result['numerical_min'] = numerical_data.min(axis=1)
        result['numerical_max'] = numerical_data.max(axis=1)
        result['numerical_range'] = result['numerical_max'] - result['numerical_min']
        result['numerical_skew'] = numerical_data.skew(axis=1)
        result['numerical_kurtosis'] = numerical_data.kurtosis(axis=1)
        
        stat_features = [
            'numerical_mean', 'numerical_std', 'numerical_median',
            'numerical_min', 'numerical_max', 'numerical_range',
            'numerical_skew', 'numerical_kurtosis'
        ]
        
        self.generated_features.extend(stat_features)
        
        return result
    
    def _fit_feature_selection(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selection methods"""
        self.logger.info("Fitting feature selection...")
        
        # Apply transformations to get all features
        X_transformed = self.transform(X)
        
        # Select numerical features for selection
        numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            # SelectKBest with f_classif
            k = min(50, len(numerical_cols))  # Select top 50 or all if less
            self.selectors['kbest'] = SelectKBest(score_func=f_classif, k=k)
            self.selectors['kbest'].fit(X_transformed[numerical_cols], y)
    
    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection"""
        if 'kbest' not in self.selectors:
            return X
            
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X
            
        # Get selected features
        selected_mask = self.selectors['kbest'].get_support()
        selected_features = numerical_cols[selected_mask]
        
        # Keep selected numerical features and all non-numerical features
        non_numerical_cols = X.select_dtypes(exclude=[np.number]).columns
        all_selected_cols = list(selected_features) + list(non_numerical_cols)
        
        return X[all_selected_cols]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        importance_dict = {}
        
        if 'kbest' in self.selectors:
            feature_names = self.selectors['kbest'].feature_names_in_
            scores = self.selectors['kbest'].scores_
            
            for name, score in zip(feature_names, scores):
                importance_dict[name] = score
        
        return importance_dict
    
    def get_generated_features(self) -> List[str]:
        """Get list of all generated features"""
        return self.generated_features.copy()

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'numerical_1': np.random.randn(n_samples),
        'numerical_2': np.random.randn(n_samples),
        'numerical_3': np.random.randn(n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y'], n_samples),
        'datetime_1': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'text_1': ['This is sample text ' + str(i) for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    target = np.random.randint(0, 2, n_samples)
    
    # Initialize feature engineer
    fe = AdvancedFeatureEngineer(
        numerical_features=['numerical_1', 'numerical_2', 'numerical_3'],
        categorical_features=['categorical_1', 'categorical_2'],
        datetime_features=['datetime_1'],
        text_features=['text_1']
    )
    
    # Fit and transform
    fe.fit(df, pd.Series(target, name='target'))
    df_transformed = fe.transform(df)
    
    print(f"Original features: {df.shape[1]}")
    print(f"Transformed features: {df_transformed.shape[1]}")
    print(f"Generated features: {len(fe.get_generated_features())}")
    print(f"Feature importance available: {len(fe.get_feature_importance())}")
