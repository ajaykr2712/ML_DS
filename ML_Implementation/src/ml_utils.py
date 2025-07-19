"""
Advanced ML Utilities and Helper Functions
==========================================

A comprehensive collection of utility functions for machine learning tasks
including data preprocessing, feature engineering, model evaluation, and visualization.

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """Advanced data preprocessing utilities."""
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame, 
        strategy: str = 'auto',
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values with various strategies.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        strategy : str, default='auto'
            Strategy: 'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
        threshold : float, default=0.5
            Threshold for dropping columns (percentage of missing values)
        
        Returns:
        --------
        df_cleaned : DataFrame
            Cleaned dataframe
        """
        df_copy = df.copy()
        
        if strategy == 'auto':
            # Drop columns with too many missing values
            missing_pct = df_copy.isnull().sum() / len(df_copy)
            cols_to_drop = missing_pct[missing_pct > threshold].index
            df_copy = df_copy.drop(columns=cols_to_drop)
            
            # Handle remaining missing values
            for col in df_copy.columns:
                if df_copy[col].isnull().any():
                    if df_copy[col].dtype in ['int64', 'float64']:
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
                    else:
                        df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        elif strategy == 'drop':
            df_copy = df_copy.dropna()
        
        elif strategy == 'mean':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        
        elif strategy == 'median':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        
        elif strategy == 'mode':
            for col in df_copy.columns:
                if df_copy[col].isnull().any():
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        elif strategy == 'forward_fill':
            df_copy = df_copy.fillna(method='ffill')
        
        elif strategy == 'backward_fill':
            df_copy = df_copy.fillna(method='bfill')
        
        return df_copy
    
    @staticmethod
    def detect_outliers(
        df: pd.DataFrame, 
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers using various methods.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        method : str, default='iqr'
            Method: 'iqr', 'z_score', 'modified_z_score'
        threshold : float, default=1.5
            Threshold for outlier detection
        
        Returns:
        --------
        outliers : dict
            Dictionary mapping column names to outlier indices
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
            elif method == 'z_score':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = df[z_scores > threshold].index
            
            elif method == 'modified_z_score':
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outliers[col] = df[np.abs(modified_z_scores) > threshold].index
        
        return outliers
    
    @staticmethod
    def encode_categorical_features(
        df: pd.DataFrame,
        encoding_type: str = 'auto',
        max_categories: int = 10
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical features with various strategies.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        encoding_type : str, default='auto'
            Encoding: 'auto', 'label', 'onehot', 'target'
        max_categories : int, default=10
            Maximum categories for one-hot encoding
        
        Returns:
        --------
        df_encoded : DataFrame
            Encoded dataframe
        encoders : dict
            Dictionary of fitted encoders
        """
        df_copy = df.copy()
        encoders = {}
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            n_categories = df_copy[col].nunique()
            
            if encoding_type == 'auto':
                if n_categories <= max_categories:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(df_copy[col], prefix=col)
                    df_copy = pd.concat([df_copy.drop(col, axis=1), dummies], axis=1)
                    encoders[col] = 'onehot'
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    encoders[col] = le
            
            elif encoding_type == 'label':
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                encoders[col] = le
            
            elif encoding_type == 'onehot':
                dummies = pd.get_dummies(df_copy[col], prefix=col)
                df_copy = pd.concat([df_copy.drop(col, axis=1), dummies], axis=1)
                encoders[col] = 'onehot'
        
        return df_copy, encoders


class FeatureEngineer:
    """Advanced feature engineering utilities."""
    
    @staticmethod
    def create_polynomial_features(
        X: np.ndarray, 
        degree: int = 2,
        interaction_only: bool = False
    ) -> np.ndarray:
        """
        Create polynomial features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        degree : int, default=2
            Degree of polynomial features
        interaction_only : bool, default=False
            If True, only interaction features are produced
        
        Returns:
        --------
        X_poly : array, shape (n_samples, n_poly_features)
            Polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
        return poly.fit_transform(X)
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Create time-based features from datetime column.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        date_col : str
            Name of datetime column
        
        Returns:
        --------
        df_with_time_features : DataFrame
            Dataframe with additional time features
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Extract time components
        df_copy[f'{date_col}_year'] = df_copy[date_col].dt.year
        df_copy[f'{date_col}_month'] = df_copy[date_col].dt.month
        df_copy[f'{date_col}_day'] = df_copy[date_col].dt.day
        df_copy[f'{date_col}_dayofweek'] = df_copy[date_col].dt.dayofweek
        df_copy[f'{date_col}_dayofyear'] = df_copy[date_col].dt.dayofyear
        df_copy[f'{date_col}_quarter'] = df_copy[date_col].dt.quarter
        df_copy[f'{date_col}_hour'] = df_copy[date_col].dt.hour
        df_copy[f'{date_col}_minute'] = df_copy[date_col].dt.minute
        
        # Cyclical encoding for periodic features
        df_copy[f'{date_col}_month_sin'] = np.sin(2 * np.pi * df_copy[f'{date_col}_month'] / 12)
        df_copy[f'{date_col}_month_cos'] = np.cos(2 * np.pi * df_copy[f'{date_col}_month'] / 12)
        df_copy[f'{date_col}_dayofweek_sin'] = np.sin(2 * np.pi * df_copy[f'{date_col}_dayofweek'] / 7)
        df_copy[f'{date_col}_dayofweek_cos'] = np.cos(2 * np.pi * df_copy[f'{date_col}_dayofweek'] / 7)
        df_copy[f'{date_col}_hour_sin'] = np.sin(2 * np.pi * df_copy[f'{date_col}_hour'] / 24)
        df_copy[f'{date_col}_hour_cos'] = np.cos(2 * np.pi * df_copy[f'{date_col}_hour'] / 24)
        
        return df_copy
    
    @staticmethod
    def create_lag_features(
        df: pd.DataFrame, 
        target_col: str, 
        lags: List[int],
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        target_col : str
            Name of target column
        lags : list of int
            List of lag periods
        group_col : str, optional
            Column to group by (e.g., for multiple time series)
        
        Returns:
        --------
        df_with_lags : DataFrame
            Dataframe with lag features
        """
        df_copy = df.copy()
        
        for lag in lags:
            if group_col:
                df_copy[f'{target_col}_lag_{lag}'] = df_copy.groupby(group_col)[target_col].shift(lag)
            else:
                df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
        
        return df_copy


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Comprehensive classification evaluation.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like, optional
            Predicted probabilities
        average : str, default='weighted'
            Averaging strategy for multi-class metrics
        
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # AUC metrics (for binary or multi-class with probabilities)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
            except Exception as e:
                print(f"Warning: Could not compute AUC score: {e}")
        
        return metrics
    
    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Comprehensive regression evaluation.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        classes : list, optional
            Class names
        figsize : tuple, default=(8, 6)
            Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classes or np.unique(y_true),
            yticklabels=classes or np.unique(y_true)
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot ROC curve for binary classification.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_proba : array-like
            Predicted probabilities for positive class
        figsize : tuple, default=(8, 6)
            Figure size
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_learning_curve(
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        train_sizes: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot learning curve to diagnose bias-variance tradeoff.
        
        Parameters:
        -----------
        estimator : estimator object
            A object of that type is instantiated for each grid point
        X : array-like, shape (n_samples, n_features)
            Training vectors
        y : array-like, shape (n_samples,)
            Target values
        cv : int, default=5
            Cross-validation fold
        train_sizes : array-like, optional
            Training set sizes
        figsize : tuple, default=(10, 6)
            Figure size
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()


class Visualizer:
    """Advanced visualization utilities for ML."""
    
    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot feature importance scores.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        importance_scores : array-like
            Feature importance scores
        top_k : int, default=20
            Number of top features to display
        figsize : tuple, default=(10, 8)
            Figure size
        """
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Select top k features
        top_features = importance_df.head(top_k)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_k} Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Plot correlation matrix heatmap.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        figsize : tuple, default=(12, 10)
            Figure size
        """
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_distribution_comparison(
        data1: np.ndarray,
        data2: np.ndarray,
        labels: List[str] = ['Data 1', 'Data 2'],
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Compare distributions of two datasets.
        
        Parameters:
        -----------
        data1 : array-like
            First dataset
        data2 : array-like
            Second dataset
        labels : list, default=['Data 1', 'Data 2']
            Labels for datasets
        figsize : tuple, default=(12, 5)
            Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram comparison
        axes[0].hist(data1, alpha=0.7, label=labels[0], bins=30)
        axes[0].hist(data2, alpha=0.7, label=labels[1], bins=30)
        axes[0].set_title('Distribution Comparison')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1].boxplot([data1, data2], labels=labels)
        axes[1].set_title('Box Plot Comparison')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data for demonstration
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    print("="*60)
    print("ML Utilities Demo")
    print("="*60)
    
    # Classification example
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    # Train a model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
    
    print("Classification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    feature_names = [f'feature_{i}' for i in range(X_clf.shape[1])]
    importance_scores = clf.feature_importances_
    
    print("\nTop 5 Important Features:")
    for i, (name, score) in enumerate(zip(feature_names, importance_scores)):
        if i < 5:
            print(f"{name}: {score:.4f}")
    
    print("\nDemo completed successfully!")
