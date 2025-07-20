"""
Comprehensive test suite for ML implementations.
Tests all core ML algorithms with various configurations and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.linear_regression import LinearRegression
from src.logistic_regression import LogisticRegression
from src.neural_network import NeuralNetwork
from src.ml_utils import (
    DataPreprocessor, FeatureSelector, ModelEvaluator,
    CrossValidator
)


class TestLinearRegression:
    """Test cases for Linear Regression implementation."""
    
    @pytest.fixture
    def regression_data(self):
        """Generate sample regression data."""
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_linear_regression_basic(self, regression_data):
        """Test basic linear regression functionality."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Test different solvers
        solvers = ['normal', 'gradient_descent', 'sgd']
        
        for solver in solvers:
            lr = LinearRegression(solver=solver, max_iter=1000)
            lr.fit(X_train, y_train)
            
            # Check predictions
            predictions = lr.predict(X_test)
            assert len(predictions) == len(y_test)
            assert isinstance(predictions, np.ndarray)
            
            # Check score
            score = lr.score(X_test, y_test)
            assert 0 <= score <= 1  # R² score should be reasonable
    
    def test_linear_regression_regularization(self, regression_data):
        """Test regularized linear regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Test Ridge regression
        ridge = LinearRegression(solver='ridge', alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_score = ridge.score(X_test, y_test)
        
        # Test Lasso regression
        lasso = LinearRegression(solver='lasso', alpha=1.0)
        lasso.fit(X_train, y_train)
        lasso_score = lasso.score(X_test, y_test)
        
        # Test Elastic Net
        elastic = LinearRegression(solver='elastic_net', alpha=1.0, l1_ratio=0.5)
        elastic.fit(X_train, y_train)
        elastic_score = elastic.score(X_test, y_test)
        
        # All should produce reasonable scores
        assert ridge_score > 0
        assert lasso_score > 0
        assert elastic_score > 0
    
    def test_linear_regression_edge_cases(self):
        """Test edge cases for linear regression."""
        # Single feature
        X_single = np.random.randn(50, 1)
        y_single = 3 * X_single.flatten() + np.random.randn(50) * 0.1
        
        lr = LinearRegression()
        lr.fit(X_single, y_single)
        predictions = lr.predict(X_single)
        assert len(predictions) == len(y_single)
        
        # Large number of features (regularization should help)
        X_large = np.random.randn(50, 100)
        y_large = np.random.randn(50)
        
        lr_reg = LinearRegression(solver='ridge', alpha=1.0)
        lr_reg.fit(X_large, y_large)
        predictions = lr_reg.predict(X_large)
        assert len(predictions) == len(y_large)


class TestLogisticRegression:
    """Test cases for Logistic Regression implementation."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, 
                                 random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    @pytest.fixture
    def multiclass_classification_data(self):
        """Generate multiclass classification data."""
        X, y = make_classification(n_samples=300, n_features=10, n_classes=3, 
                                 n_redundant=0, random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_binary_classification(self, binary_classification_data):
        """Test binary classification."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        # Test different solvers
        solvers = ['gradient_descent', 'newton', 'lbfgs']
        
        for solver in solvers:
            lr = LogisticRegression(solver=solver, max_iter=1000)
            lr.fit(X_train, y_train)
            
            # Check predictions
            predictions = lr.predict(X_test)
            assert len(predictions) == len(y_test)
            assert set(predictions).issubset({0, 1})
            
            # Check probabilities
            probabilities = lr.predict_proba(X_test)
            assert probabilities.shape == (len(y_test), 2)
            assert np.allclose(probabilities.sum(axis=1), 1.0)
            
            # Check accuracy
            accuracy = lr.score(X_test, y_test)
            assert 0.5 <= accuracy <= 1.0  # Should be better than random
    
    def test_multiclass_classification(self, multiclass_classification_data):
        """Test multiclass classification."""
        X_train, X_test, y_train, y_test = multiclass_classification_data
        
        lr = LogisticRegression(multi_class='ovr', max_iter=1000)
        lr.fit(X_train, y_train)
        
        # Check predictions
        predictions = lr.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1, 2})
        
        # Check probabilities
        probabilities = lr.predict_proba(X_test)
        assert probabilities.shape == (len(y_test), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_regularization(self, binary_classification_data):
        """Test regularized logistic regression."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        # Test L1 regularization
        lr_l1 = LogisticRegression(penalty='l1', C=1.0, max_iter=1000)
        lr_l1.fit(X_train, y_train)
        accuracy_l1 = lr_l1.score(X_test, y_test)
        
        # Test L2 regularization
        lr_l2 = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        lr_l2.fit(X_train, y_train)
        accuracy_l2 = lr_l2.score(X_test, y_test)
        
        # Both should work reasonably well
        assert accuracy_l1 > 0.5
        assert accuracy_l2 > 0.5


class TestNeuralNetwork:
    """Test cases for Neural Network implementation."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification data for neural network."""
        X, y = make_classification(n_samples=500, n_features=20, n_classes=3,
                                 n_redundant=0, random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data for neural network."""
        X, y = make_regression(n_samples=500, n_features=20, noise=0.1, 
                             random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_classification(self, classification_data):
        """Test neural network for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test different architectures
        architectures = [
            [50, 3],  # Single hidden layer
            [50, 25, 3],  # Two hidden layers
            [100, 50, 25, 3]  # Three hidden layers
        ]
        
        for hidden_layers in architectures:
            nn = NeuralNetwork(
                hidden_layers=hidden_layers,
                task='classification',
                max_iter=100,
                learning_rate=0.01
            )
            nn.fit(X_train_scaled, y_train)
            
            # Check predictions
            predictions = nn.predict(X_test_scaled)
            assert len(predictions) == len(y_test)
            
            # Check accuracy
            accuracy = nn.score(X_test_scaled, y_test)
            assert accuracy > 0.3  # Should be better than random for 3 classes
    
    def test_regression(self, regression_data):
        """Test neural network for regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        nn = NeuralNetwork(
            hidden_layers=[50, 25, 1],
            task='regression',
            max_iter=100,
            learning_rate=0.01
        )
        nn.fit(X_train_scaled, y_train)
        
        # Check predictions
        predictions = nn.predict(X_test_scaled)
        assert len(predictions) == len(y_test)
        
        # Check that predictions are reasonable
        assert np.isfinite(predictions).all()
    
    def test_different_activations(self, classification_data):
        """Test different activation functions."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        activations = ['relu', 'tanh', 'sigmoid']
        
        for activation in activations:
            nn = NeuralNetwork(
                hidden_layers=[50, 3],
                activation=activation,
                task='classification',
                max_iter=50
            )
            nn.fit(X_train_scaled, y_train)
            
            predictions = nn.predict(X_test_scaled)
            assert len(predictions) == len(y_test)
    
    def test_regularization(self, classification_data):
        """Test neural network regularization."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test with regularization
        nn_reg = NeuralNetwork(
            hidden_layers=[100, 50, 3],
            task='classification',
            alpha=0.01,  # L2 regularization
            max_iter=50
        )
        nn_reg.fit(X_train_scaled, y_train)
        
        predictions = nn_reg.predict(X_test_scaled)
        assert len(predictions) == len(y_test)


class TestMLUtils:
    """Test cases for ML utility functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing utilities."""
        X, y = make_classification(n_samples=200, n_features=20, n_classes=2,
                                 random_state=42)
        # Add some categorical and missing data
        df = pd.DataFrame(X)
        df['target'] = y
        df['categorical'] = np.random.choice(['A', 'B', 'C'], size=200)
        
        # Introduce some missing values
        df.iloc[::10, 0] = np.nan
        
        return df
    
    def test_data_preprocessor(self, sample_data):
        """Test data preprocessing utilities."""
        preprocessor = DataPreprocessor()
        
        # Test missing value handling
        df_filled = preprocessor.handle_missing_values(
            sample_data, strategy='mean', columns=sample_data.select_dtypes(include=[np.number]).columns
        )
        assert df_filled.isnull().sum().sum() < sample_data.isnull().sum().sum()
        
        # Test encoding
        df_encoded = preprocessor.encode_categorical(df_filled, ['categorical'])
        assert 'categorical_A' in df_encoded.columns or 'categorical' not in df_encoded.columns
        
        # Test scaling
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
        df_scaled = preprocessor.scale_features(df_encoded, numeric_cols, method='standard')
        
        # Check that scaling worked
        for col in numeric_cols:
            if col != 'target':  # Don't check target variable
                assert abs(df_scaled[col].mean()) < 1e-10  # Mean should be close to 0
                assert abs(df_scaled[col].std() - 1.0) < 1e-10  # Std should be close to 1
    
    def test_feature_selector(self, sample_data):
        """Test feature selection utilities."""
        # Prepare data
        X = sample_data.drop(['target', 'categorical'], axis=1)
        y = sample_data['target']
        
        # Remove rows with NaN values for this test
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        selector = FeatureSelector()
        
        # Test correlation-based selection
        selected_features = selector.select_by_correlation(X, threshold=0.9)
        assert len(selected_features) <= X.shape[1]
        
        # Test variance-based selection
        selected_features = selector.select_by_variance(X, threshold=0.01)
        assert len(selected_features) <= X.shape[1]
        
        # Test univariate selection
        selected_features = selector.select_k_best(X, y, k=10)
        assert len(selected_features) == min(10, X.shape[1])
    
    def test_model_evaluator(self):
        """Test model evaluation utilities."""
        # Generate some sample predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        y_prob = np.random.rand(10, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize to probabilities
        
        evaluator = ModelEvaluator()
        
        # Test classification metrics
        metrics = evaluator.evaluate_classification(y_true, y_pred, y_prob)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
        
        # Test regression metrics
        y_true_reg = np.random.randn(100)
        y_pred_reg = y_true_reg + np.random.randn(100) * 0.1
        
        metrics_reg = evaluator.evaluate_regression(y_true_reg, y_pred_reg)
        
        required_metrics_reg = ['mse', 'rmse', 'mae', 'r2']
        for metric in required_metrics_reg:
            assert metric in metrics_reg
    
    def test_cross_validator(self, sample_data):
        """Test cross-validation utilities."""
        # Prepare data
        X = sample_data.drop(['target', 'categorical'], axis=1).fillna(0)
        y = sample_data['target']
        
        cv = CrossValidator()
        
        # Test k-fold CV
        scores = cv.k_fold_cv(X, y, LogisticRegression(), k=3)
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)
        
        # Test stratified CV
        scores = cv.stratified_cv(X, y, LogisticRegression(), k=3)
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_complete_ml_pipeline(self):
        """Test a complete ML pipeline from data to prediction."""
        # Generate data
        X, y = make_classification(n_samples=300, n_features=20, n_classes=2,
                                 random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_train_scaled = preprocessor.scale_features(
            pd.DataFrame(X_train), range(X_train.shape[1]), method='standard'
        ).values
        X_test_scaled = preprocessor.scale_features(
            pd.DataFrame(X_test), range(X_test.shape[1]), method='standard'
        ).values
        
        # Train models
        models = {
            'logistic': LogisticRegression(max_iter=1000),
            'neural_net': NeuralNetwork([50, 25, 1], task='classification', max_iter=100)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            accuracy = (predictions == y_test).mean()
            results[name] = accuracy
        
        # All models should perform reasonably well
        for name, accuracy in results.items():
            assert accuracy > 0.6, f"{name} accuracy too low: {accuracy}"
    
    def test_model_comparison(self):
        """Test comparing different models on the same dataset."""
        # Generate challenging dataset
        X, y = make_classification(
            n_samples=500, n_features=20, n_classes=3,
            n_informative=10, n_redundant=5, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test different models
        models = {
            'logistic_ovr': LogisticRegression(multi_class='ovr', max_iter=1000),
            'neural_net_small': NeuralNetwork([30, 3], task='classification', max_iter=200),
            'neural_net_large': NeuralNetwork([100, 50, 3], task='classification', max_iter=200)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            accuracy = (predictions == y_test).mean()
            results[name] = accuracy
        
        # All models should learn something
        for name, accuracy in results.items():
            assert accuracy > 0.33, f"{name} should beat random (33%) but got {accuracy}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
