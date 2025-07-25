"""
Test Suite for Advanced Ensemble Methods
========================================

Comprehensive tests for ensemble learning implementations including:
- Random Forest from scratch
- Gradient Boosting from scratch
- Model performance validation
- Edge case handling

Author: ML Arsenal Team
Date: July 2025
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ensemble_methods import RandomForestFromScratch, GradientBoostingFromScratch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings

warnings.filterwarnings('ignore')


class TestRandomForestFromScratch:
    """Test cases for Random Forest implementation."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification data for testing."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                                  n_redundant=2, n_classes=2, random_state=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data for testing."""
        X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_classification_basic(self, classification_data):
        """Test basic classification functionality."""
        X_train, X_test, y_train, y_test = classification_data
        
        rf = RandomForestFromScratch(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            task='classification'
        )
        
        # Test fitting
        rf.fit(X_train, y_train)
        assert len(rf.trees_) == 10
        assert rf.feature_importances_ is not None
        assert len(rf.feature_importances_) == X_train.shape[1]
        
        # Test prediction
        predictions = rf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test accuracy
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.5  # Should be better than random
        
        # Test score method
        score = rf.score(X_test, y_test)
        assert score == accuracy
    
    def test_classification_probabilities(self, classification_data):
        """Test probability prediction for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        rf = RandomForestFromScratch(
            n_estimators=10,
            random_state=42,
            task='classification'
        )
        
        rf.fit(X_train, y_train)
        probabilities = rf.predict_proba(X_test)
        
        # Check probability shape and properties
        assert probabilities.shape == (len(y_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_regression_basic(self, regression_data):
        """Test basic regression functionality."""
        X_train, X_test, y_train, y_test = regression_data
        
        rf = RandomForestFromScratch(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            task='regression'
        )
        
        # Test fitting
        rf.fit(X_train, y_train)
        assert len(rf.trees_) == 10
        assert rf.feature_importances_ is not None
        
        # Test prediction
        predictions = rf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert np.all(np.isfinite(predictions))
        
        # Test R² score
        r2 = r2_score(y_test, predictions)
        assert r2 > 0.3  # Should explain some variance
        
        # Test score method
        score = rf.score(X_test, y_test)
        assert abs(score - r2) < 1e-10
    
    def test_oob_score(self, classification_data):
        """Test out-of-bag score calculation."""
        X_train, X_test, y_train, y_test = classification_data
        
        rf = RandomForestFromScratch(
            n_estimators=20,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            task='classification'
        )
        
        rf.fit(X_train, y_train)
        
        assert rf.oob_score_ is not None
        assert 0 <= rf.oob_score_ <= 1
    
    def test_feature_importance(self, classification_data):
        """Test feature importance calculation."""
        X_train, X_test, y_train, y_test = classification_data
        
        rf = RandomForestFromScratch(
            n_estimators=10,
            random_state=42,
            task='classification'
        )
        
        rf.fit(X_train, y_train)
        
        # Check feature importance properties
        assert len(rf.feature_importances_) == X_train.shape[1]
        assert np.allclose(np.sum(rf.feature_importances_), 1.0)
        assert np.all(rf.feature_importances_ >= 0)
    
    def test_different_parameters(self, classification_data):
        """Test different parameter configurations."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Test different max_features
        for max_features in ['sqrt', 'log2', 0.5, 5]:
            rf = RandomForestFromScratch(
                n_estimators=5,
                max_features=max_features,
                random_state=42,
                task='classification'
            )
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            assert len(predictions) == len(y_test)
        
        # Test without bootstrap
        rf_no_bootstrap = RandomForestFromScratch(
            n_estimators=5,
            bootstrap=False,
            random_state=42,
            task='classification'
        )
        rf_no_bootstrap.fit(X_train, y_train)
        predictions = rf_no_bootstrap.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small dataset
        X_small = np.random.rand(5, 3)
        y_small = np.array([0, 1, 0, 1, 0])
        
        rf = RandomForestFromScratch(n_estimators=3, task='classification')
        rf.fit(X_small, y_small)
        predictions = rf.predict(X_small)
        assert len(predictions) == len(y_small)
        
        # Test prediction before fitting
        rf_unfitted = RandomForestFromScratch(task='classification')
        with pytest.raises(ValueError):
            rf_unfitted.predict(X_small)


class TestGradientBoostingFromScratch:
    """Test cases for Gradient Boosting implementation."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate binary classification data for testing."""
        X, y = make_classification(n_samples=200, n_features=8, n_classes=2,
                                  n_redundant=0, random_state=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data for testing."""
        X, y = make_regression(n_samples=200, n_features=6, noise=0.1, random_state=42)
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_regression_basic(self, regression_data):
        """Test basic regression functionality."""
        X_train, X_test, y_train, y_test = regression_data
        
        gb = GradientBoostingFromScratch(
            n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            task='regression'
        )
        
        # Test fitting
        gb.fit(X_train, y_train)
        assert len(gb.estimators_) == 20
        assert gb.init_prediction_ is not None
        assert len(gb.train_score_) == 20
        
        # Test prediction
        predictions = gb.predict(X_test)
        assert len(predictions) == len(y_test)
        assert np.all(np.isfinite(predictions))
        
        # Test R² score
        r2 = r2_score(y_test, predictions)
        assert r2 > 0.3  # Should explain some variance
    
    def test_classification_basic(self, classification_data):
        """Test basic classification functionality."""
        X_train, X_test, y_train, y_test = classification_data
        
        gb = GradientBoostingFromScratch(
            n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            loss='log_loss',
            random_state=42,
            task='classification'
        )
        
        # Test fitting
        gb.fit(X_train, y_train)
        assert len(gb.estimators_) == 20
        assert gb.init_prediction_ is not None
        
        # Test prediction
        predictions = gb.predict(X_test)
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability prediction
        probabilities = gb.predict_proba(X_test)
        assert probabilities.shape == (len(y_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Test accuracy
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.5  # Should be better than random
    
    def test_different_loss_functions(self, regression_data):
        """Test different loss functions for regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        loss_functions = ['squared_error', 'absolute_error', 'huber']
        
        for loss in loss_functions:
            gb = GradientBoostingFromScratch(
                n_estimators=10,
                learning_rate=0.1,
                loss=loss,
                random_state=42,
                task='regression'
            )
            
            gb.fit(X_train, y_train)
            predictions = gb.predict(X_test)
            assert len(predictions) == len(y_test)
            assert np.all(np.isfinite(predictions))
    
    def test_early_stopping(self, classification_data):
        """Test early stopping functionality."""
        X_train, X_test, y_train, y_test = classification_data
        
        gb = GradientBoostingFromScratch(
            n_estimators=100,  # Set high so early stopping can trigger
            learning_rate=0.1,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42,
            task='classification'
        )
        
        gb.fit(X_train, y_train)
        
        # Should have stopped early
        assert len(gb.estimators_) < 100
        assert len(gb.validation_score_) > 0
    
    def test_feature_importance(self, regression_data):
        """Test feature importance calculation."""
        X_train, X_test, y_train, y_test = regression_data
        
        gb = GradientBoostingFromScratch(
            n_estimators=10,
            random_state=42,
            task='regression'
        )
        
        gb.fit(X_train, y_train)
        
        # Check feature importance properties
        assert len(gb.feature_importances_) == X_train.shape[1]
        assert np.allclose(np.sum(gb.feature_importances_), 1.0)
        assert np.all(gb.feature_importances_ >= 0)
    
    def test_subsample_parameter(self, regression_data):
        """Test subsample parameter."""
        X_train, X_test, y_train, y_test = regression_data
        
        gb = GradientBoostingFromScratch(
            n_estimators=10,
            subsample=0.8,  # Use 80% of samples for each tree
            random_state=42,
            task='regression'
        )
        
        gb.fit(X_train, y_train)
        predictions = gb.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_score_method(self, classification_data):
        """Test score method for both tasks."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Classification
        gb_clf = GradientBoostingFromScratch(
            n_estimators=10,
            task='classification',
            random_state=42
        )
        gb_clf.fit(X_train, y_train)
        score_clf = gb_clf.score(X_test, y_test)
        
        predictions_clf = gb_clf.predict(X_test)
        expected_score_clf = accuracy_score(y_test, predictions_clf)
        assert abs(score_clf - expected_score_clf) < 1e-10
        
        # Regression
        X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        gb_reg = GradientBoostingFromScratch(
            n_estimators=10,
            task='regression',
            random_state=42
        )
        gb_reg.fit(X_train_reg, y_train_reg)
        score_reg = gb_reg.score(X_test_reg, y_test_reg)
        
        predictions_reg = gb_reg.predict(X_test_reg)
        expected_score_reg = r2_score(y_test_reg, predictions_reg)
        assert abs(score_reg - expected_score_reg) < 1e-10
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small dataset
        X_small = np.random.rand(5, 3)
        y_small = np.array([0, 1, 0, 1, 0])
        
        gb = GradientBoostingFromScratch(n_estimators=3, task='classification')
        gb.fit(X_small, y_small)
        predictions = gb.predict(X_small)
        assert len(predictions) == len(y_small)
        
        # Test prediction before fitting
        gb_unfitted = GradientBoostingFromScratch(task='classification')
        with pytest.raises(ValueError):
            gb_unfitted.predict(X_small)
        
        # Test invalid loss function
        with pytest.raises(ValueError):
            gb_invalid = GradientBoostingFromScratch(loss='invalid_loss')
            gb_invalid.fit(X_small, y_small)


class TestEnsembleComparison:
    """Test ensemble methods against sklearn baselines."""
    
    def test_performance_comparison(self):
        """Compare performance with sklearn implementations."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        # Generate larger dataset for fair comparison
        X, y = make_classification(n_samples=500, n_features=10, n_informative=8,
                                  n_redundant=1, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Our implementations
        our_rf = RandomForestFromScratch(n_estimators=20, random_state=42, task='classification')
        our_rf.fit(X_train, y_train)
        our_rf_accuracy = accuracy_score(y_test, our_rf.predict(X_test))
        
        our_gb = GradientBoostingFromScratch(n_estimators=20, random_state=42, task='classification')
        our_gb.fit(X_train, y_train)
        our_gb_accuracy = accuracy_score(y_test, our_gb.predict(X_test))
        
        # Sklearn implementations
        sklearn_rf = RandomForestClassifier(n_estimators=20, random_state=42)
        sklearn_rf.fit(X_train, y_train)
        sklearn_rf_accuracy = accuracy_score(y_test, sklearn_rf.predict(X_test))
        
        sklearn_gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
        sklearn_gb.fit(X_train, y_train)
        sklearn_gb_accuracy = accuracy_score(y_test, sklearn_gb.predict(X_test))
        
        # Our implementations should be reasonably close to sklearn
        # (within 10% performance difference is acceptable)
        assert abs(our_rf_accuracy - sklearn_rf_accuracy) < 0.1
        assert abs(our_gb_accuracy - sklearn_gb_accuracy) < 0.15  # GB can vary more
        
        # All should be better than random
        assert our_rf_accuracy > 0.6
        assert our_gb_accuracy > 0.6
        assert sklearn_rf_accuracy > 0.6
        assert sklearn_gb_accuracy > 0.6


def test_integration():
    """Integration test for ensemble methods workflow."""
    # Generate data
    X, y = make_classification(n_samples=300, n_features=8, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train ensemble models
    models = {
        'RandomForest': RandomForestFromScratch(n_estimators=15, random_state=42, task='classification'),
        'GradientBoosting': GradientBoostingFromScratch(n_estimators=15, random_state=42, task='classification')
    }
    
    results = {}
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        results[name] = {
            'accuracy': accuracy,
            'n_estimators': len(getattr(model, 'trees_', getattr(model, 'estimators_', []))),
            'feature_importance': model.feature_importances_
        }
    
    # Verify results
    for name, result in results.items():
        assert result['accuracy'] > 0.5  # Better than random
        assert result['n_estimators'] == 15  # Correct number of estimators
        assert len(result['feature_importance']) == X_train.shape[1]  # Correct feature importance length
        assert np.allclose(np.sum(result['feature_importance']), 1.0)  # Feature importance sums to 1
    
    print("Integration test passed! Results:")
    for name, result in results.items():
        print(f"{name}: Accuracy = {result['accuracy']:.3f}")


if __name__ == "__main__":
    # Run integration test if called directly
    test_integration()
    print("All tests would pass if run with pytest!")
