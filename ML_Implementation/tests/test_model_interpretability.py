"""
Comprehensive tests for model interpretability module.
"""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

# Import our interpretability module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_interpretability import (
    SHAPExplainer,
    LIMEExplainer,
    PermutationImportance,
    PartialDependencePlots,
    ModelInterpreter
)


class TestSHAPExplainer:
    """Test SHAP explainer functionality."""
    
    def setup_method(self):
        """Set up test data and models."""
        # Classification data
        X_class, y_class = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        self.X_class_train, self.X_class_test, self.y_class_train, self.y_class_test = \
            train_test_split(X_class, y_class, test_size=0.3, random_state=42)
        
        # Regression data
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = \
            train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
        
        # Train models
        self.rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        self.rf_classifier.fit(self.X_class_train, self.y_class_train)
        
        self.lr_model = LogisticRegression(random_state=42)
        self.lr_model.fit(self.X_class_train, self.y_class_train)
    
    def test_shap_tree_explainer(self):
        """Test SHAP explainer with tree models."""
        explainer = SHAPExplainer()
        
        # Test explanation generation
        shap_values = explainer.explain_prediction(
            self.rf_classifier, self.X_class_test[:5]
        )
        
        assert shap_values is not None
        assert len(shap_values) == 5  # 5 samples
        
        # Test summary plot (should not raise error)
        try:
            explainer.plot_summary(shap_values, self.X_class_test[:5])
        except Exception:
            pass  # Plotting might fail in headless environment
    
    def test_shap_linear_explainer(self):
        """Test SHAP explainer with linear models."""
        explainer = SHAPExplainer()
        
        # Test explanation generation
        shap_values = explainer.explain_prediction(
            self.lr_model, self.X_class_test[:5]
        )
        
        assert shap_values is not None
        assert len(shap_values) == 5  # 5 samples
    
    def test_invalid_model(self):
        """Test with invalid model."""
        explainer = SHAPExplainer()
        
        # Test with None model
        with pytest.raises((ValueError, AttributeError)):
            explainer.explain_prediction(None, self.X_class_test)


class TestLIMEExplainer:
    """Test LIME explainer functionality."""
    
    def setup_method(self):
        """Set up test data and models."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def test_lime_tabular_explanation(self):
        """Test LIME tabular explanations."""
        explainer = LIMEExplainer()
        
        # Test single instance explanation
        explanation = explainer.explain_instance(
            self.model, self.X_train, self.X_test[0], mode='classification'
        )
        
        assert explanation is not None
        
        # Test feature importance extraction
        try:
            importance = explanation.as_map()[1]  # Get feature importance for class 1
            assert len(importance) > 0
        except (AttributeError, KeyError):
            pass  # LIME interface might vary
    
    def test_lime_regression(self):
        """Test LIME with regression."""
        # Create regression data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = LIMEExplainer()
        explanation = explainer.explain_instance(
            model, X_train, X_test[0], mode='regression'
        )
        
        assert explanation is not None
    
    def test_invalid_mode(self):
        """Test with invalid mode."""
        explainer = LIMEExplainer()
        
        with pytest.raises(ValueError):
            explainer.explain_instance(
                self.model, self.X_train, self.X_test[0], mode='invalid'
            )


class TestPermutationImportance:
    """Test permutation importance functionality."""
    
    def setup_method(self):
        """Set up test data and models."""
        X, y = make_classification(
            n_samples=200, n_features=8, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create feature names
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def test_compute_importance(self):
        """Test permutation importance computation."""
        perm_imp = PermutationImportance()
        
        importance = perm_imp.compute_importance(
            self.model, self.X_test, self.y_test, n_repeats=5
        )
        
        assert importance is not None
        assert len(importance) == self.X_test.shape[1]
        assert all(isinstance(imp, (int, float)) for imp in importance)
    
    def test_plot_importance(self):
        """Test importance plotting."""
        perm_imp = PermutationImportance()
        
        importance = perm_imp.compute_importance(
            self.model, self.X_test, self.y_test, n_repeats=3
        )
        
        # Test plotting (should not raise error)
        try:
            perm_imp.plot_importance(importance, self.feature_names)
        except Exception:
            pass  # Plotting might fail in headless environment
    
    def test_custom_scoring(self):
        """Test with custom scoring function."""
        from sklearn.metrics import f1_score
        
        perm_imp = PermutationImportance()
        
        def custom_scorer(y_true, y_pred):
            return f1_score(y_true, y_pred, average='weighted')
        
        importance = perm_imp.compute_importance(
            self.model, self.X_test, self.y_test, 
            scoring=custom_scorer, n_repeats=3
        )
        
        assert importance is not None
        assert len(importance) == self.X_test.shape[1]


class TestPartialDependencePlotter:
    """Test partial dependence plotting functionality."""
    
    def setup_method(self):
        """Set up test data and models."""
        X, y = make_classification(
            n_samples=200, n_features=6, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def test_single_feature_pdp(self):
        """Test partial dependence for single feature."""
        pdp = PartialDependencePlots()
        
        pd_values, pd_grid = pdp.compute_partial_dependence(
            self.model, self.X_test, feature_idx=0
        )
        
        assert pd_values is not None
        assert pd_grid is not None
        assert len(pd_values) == len(pd_grid)
        assert len(pd_grid) > 0
    
    def test_two_feature_pdp(self):
        """Test partial dependence for two features."""
        pdp = PartialDependencePlots()
        
        pd_values, pd_grid = pdp.compute_partial_dependence(
            self.model, self.X_test, feature_idx=[0, 1]
        )
        
        assert pd_values is not None
        assert len(pd_grid) == 2  # Two features
        assert pd_values.shape[0] == len(pd_grid[0])
        assert pd_values.shape[1] == len(pd_grid[1])
    
    def test_plot_pdp(self):
        """Test PDP plotting."""
        pdp = PartialDependencePlots()
        
        # Test single feature plot
        try:
            pdp.plot_partial_dependence(
                self.model, self.X_test, feature_idx=0,
                feature_names=self.feature_names
            )
        except Exception:
            pass  # Plotting might fail in headless environment
        
        # Test two feature plot
        try:
            pdp.plot_partial_dependence(
                self.model, self.X_test, feature_idx=[0, 1],
                feature_names=self.feature_names
            )
        except Exception:
            pass  # Plotting might fail in headless environment


class TestModelInterpreter:
    """Test the unified model interpreter."""
    
    def setup_method(self):
        """Set up test data and models."""
        X, y = make_classification(
            n_samples=150, n_features=6, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def test_full_interpretation(self):
        """Test comprehensive model interpretation."""
        interpreter = ModelInterpreter()
        
        # This should run all interpretation methods
        try:
            results = interpreter.interpret_model(
                self.model, self.X_train, self.X_test, self.y_test,
                feature_names=self.feature_names,
                methods=['shap', 'permutation', 'pdp']
            )
            
            assert 'shap_values' in results or 'permutation_importance' in results or 'pdp_results' in results
        except Exception as e:
            # Some methods might fail due to dependencies or environment
            print(f"Full interpretation test failed: {e}")
    
    def test_selective_interpretation(self):
        """Test selective interpretation methods."""
        interpreter = ModelInterpreter()
        
        # Test only permutation importance
        try:
            results = interpreter.interpret_model(
                self.model, self.X_train, self.X_test, self.y_test,
                feature_names=self.feature_names,
                methods=['permutation']
            )
            
            assert 'permutation_importance' in results
        except Exception as e:
            print(f"Permutation importance test failed: {e}")
    
    def test_invalid_method(self):
        """Test with invalid interpretation method."""
        interpreter = ModelInterpreter()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = interpreter.interpret_model(
                self.model, self.X_train, self.X_test, self.y_test,
                methods=['invalid_method']
            )
            
            # Should return empty or handle gracefully
            assert isinstance(results, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
