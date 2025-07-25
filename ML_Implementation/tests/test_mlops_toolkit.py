"""
Comprehensive tests for MLOps toolkit module.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import our MLOps module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlops_toolkit import (
    ModelRegistry,
    DataDriftDetector,
    ModelMonitor,
    ABTestFramework,
    MLOpsToolkit
)


class TestModelRegistry:
    """Test model registry functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(base_path=self.temp_dir)
        
        # Create a test model
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.metadata = {
            "accuracy": 0.95,
            "dataset": "test_data",
            "features": ["f1", "f2", "f3", "f4", "f5"]
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_register_model(self):
        """Test model registration."""
        model_id = self.registry.register_model(
            self.model, "test_model", "1.0", self.metadata
        )
        
        assert model_id is not None
        assert isinstance(model_id, str)
        
        # Check if model file exists
        model_path = os.path.join(self.temp_dir, f"{model_id}.pkl")
        assert os.path.exists(model_path)
        
        # Check if metadata file exists
        metadata_path = os.path.join(self.temp_dir, f"{model_id}_metadata.json")
        assert os.path.exists(metadata_path)
    
    def test_load_model(self):
        """Test model loading."""
        model_id = self.registry.register_model(
            self.model, "test_model", "1.0", self.metadata
        )
        
        loaded_model, loaded_metadata = self.registry.load_model(model_id)
        
        assert loaded_model is not None
        assert loaded_metadata is not None
        assert loaded_metadata["accuracy"] == 0.95
        assert loaded_metadata["dataset"] == "test_data"
    
    def test_list_models(self):
        """Test listing models."""
        # Register multiple models
        model_id1 = self.registry.register_model(
            self.model, "model1", "1.0", self.metadata
        )
        model_id2 = self.registry.register_model(
            self.model, "model2", "1.0", self.metadata
        )
        
        models = self.registry.list_models()
        
        assert len(models) >= 2
        model_ids = [model["model_id"] for model in models]
        assert model_id1 in model_ids
        assert model_id2 in model_ids
    
    def test_delete_model(self):
        """Test model deletion."""
        model_id = self.registry.register_model(
            self.model, "test_model", "1.0", self.metadata
        )
        
        # Verify model exists
        loaded_model, _ = self.registry.load_model(model_id)
        assert loaded_model is not None
        
        # Delete model
        success = self.registry.delete_model(model_id)
        assert success
        
        # Verify model is deleted
        with pytest.raises(FileNotFoundError):
            self.registry.load_model(model_id)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            self.registry.load_model("nonexistent_id")


class TestDataDriftDetector:
    """Test data drift detection functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.detector = DataDriftDetector()
        
        # Create reference and new data
        self.reference_data, _ = make_classification(
            n_samples=200, n_features=5, random_state=42
        )
        
        # Create slightly drifted data
        self.drifted_data, _ = make_classification(
            n_samples=200, n_features=5, random_state=123  # Different seed
        )
        
        # Create heavily drifted data
        self.heavy_drift_data = self.reference_data + 2.0  # Add constant
    
    def test_ks_drift_detection(self):
        """Test KS drift detection."""
        # Test no drift (same data)
        drift_detected, p_values = self.detector.detect_drift(
            self.reference_data, self.reference_data, method='ks'
        )
        
        assert not drift_detected
        assert all(p > 0.05 for p in p_values)  # High p-values indicate no drift
        
        # Test drift detection
        drift_detected, p_values = self.detector.detect_drift(
            self.reference_data, self.heavy_drift_data, method='ks'
        )
        
        assert drift_detected
        assert any(p < 0.05 for p in p_values)  # Low p-values indicate drift
    
    def test_psi_drift_detection(self):
        """Test PSI drift detection."""
        drift_detected, psi_values = self.detector.detect_drift(
            self.reference_data, self.reference_data, method='psi'
        )
        
        assert not drift_detected
        assert all(psi < 0.1 for psi in psi_values)  # Low PSI indicates no drift
        
        # Test with drifted data
        drift_detected, psi_values = self.detector.detect_drift(
            self.reference_data, self.heavy_drift_data, method='psi'
        )
        
        # Heavy drift should be detected
        assert any(psi > 0.1 for psi in psi_values)
    
    def test_invalid_method(self):
        """Test invalid drift detection method."""
        with pytest.raises(ValueError):
            self.detector.detect_drift(
                self.reference_data, self.drifted_data, method='invalid'
            )
    
    def test_mismatched_features(self):
        """Test with mismatched number of features."""
        data_3_features, _ = make_classification(
            n_samples=100, n_features=3, random_state=42
        )
        
        with pytest.raises(ValueError):
            self.detector.detect_drift(
                self.reference_data, data_3_features
            )


class TestModelMonitor:
    """Test model monitoring functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.monitor = ModelMonitor()
        
        # Create test data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model and get predictions
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)
        self.y_proba = self.model.predict_proba(X_test)[:, 1]
    
    def test_log_prediction(self):
        """Test prediction logging."""
        # Log multiple predictions
        for i in range(10):
            self.monitor.log_prediction(
                prediction_id=f"pred_{i}",
                features=list(range(5)),
                prediction=int(self.y_pred[i]),
                confidence=float(self.y_proba[i]),
                actual=int(self.y_test[i]) if i < len(self.y_test) else None
            )
        
        assert len(self.monitor.prediction_log) == 10
        
        # Check log structure
        first_log = self.monitor.prediction_log[0]
        assert "prediction_id" in first_log
        assert "features" in first_log
        assert "prediction" in first_log
        assert "confidence" in first_log
        assert "timestamp" in first_log
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Log predictions with actuals
        for i in range(min(10, len(self.y_test))):
            self.monitor.log_prediction(
                prediction_id=f"pred_{i}",
                features=list(range(5)),
                prediction=int(self.y_pred[i]),
                confidence=float(self.y_proba[i]),
                actual=int(self.y_test[i])
            )
        
        metrics = self.monitor.get_performance_metrics()
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # Check metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
    
    def test_empty_log_metrics(self):
        """Test metrics with empty log."""
        metrics = self.monitor.get_performance_metrics()
        
        # Should return None or empty dict when no data
        assert metrics is None or len(metrics) == 0
    
    def test_detect_performance_degradation(self):
        """Test performance degradation detection."""
        # Log good predictions first
        for i in range(5):
            self.monitor.log_prediction(
                prediction_id=f"pred_{i}",
                features=list(range(5)),
                prediction=int(self.y_test[i]),  # Correct predictions
                confidence=0.9,
                actual=int(self.y_test[i])
            )
        
        # Log bad predictions
        for i in range(5, 10):
            if i < len(self.y_test):
                self.monitor.log_prediction(
                    prediction_id=f"pred_{i}",
                    features=list(range(5)),
                    prediction=1 - int(self.y_test[i]),  # Wrong predictions
                    confidence=0.9,
                    actual=int(self.y_test[i])
                )
        
        # Check for degradation
        is_degraded = self.monitor.detect_performance_degradation(
            metric='accuracy', threshold=0.8, window_size=5
        )
        
        # Recent window should show poor performance
        assert is_degraded


class TestABTestFramework:
    """Test A/B testing framework."""
    
    def setup_method(self):
        """Set up test environment."""
        self.ab_test = ABTestFramework()
        
        # Create mock models
        self.model_a = Mock()
        self.model_a.predict.return_value = [0, 1, 0, 1, 0]
        self.model_a.predict_proba.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]
        
        self.model_b = Mock()
        self.model_b.predict.return_value = [0, 1, 1, 1, 0]
        self.model_b.predict_proba.return_value = [[0.7, 0.3], [0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.8, 0.2]]
    
    def test_create_experiment(self):
        """Test experiment creation."""
        exp_id = self.ab_test.create_experiment(
            name="test_experiment",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.5
        )
        
        assert exp_id is not None
        assert exp_id in self.ab_test.experiments
        
        experiment = self.ab_test.experiments[exp_id]
        assert experiment["name"] == "test_experiment"
        assert experiment["traffic_split"] == 0.5
        assert experiment["status"] == "active"
    
    def test_route_traffic(self):
        """Test traffic routing."""
        exp_id = self.ab_test.create_experiment(
            name="test_experiment",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.5
        )
        
        # Test multiple routings
        routes = []
        for i in range(100):
            user_id = f"user_{i}"
            model = self.ab_test.route_traffic(exp_id, user_id)
            routes.append(model)
        
        # Should have both models
        assert self.model_a in routes
        assert self.model_b in routes
        
        # Check roughly even split (with some tolerance)
        a_count = routes.count(self.model_a)
        b_count = routes.count(self.model_b)
        assert 30 <= a_count <= 70  # Allow some variance
        assert 30 <= b_count <= 70
    
    def test_log_result(self):
        """Test result logging."""
        exp_id = self.ab_test.create_experiment(
            name="test_experiment",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.5
        )
        
        # Log results
        self.ab_test.log_result(exp_id, "user_1", "A", 1, 0.8, 1)
        self.ab_test.log_result(exp_id, "user_2", "B", 0, 0.3, 0)
        
        results = self.ab_test.experiments[exp_id]["results"]
        assert len(results) == 2
        
        # Check result structure
        assert results[0]["user_id"] == "user_1"
        assert results[0]["variant"] == "A"
        assert results[0]["prediction"] == 1
    
    def test_analyze_results(self):
        """Test result analysis."""
        exp_id = self.ab_test.create_experiment(
            name="test_experiment",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.5
        )
        
        # Log some results
        for i in range(20):
            variant = "A" if i % 2 == 0 else "B"
            # Simulate A being better
            correct = 1 if (variant == "A" and i % 3 != 0) or (variant == "B" and i % 4 == 0) else 0
            self.ab_test.log_result(exp_id, f"user_{i}", variant, correct, 0.7, correct)
        
        analysis = self.ab_test.analyze_results(exp_id)
        
        assert "variant_A" in analysis
        assert "variant_B" in analysis
        assert "statistical_significance" in analysis
        
        # Check metrics exist
        assert "conversion_rate" in analysis["variant_A"]
        assert "sample_size" in analysis["variant_A"]
    
    def test_stop_experiment(self):
        """Test experiment stopping."""
        exp_id = self.ab_test.create_experiment(
            name="test_experiment",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.5
        )
        
        # Stop experiment
        self.ab_test.stop_experiment(exp_id)
        
        experiment = self.ab_test.experiments[exp_id]
        assert experiment["status"] == "stopped"
    
    def test_nonexistent_experiment(self):
        """Test operations on non-existent experiment."""
        with pytest.raises(KeyError):
            self.ab_test.route_traffic("nonexistent", "user_1")
        
        with pytest.raises(KeyError):
            self.ab_test.log_result("nonexistent", "user_1", "A", 1, 0.8, 1)


class TestMLOpsToolkit:
    """Test the unified MLOps toolkit."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.toolkit = MLOpsToolkit(registry_path=self.temp_dir)
        
        # Create test model and data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deploy_model(self):
        """Test model deployment."""
        model_id = self.toolkit.deploy_model(
            model=self.model,
            name="test_model",
            version="1.0",
            metadata={"accuracy": 0.95}
        )
        
        assert model_id is not None
        
        # Test model serving
        predictions = self.toolkit.serve_model(model_id, self.X_test[:5])
        assert len(predictions) == 5
        assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    def test_monitor_model(self):
        """Test model monitoring."""
        model_id = self.toolkit.deploy_model(
            model=self.model,
            name="test_model",
            version="1.0",
            metadata={"accuracy": 0.95}
        )
        
        # Make predictions and monitor
        y_pred = self.model.predict(self.X_test)
        
        for i in range(min(10, len(self.X_test))):
            self.toolkit.monitor_prediction(
                model_id=model_id,
                prediction_id=f"pred_{i}",
                features=self.X_test[i].tolist(),
                prediction=int(y_pred[i]),
                actual=int(self.y_test[i])
            )
        
        # Get monitoring metrics
        metrics = self.toolkit.get_model_metrics(model_id)
        assert metrics is not None
        if metrics:  # If metrics were calculated
            assert "accuracy" in metrics
    
    def test_detect_drift(self):
        """Test drift detection."""
        model_id = self.toolkit.deploy_model(
            model=self.model,
            name="test_model",
            version="1.0",
            metadata={"accuracy": 0.95}
        )
        
        # Test drift detection
        drift_detected = self.toolkit.detect_data_drift(
            model_id, self.X_train, self.X_test
        )
        
        assert isinstance(drift_detected, bool)
    
    def test_ab_test_integration(self):
        """Test A/B testing integration."""
        # Deploy two models
        model_id_a = self.toolkit.deploy_model(
            model=self.model,
            name="model_a",
            version="1.0",
            metadata={"accuracy": 0.95}
        )
        
        # Create a slightly different model
        model_b = RandomForestClassifier(n_estimators=15, random_state=123)
        model_b.fit(self.X_train, self.y_train)
        
        model_id_b = self.toolkit.deploy_model(
            model=model_b,
            name="model_b",
            version="1.0",
            metadata={"accuracy": 0.93}
        )
        
        # Start A/B test
        exp_id = self.toolkit.start_ab_test(
            name="model_comparison",
            model_a_id=model_id_a,
            model_b_id=model_id_b,
            traffic_split=0.5
        )
        
        assert exp_id is not None
        
        # Test traffic routing
        for i in range(10):
            user_id = f"user_{i}"
            prediction = self.toolkit.predict_with_ab_test(
                exp_id, user_id, self.X_test[i % len(self.X_test)]
            )
            assert prediction is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
