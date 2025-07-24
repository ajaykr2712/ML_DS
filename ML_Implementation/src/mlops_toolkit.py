"""
MLOps and Model Deployment Toolkit
==================================

Production-ready ML deployment and monitoring tools:
- Model versioning and registry
- Automated model validation
- Performance monitoring
- Data drift detection
- A/B testing framework
- Model serving infrastructure
- Experiment tracking

Author: ML Arsenal Team
Date: July 2025
"""

import numpy as np
import pandas as pd
import json
import pickle
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt


class ModelRegistry:
    """
    Model Registry for version control and metadata management.
    
    Features:
    - Model versioning with metadata
    - Performance tracking across versions
    - Model comparison utilities
    - Automated model validation
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize Model Registry.
        
        Parameters:
        -----------
        registry_path : str, default="model_registry"
            Path to store model registry files
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        
        # Load existing metadata or create new
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "models": {},
                "experiments": {},
                "created_at": datetime.now().isoformat()
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _compute_model_hash(self, model: Any, X_sample: np.ndarray) -> str:
        """Compute hash of model based on predictions on sample data."""
        predictions = model.predict(X_sample)
        model_str = str(predictions) + str(type(model).__name__)
        return hashlib.md5(model_str.encode()).hexdigest()
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        version: Optional[str] = None,
        X_validation: np.ndarray = None,
        y_validation: np.ndarray = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        author: str = "unknown"
    ) -> str:
        """
        Register a new model version.
        
        Parameters:
        -----------
        model : object
            Trained ML model
        model_name : str
            Name of the model
        version : str, optional
            Version string (auto-generated if None)
        X_validation : ndarray, optional
            Validation features
        y_validation : ndarray, optional
            Validation targets
        metrics : dict, optional
            Performance metrics
        tags : list, optional
            Tags for categorization
        description : str
            Model description
        author : str
            Model author
        
        Returns:
        --------
        version : str
            Version of the registered model
        """
        # Auto-generate version if not provided
        if version is None:
            if model_name not in self.metadata["models"]:
                version = "v1.0.0"
            else:
                versions = list(self.metadata["models"][model_name].keys())
                latest_version = max(versions, key=lambda x: x.split('.'))
                major, minor, patch = map(int, latest_version[1:].split('.'))
                version = f"v{major}.{minor}.{patch + 1}"
        
        # Compute model hash for duplicate detection
        if X_validation is not None:
            model_hash = self._compute_model_hash(model, X_validation[:min(100, len(X_validation))])
        else:
            model_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        
        # Compute validation metrics if data provided
        if X_validation is not None and y_validation is not None and metrics is None:
            predictions = model.predict(X_validation)
            
            # Determine task type
            if len(np.unique(y_validation)) <= 10 and np.all(np.unique(y_validation) == np.unique(y_validation).astype(int)):
                # Classification
                metrics = {
                    "accuracy": accuracy_score(y_validation, predictions),
                    "validation_samples": len(y_validation)
                }
            else:
                # Regression
                metrics = {
                    "r2_score": r2_score(y_validation, predictions),
                    "rmse": np.sqrt(mean_squared_error(y_validation, predictions)),
                    "validation_samples": len(y_validation)
                }
        
        # Save model
        model_file = self.registry_path / f"{model_name}_{version}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Update metadata
        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {}
        
        self.metadata["models"][model_name][version] = {
            "model_hash": model_hash,
            "file_path": str(model_file),
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "tags": tags or [],
            "description": description,
            "author": author,
            "model_type": type(model).__name__
        }
        
        self._save_metadata()
        
        print(f"Model {model_name} version {version} registered successfully!")
        return version
    
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """
        Load a model from registry.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        version : str, default="latest"
            Version to load
        
        Returns:
        --------
        model : object
            Loaded model
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version == "latest":
            version = max(self.metadata["models"][model_name].keys(), 
                         key=lambda x: x.split('.'))
        
        if version not in self.metadata["models"][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        model_file = self.metadata["models"][model_name][version]["file_path"]
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def list_models(self) -> pd.DataFrame:
        """
        List all registered models.
        
        Returns:
        --------
        models_df : DataFrame
            DataFrame with model information
        """
        rows = []
        for model_name, versions in self.metadata["models"].items():
            for version, info in versions.items():
                row = {
                    "model_name": model_name,
                    "version": version,
                    "registered_at": info["registered_at"],
                    "author": info["author"],
                    "model_type": info["model_type"],
                    "description": info["description"]
                }
                # Add metrics
                for metric, value in info["metrics"].items():
                    row[f"metric_{metric}"] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def compare_models(self, model_name: str, metric: str = "accuracy") -> pd.DataFrame:
        """
        Compare different versions of a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        metric : str, default="accuracy"
            Metric to compare
        
        Returns:
        --------
        comparison_df : DataFrame
            Comparison results
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        rows = []
        for version, info in self.metadata["models"][model_name].items():
            if metric in info["metrics"]:
                rows.append({
                    "version": version,
                    "registered_at": info["registered_at"],
                    metric: info["metrics"][metric],
                    "author": info["author"]
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(metric, ascending=False)
        
        return df


class DataDriftDetector:
    """
    Data Drift Detection for monitoring input data changes.
    
    Detects statistical changes in input features that might
    indicate model performance degradation.
    """
    
    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        drift_threshold: float = 0.05
    ):
        """
        Initialize Data Drift Detector.
        
        Parameters:
        -----------
        reference_data : ndarray
            Reference dataset (training data)
        feature_names : list, optional
            Names of features
        drift_threshold : float, default=0.05
            P-value threshold for drift detection
        """
        self.reference_data = np.array(reference_data)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(reference_data.shape[1])]
        self.drift_threshold = drift_threshold
        
        # Compute reference statistics
        self.reference_stats = self._compute_statistics(self.reference_data)
    
    def _compute_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute statistical properties of data."""
        stats_dict = {}
        
        for i in range(data.shape[1]):
            feature_data = data[:, i]
            stats_dict[f"feature_{i}"] = {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "min": np.min(feature_data),
                "max": np.max(feature_data),
                "q25": np.percentile(feature_data, 25),
                "q50": np.percentile(feature_data, 50),
                "q75": np.percentile(feature_data, 75),
                "skewness": stats.skew(feature_data),
                "kurtosis": stats.kurtosis(feature_data)
            }
        
        return stats_dict
    
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in new data compared to reference.
        
        Parameters:
        -----------
        new_data : ndarray
            New data to check for drift
        
        Returns:
        --------
        drift_results : dict
            Drift detection results
        """
        new_data = np.array(new_data)
        drift_results = {
            "overall_drift": False,
            "drifted_features": [],
            "feature_results": {},
            "drift_score": 0.0
        }
        
        n_features = min(self.reference_data.shape[1], new_data.shape[1])
        significant_features = 0
        
        for i in range(n_features):
            feature_name = self.feature_names[i]
            ref_feature = self.reference_data[:, i]
            new_feature = new_data[:, i]
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_feature, new_feature)
            
            # Chi-square test for distributions (discretized)
            try:
                # Create bins for continuous data
                combined_data = np.concatenate([ref_feature, new_feature])
                bins = np.histogram_bin_edges(combined_data, bins=10)
                
                ref_hist, _ = np.histogram(ref_feature, bins=bins)
                new_hist, _ = np.histogram(new_feature, bins=bins)
                
                # Add small constant to avoid zero counts
                ref_hist = ref_hist + 1
                new_hist = new_hist + 1
                
                chi2_stat, chi2_pvalue = stats.chisquare(new_hist, ref_hist)
            except Exception:
                chi2_stat, chi2_pvalue = np.nan, 1.0
            
            # Mann-Whitney U test
            try:
                u_stat, u_pvalue = stats.mannwhitneyu(ref_feature, new_feature, alternative='two-sided')
            except Exception:
                u_stat, u_pvalue = np.nan, 1.0
            
            # Determine if feature has drifted
            feature_drifted = (ks_pvalue < self.drift_threshold or 
                             chi2_pvalue < self.drift_threshold or 
                             u_pvalue < self.drift_threshold)
            
            if feature_drifted:
                significant_features += 1
                drift_results["drifted_features"].append(feature_name)
            
            drift_results["feature_results"][feature_name] = {
                "drifted": feature_drifted,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "chi2_statistic": chi2_stat,
                "chi2_pvalue": chi2_pvalue,
                "mannwhitney_statistic": u_stat,
                "mannwhitney_pvalue": u_pvalue,
                "mean_shift": np.mean(new_feature) - np.mean(ref_feature),
                "std_ratio": np.std(new_feature) / (np.std(ref_feature) + 1e-8)
            }
        
        # Overall drift assessment
        drift_results["drift_score"] = significant_features / n_features
        drift_results["overall_drift"] = drift_results["drift_score"] > 0.2  # 20% threshold
        
        return drift_results
    
    def plot_feature_comparison(self, new_data: np.ndarray, feature_idx: int = 0,
                               figsize: Tuple[int, int] = (12, 5)):
        """
        Plot comparison between reference and new data for a feature.
        
        Parameters:
        -----------
        new_data : ndarray
            New data
        feature_idx : int, default=0
            Index of feature to plot
        figsize : tuple, default=(12, 5)
            Figure size
        """
        ref_feature = self.reference_data[:, feature_idx]
        new_feature = new_data[:, feature_idx]
        feature_name = self.feature_names[feature_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram comparison
        axes[0].hist(ref_feature, alpha=0.7, label='Reference', bins=30, density=True)
        axes[0].hist(new_feature, alpha=0.7, label='New Data', bins=30, density=True)
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Distribution Comparison: {feature_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1].boxplot([ref_feature, new_feature], labels=['Reference', 'New Data'])
        axes[1].set_ylabel(feature_name)
        axes[1].set_title(f'Box Plot Comparison: {feature_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class ModelMonitor:
    """
    Real-time model performance monitoring.
    
    Tracks model predictions and performance metrics over time,
    alerting when performance degrades.
    """
    
    def __init__(
        self,
        model: Any,
        model_name: str,
        task: str = 'classification',
        performance_threshold: float = 0.05,
        window_size: int = 100
    ):
        """
        Initialize Model Monitor.
        
        Parameters:
        -----------
        model : object
            Model to monitor
        model_name : str
            Name of the model
        task : str, default='classification'
            'classification' or 'regression'
        performance_threshold : float, default=0.05
            Threshold for performance degradation alert
        window_size : int, default=100
            Window size for rolling metrics
        """
        self.model = model
        self.model_name = model_name
        self.task = task
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        # Initialize monitoring data
        self.predictions_log = []
        self.performance_log = []
        self.timestamps = []
        self.baseline_performance = None
    
    def log_prediction(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Log a prediction and optionally compute performance.
        
        Parameters:
        -----------
        X : ndarray
            Input features
        y_true : ndarray, optional
            True labels (if available)
        
        Returns:
        --------
        predictions : ndarray
            Model predictions
        """
        predictions = self.model.predict(X)
        
        # Log prediction
        prediction_entry = {
            "timestamp": datetime.now(),
            "predictions": predictions.copy(),
            "input_shape": X.shape,
            "has_ground_truth": y_true is not None
        }
        
        if y_true is not None:
            prediction_entry["y_true"] = y_true.copy()
            
            # Compute performance
            if self.task == 'classification':
                performance = accuracy_score(y_true, predictions)
            else:
                performance = r2_score(y_true, predictions)
            
            prediction_entry["performance"] = performance
            self.performance_log.append(performance)
            
            # Set baseline if first measurement
            if self.baseline_performance is None:
                self.baseline_performance = performance
            
            # Check for performance degradation
            self._check_performance_alert(performance)
        
        self.predictions_log.append(prediction_entry)
        self.timestamps.append(datetime.now())
        
        return predictions
    
    def _check_performance_alert(self, current_performance: float):
        """Check if performance has degraded significantly."""
        if self.baseline_performance is None:
            return
        
        performance_drop = self.baseline_performance - current_performance
        
        if performance_drop > self.performance_threshold:
            print(f"⚠️  PERFORMANCE ALERT for {self.model_name}:")
            print(f"   Performance dropped by {performance_drop:.3f}")
            print(f"   Baseline: {self.baseline_performance:.3f}")
            print(f"   Current: {current_performance:.3f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
        --------
        summary : dict
            Performance summary
        """
        if not self.performance_log:
            return {"error": "No performance data available"}
        
        recent_performance = self.performance_log[-self.window_size:]
        
        summary = {
            "total_predictions": len(self.predictions_log),
            "total_with_ground_truth": len(self.performance_log),
            "baseline_performance": self.baseline_performance,
            "latest_performance": self.performance_log[-1],
            "recent_mean_performance": np.mean(recent_performance),
            "recent_std_performance": np.std(recent_performance),
            "performance_trend": "stable"
        }
        
        # Determine trend
        if len(self.performance_log) >= 20:
            early_performance = np.mean(self.performance_log[:10])
            recent_performance_mean = np.mean(self.performance_log[-10:])
            
            if recent_performance_mean > early_performance + 0.02:
                summary["performance_trend"] = "improving"
            elif recent_performance_mean < early_performance - 0.02:
                summary["performance_trend"] = "declining"
        
        return summary
    
    def plot_performance_over_time(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot performance metrics over time.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.performance_log:
            print("No performance data to plot")
            return
        
        # Extract timestamps for performance data
        perf_timestamps = [entry["timestamp"] for entry in self.predictions_log 
                          if "performance" in entry]
        
        plt.figure(figsize=figsize)
        
        # Plot performance over time
        plt.plot(perf_timestamps, self.performance_log, 'b-', linewidth=2, label='Performance')
        
        # Add baseline line
        if self.baseline_performance is not None:
            plt.axhline(y=self.baseline_performance, color='g', linestyle='--', 
                       label=f'Baseline ({self.baseline_performance:.3f})')
        
        # Add rolling mean
        if len(self.performance_log) >= 10:
            window = min(10, len(self.performance_log) // 2)
            rolling_mean = pd.Series(self.performance_log).rolling(window=window).mean()
            plt.plot(perf_timestamps, rolling_mean, 'r--', linewidth=2, 
                    label=f'Rolling Mean (window={window})')
        
        plt.xlabel('Time')
        plt.ylabel('Performance Score')
        plt.title(f'Model Performance Over Time: {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class ABTestFramework:
    """
    A/B Testing framework for model comparison.
    
    Enables controlled experiments to compare model performance
    with statistical significance testing.
    """
    
    def __init__(
        self,
        model_a: Any,
        model_b: Any,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        task: str = 'classification'
    ):
        """
        Initialize A/B Test Framework.
        
        Parameters:
        -----------
        model_a : object
            First model (control)
        model_b : object
            Second model (treatment)
        model_a_name : str, default="Model A"
            Name of first model
        model_b_name : str, default="Model B"
            Name of second model
        task : str, default='classification'
            'classification' or 'regression'
        """
        self.model_a = model_a
        self.model_b = model_b
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.task = task
        
        # Experiment data
        self.experiment_data = {
            "model_a_results": [],
            "model_b_results": [],
            "assignments": [],
            "timestamps": []
        }
    
    def run_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        traffic_split: float = 0.5,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run A/B test experiment.
        
        Parameters:
        -----------
        X : ndarray
            Input features
        y : ndarray
            True labels
        traffic_split : float, default=0.5
            Fraction of traffic for model A
        random_state : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        results : dict
            Experiment results
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        
        # Random assignment to groups
        assignments = np.random.random(n_samples) < traffic_split
        
        # Split data
        X_a = X[assignments]
        y_a = y[assignments]
        X_b = X[~assignments]
        y_b = y[~assignments]
        
        # Get predictions
        pred_a = self.model_a.predict(X_a)
        pred_b = self.model_b.predict(X_b)
        
        # Compute performance
        if self.task == 'classification':
            perf_a = accuracy_score(y_a, pred_a)
            perf_b = accuracy_score(y_b, pred_b)
            metric_name = 'accuracy'
        else:
            perf_a = r2_score(y_a, pred_a)
            perf_b = r2_score(y_b, pred_b)
            metric_name = 'r2_score'
        
        # Statistical significance test
        # For classification, use chi-square test
        if self.task == 'classification':
            # Convert to success/failure counts
            successes_a = np.sum(pred_a == y_a)
            failures_a = len(y_a) - successes_a
            successes_b = np.sum(pred_b == y_b)
            failures_b = len(y_b) - successes_b
            
            # Chi-square test
            observed = np.array([[successes_a, failures_a], [successes_b, failures_b]])
            try:
                chi2_stat, p_value = stats.chi2_contingency(observed)[:2]
            except Exception:
                p_value = 1.0
        else:
            # For regression, use t-test on residuals
            residuals_a = np.abs(y_a - pred_a)
            residuals_b = np.abs(y_b - pred_b)
            
            try:
                _, p_value = stats.ttest_ind(residuals_a, residuals_b)
            except Exception:
                p_value = 1.0
        
        # Store experiment data
        self.experiment_data["model_a_results"].extend([perf_a] * len(y_a))
        self.experiment_data["model_b_results"].extend([perf_b] * len(y_b))
        self.experiment_data["assignments"].extend(assignments.tolist())
        self.experiment_data["timestamps"].extend([datetime.now()] * n_samples)
        
        results = {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "model_a_performance": perf_a,
            "model_b_performance": perf_b,
            "performance_difference": perf_b - perf_a,
            "sample_size_a": len(y_a),
            "sample_size_b": len(y_b),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "metric": metric_name,
            "traffic_split": traffic_split,
            "winner": self.model_b_name if perf_b > perf_a else self.model_a_name
        }
        
        return results
    
    def plot_experiment_results(self, results: Dict[str, Any], figsize: Tuple[int, int] = (10, 6)):
        """
        Plot A/B test results.
        
        Parameters:
        -----------
        results : dict
            Experiment results from run_experiment
        figsize : tuple, default=(10, 6)
            Figure size
        """
        models = [results["model_a_name"], results["model_b_name"]]
        performances = [results["model_a_performance"], results["model_b_performance"]]
        sample_sizes = [results["sample_size_a"], results["sample_size_b"]]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Performance comparison
        bars = ax1.bar(models, performances, color=['blue', 'red'], alpha=0.7)
        ax1.set_ylabel(f'{results["metric"].title()}')
        ax1.set_title('Model Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, perf in zip(bars, performances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{perf:.3f}', ha='center', va='bottom')
        
        # Sample size comparison
        ax2.bar(models, sample_sizes, color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('Sample Size')
        ax2.set_title('Sample Size Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add significance indicator
        significance_text = "Statistically Significant" if results["significant"] else "Not Significant"
        fig.suptitle(f'A/B Test Results: {significance_text} (p={results["p_value"]:.4f})')
        
        plt.tight_layout()
        plt.show()


class MLOpsToolkit:
    """
    Unified MLOps toolkit combining all components.
    
    This class provides a high-level interface to all MLOps functionality
    including model registry, monitoring, drift detection, and A/B testing.
    """
    
    def __init__(self, registry_path: str = "./models"):
        """
        Initialize MLOps toolkit.
        
        Args:
            registry_path: Path for model registry storage
        """
        self.registry = ModelRegistry(base_path=registry_path)
        self.monitor = ModelMonitor()
        self.drift_detector = DataDriftDetector()
        self.ab_framework = ABTestFramework()
        self.deployed_models = {}
    
    def deploy_model(self, model, name: str, version: str, metadata: dict = None):
        """Deploy a model to the registry."""
        model_id = self.registry.register_model(model, name, version, metadata or {})
        self.deployed_models[model_id] = model
        return model_id
    
    def serve_model(self, model_id: str, X):
        """Make predictions using a deployed model."""
        if model_id in self.deployed_models:
            model = self.deployed_models[model_id]
            return model.predict(X)
        else:
            model, _ = self.registry.load_model(model_id)
            self.deployed_models[model_id] = model
            return model.predict(X)
    
    def monitor_prediction(self, model_id: str, prediction_id: str, features, prediction, actual=None):
        """Log a prediction for monitoring."""
        self.monitor.log_prediction(prediction_id, features, prediction, 0.8, actual)
    
    def get_model_metrics(self, model_id: str):
        """Get performance metrics for a model."""
        return self.monitor.get_performance_metrics()
    
    def detect_data_drift(self, model_id: str, reference_data, new_data):
        """Detect data drift for a model."""
        drift_detected, _ = self.drift_detector.detect_drift(reference_data, new_data)
        return drift_detected
    
    def start_ab_test(self, name: str, model_a_id: str, model_b_id: str, traffic_split: float = 0.5):
        """Start an A/B test between two models."""
        model_a = self.deployed_models.get(model_a_id)
        model_b = self.deployed_models.get(model_b_id)
        
        if not model_a:
            model_a, _ = self.registry.load_model(model_a_id)
        if not model_b:
            model_b, _ = self.registry.load_model(model_b_id)
        
        return self.ab_framework.create_experiment(name, model_a, model_b, traffic_split)
    
    def predict_with_ab_test(self, experiment_id: str, user_id: str, features):
        """Make prediction using A/B test routing."""
        model = self.ab_framework.route_traffic(experiment_id, user_id)
        return model.predict([features])[0]


# Example usage and comprehensive testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    print("="*60)
    print("MLOps and Model Deployment Toolkit Demo")
    print("="*60)
    
    # 1. Model Registry Demo
    print("\n1. Model Registry Demo:")
    
    # Generate sample data
    X_clf, y_clf = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Initialize registry
    registry = ModelRegistry("demo_registry")
    
    # Register models
    rf_version = registry.register_model(
        rf_model, "credit_risk_model", 
        X_validation=X_test, y_validation=y_test,
        description="Random Forest for credit risk prediction",
        author="Data Science Team"
    )
    
    lr_version = registry.register_model(
        lr_model, "credit_risk_model",
        X_validation=X_test, y_validation=y_test,
        description="Logistic Regression baseline",
        author="Data Science Team"
    )
    
    # List models
    models_df = registry.list_models()
    print(f"Registered {len(models_df)} model versions")
    
    # Compare models
    comparison = registry.compare_models("credit_risk_model", "accuracy")
    print(f"Best model version: {comparison.iloc[0]['version']} with accuracy: {comparison.iloc[0]['accuracy']:.3f}")
    
    # 2. Data Drift Detection Demo
    print("\n2. Data Drift Detection Demo:")
    
    # Create reference and new data (with drift)
    reference_data = X_train
    new_data = X_test + np.random.normal(0, 0.1, X_test.shape)  # Add some drift
    
    drift_detector = DataDriftDetector(reference_data)
    drift_results = drift_detector.detect_drift(new_data)
    
    print(f"Overall drift detected: {drift_results['overall_drift']}")
    print(f"Drift score: {drift_results['drift_score']:.3f}")
    print(f"Drifted features: {len(drift_results['drifted_features'])}")
    
    # 3. Model Monitoring Demo
    print("\n3. Model Monitoring Demo:")
    
    monitor = ModelMonitor(rf_model, "credit_risk_model", task='classification')
    
    # Simulate predictions over time
    for i in range(5):
        batch_X = X_test[i*20:(i+1)*20]
        batch_y = y_test[i*20:(i+1)*20]
        
        # Add some noise to simulate performance degradation
        if i >= 3:
            batch_y = np.random.choice([0, 1], size=len(batch_y))  # Random labels
        
        predictions = monitor.log_prediction(batch_X, batch_y)
        time.sleep(0.1)  # Simulate time passing
    
    performance_summary = monitor.get_performance_summary()
    print(f"Total predictions logged: {performance_summary['total_predictions']}")
    print(f"Performance trend: {performance_summary['performance_trend']}")
    
    # 4. A/B Testing Demo
    print("\n4. A/B Testing Demo:")
    
    ab_test = ABTestFramework(
        lr_model, rf_model,
        "Logistic Regression", "Random Forest",
        task='classification'
    )
    
    ab_results = ab_test.run_experiment(X_test, y_test, traffic_split=0.5, random_state=42)
    
    print(f"Winner: {ab_results['winner']}")
    print(f"Performance difference: {ab_results['performance_difference']:.3f}")
    print(f"Statistically significant: {ab_results['significant']}")
    
    print("\nAll MLOps demos completed successfully!")
