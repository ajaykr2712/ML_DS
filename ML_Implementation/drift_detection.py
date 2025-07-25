# Model Drift Detection and Monitoring System
# ===========================================

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    statistical_test: str = "ks_test"  # ks_test, chi2_test, psi
    significance_level: float = 0.05
    window_size: int = 1000
    monitoring_frequency: str = "daily"
    alert_threshold: float = 0.1
    
class StatisticalDriftDetector:
    """Statistical drift detection using various tests."""
    
    def __init__(self, config: DriftConfig):
        self.config = config
        self.reference_data = None
        self.drift_history = []
        self.logger = logging.getLogger(__name__)
    
    def set_reference_data(self, data: np.ndarray):
        """Set reference data for drift detection."""
        self.reference_data = data
        self.logger.info(f"Reference data set with {len(data)} samples")
    
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect drift between reference and current data."""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        drift_results = {}
        
        if self.config.statistical_test == "ks_test":
            drift_results = self._kolmogorov_smirnov_test(current_data)
        elif self.config.statistical_test == "chi2_test":
            drift_results = self._chi_square_test(current_data)
        elif self.config.statistical_test == "psi":
            drift_results = self._population_stability_index(current_data)
        
        # Add metadata
        drift_results.update({
            'timestamp': datetime.now(),
            'reference_size': len(self.reference_data),
            'current_size': len(current_data),
            'test_method': self.config.statistical_test
        })
        
        self.drift_history.append(drift_results)
        return drift_results
    
    def _kolmogorov_smirnov_test(self, current_data: np.ndarray) -> Dict:
        """Perform Kolmogorov-Smirnov test for drift detection."""
        if len(self.reference_data.shape) == 1:
            # Univariate case
            statistic, p_value = stats.ks_2samp(self.reference_data, current_data)
            drift_detected = p_value < self.config.significance_level
            
            return {
                'drift_detected': drift_detected,
                'drift_score': statistic,
                'p_value': p_value,
                'interpretation': 'Drift detected' if drift_detected else 'No drift detected'
            }
        else:
            # Multivariate case - test each feature
            feature_results = []
            overall_drift = False
            
            for i in range(self.reference_data.shape[1]):
                ref_feature = self.reference_data[:, i]
                curr_feature = current_data[:, i]
                
                statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
                feature_drift = p_value < self.config.significance_level
                
                if feature_drift:
                    overall_drift = True
                
                feature_results.append({
                    'feature_index': i,
                    'drift_detected': feature_drift,
                    'drift_score': statistic,
                    'p_value': p_value
                })
            
            return {
                'drift_detected': overall_drift,
                'feature_results': feature_results,
                'num_drifted_features': sum(1 for r in feature_results if r['drift_detected'])
            }
    
    def _chi_square_test(self, current_data: np.ndarray) -> Dict:
        """Perform Chi-square test for categorical drift detection."""
        # Discretize continuous data into bins
        n_bins = 10
        
        if len(self.reference_data.shape) == 1:
            # Create bins based on reference data
            bin_edges = np.percentile(self.reference_data, np.linspace(0, 100, n_bins + 1))
            
            # Count observations in each bin
            ref_counts, _ = np.histogram(self.reference_data, bins=bin_edges)
            curr_counts, _ = np.histogram(current_data, bins=bin_edges)
            
            # Avoid zero counts
            ref_counts = ref_counts + 1
            curr_counts = curr_counts + 1
            
            # Perform chi-square test
            statistic, p_value = stats.chisquare(curr_counts, ref_counts)
            drift_detected = p_value < self.config.significance_level
            
            return {
                'drift_detected': drift_detected,
                'drift_score': statistic,
                'p_value': p_value,
                'interpretation': 'Drift detected' if drift_detected else 'No drift detected'
            }
        else:
            # Multivariate case
            feature_results = []
            overall_drift = False
            
            for i in range(self.reference_data.shape[1]):
                ref_feature = self.reference_data[:, i]
                curr_feature = current_data[:, i]
                
                bin_edges = np.percentile(ref_feature, np.linspace(0, 100, n_bins + 1))
                ref_counts, _ = np.histogram(ref_feature, bins=bin_edges)
                curr_counts, _ = np.histogram(curr_feature, bins=bin_edges)
                
                ref_counts = ref_counts + 1
                curr_counts = curr_counts + 1
                
                statistic, p_value = stats.chisquare(curr_counts, ref_counts)
                feature_drift = p_value < self.config.significance_level
                
                if feature_drift:
                    overall_drift = True
                
                feature_results.append({
                    'feature_index': i,
                    'drift_detected': feature_drift,
                    'drift_score': statistic,
                    'p_value': p_value
                })
            
            return {
                'drift_detected': overall_drift,
                'feature_results': feature_results,
                'num_drifted_features': sum(1 for r in feature_results if r['drift_detected'])
            }
    
    def _population_stability_index(self, current_data: np.ndarray) -> Dict:
        """Calculate Population Stability Index (PSI)."""
        n_bins = 10
        
        if len(self.reference_data.shape) == 1:
            # Create bins based on reference data quantiles
            bin_edges = np.percentile(self.reference_data, np.linspace(0, 100, n_bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate proportions
            ref_proportions = self._get_bin_proportions(self.reference_data, bin_edges)
            curr_proportions = self._get_bin_proportions(current_data, bin_edges)
            
            # Calculate PSI
            psi = self._calculate_psi(ref_proportions, curr_proportions)
            drift_detected = psi > self.config.alert_threshold
            
            return {
                'drift_detected': drift_detected,
                'drift_score': psi,
                'interpretation': self._interpret_psi(psi)
            }
        else:
            # Multivariate case
            feature_results = []
            overall_psi = 0.0
            
            for i in range(self.reference_data.shape[1]):
                ref_feature = self.reference_data[:, i]
                curr_feature = current_data[:, i]
                
                bin_edges = np.percentile(ref_feature, np.linspace(0, 100, n_bins + 1))
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                
                ref_proportions = self._get_bin_proportions(ref_feature, bin_edges)
                curr_proportions = self._get_bin_proportions(curr_feature, bin_edges)
                
                feature_psi = self._calculate_psi(ref_proportions, curr_proportions)
                overall_psi += feature_psi
                
                feature_results.append({
                    'feature_index': i,
                    'psi_score': feature_psi,
                    'interpretation': self._interpret_psi(feature_psi)
                })
            
            avg_psi = overall_psi / self.reference_data.shape[1]
            drift_detected = avg_psi > self.config.alert_threshold
            
            return {
                'drift_detected': drift_detected,
                'drift_score': avg_psi,
                'feature_results': feature_results,
                'interpretation': self._interpret_psi(avg_psi)
            }
    
    def _get_bin_proportions(self, data: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        """Get proportions of data in each bin."""
        counts, _ = np.histogram(data, bins=bin_edges)
        proportions = counts / len(data)
        # Avoid zero proportions
        proportions = np.maximum(proportions, 1e-6)
        return proportions
    
    def _calculate_psi(self, ref_proportions: np.ndarray, curr_proportions: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        psi = np.sum((curr_proportions - ref_proportions) * np.log(curr_proportions / ref_proportions))
        return psi
    
    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return "No significant change"
        elif psi < 0.2:
            return "Small change detected"
        else:
            return "Large change detected - investigate"

class ModelPerformanceDriftDetector:
    """Detect drift in model performance metrics."""
    
    def __init__(self, config: DriftConfig):
        self.config = config
        self.baseline_metrics = None
        self.performance_history = []
        self.logger = logging.getLogger(__name__)
    
    def set_baseline_performance(self, metrics: Dict[str, float]):
        """Set baseline performance metrics."""
        self.baseline_metrics = metrics
        self.logger.info(f"Baseline performance set: {metrics}")
    
    def detect_performance_drift(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect drift in model performance."""
        if self.baseline_metrics is None:
            raise ValueError("Baseline performance not set")
        
        drift_results = {
            'timestamp': datetime.now(),
            'metric_drifts': {},
            'overall_drift_detected': False
        }
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                relative_change = abs(current_value - baseline_value) / baseline_value
                
                drift_detected = relative_change > self.config.alert_threshold
                
                drift_results['metric_drifts'][metric_name] = {
                    'baseline_value': baseline_value,
                    'current_value': current_value,
                    'relative_change': relative_change,
                    'drift_detected': drift_detected
                }
                
                if drift_detected:
                    drift_results['overall_drift_detected'] = True
        
        self.performance_history.append(drift_results)
        return drift_results

# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    print("Model Drift Detection System Demo")
    print("=" * 50)
    
    # Generate sample data
    X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
    X_ref, X_current, y_ref, y_current = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Test 1: Statistical Drift Detection
    print("\n1. Testing Statistical Drift Detection:")
    
    config = DriftConfig(statistical_test="ks_test", significance_level=0.05)
    drift_detector = StatisticalDriftDetector(config)
    drift_detector.set_reference_data(X_ref)
    
    # Test with same distribution (no drift expected)
    drift_results = drift_detector.detect_drift(X_current)
    print(f"Drift detected (same distribution): {drift_results['drift_detected']}")
    
    # Test with shifted distribution (drift expected)
    X_shifted = X_current + np.random.normal(0, 0.5, X_current.shape)
    drift_results_shifted = drift_detector.detect_drift(X_shifted)
    print(f"Drift detected (shifted distribution): {drift_results_shifted['drift_detected']}")
    print(f"Number of drifted features: {drift_results_shifted.get('num_drifted_features', 0)}")
    
    # Test 2: PSI Drift Detection
    print("\n2. Testing PSI Drift Detection:")
    
    psi_config = DriftConfig(statistical_test="psi", alert_threshold=0.1)
    psi_detector = StatisticalDriftDetector(psi_config)
    psi_detector.set_reference_data(X_ref)
    
    psi_results = psi_detector.detect_drift(X_shifted)
    print(f"PSI Drift Score: {psi_results['drift_score']:.4f}")
    print(f"PSI Interpretation: {psi_results['interpretation']}")
    
    # Test 3: Model Performance Drift
    print("\n3. Testing Model Performance Drift:")
    
    # Train model on reference data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_ref, y_ref)
    
    # Baseline performance
    baseline_preds = model.predict(X_ref)
    baseline_metrics = {
        'accuracy': accuracy_score(y_ref, baseline_preds),
        'f1_score': f1_score(y_ref, baseline_preds, average='weighted')
    }
    
    # Current performance (on shifted data)
    current_preds = model.predict(X_shifted)
    current_metrics = {
        'accuracy': accuracy_score(y_current, current_preds),
        'f1_score': f1_score(y_current, current_preds, average='weighted')
    }
    
    perf_detector = ModelPerformanceDriftDetector(DriftConfig(alert_threshold=0.1))
    perf_detector.set_baseline_performance(baseline_metrics)
    
    perf_drift_results = perf_detector.detect_performance_drift(current_metrics)
    
    print(f"Performance drift detected: {perf_drift_results['overall_drift_detected']}")
    for metric, details in perf_drift_results['metric_drifts'].items():
        print(f"  {metric}: {details['baseline_value']:.4f} -> {details['current_value']:.4f} "
              f"(change: {details['relative_change']:.2%})")
    
    print("\nDrift detection demo completed!")
