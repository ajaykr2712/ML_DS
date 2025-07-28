"""
Advanced Model Monitoring Suite
Real-time monitoring of model performance, drift detection, and alerting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from abc import ABC, abstractmethod
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class DriftAlert:
    """Data drift alert information."""
    feature_name: str
    drift_score: float
    threshold: float
    drift_type: str  # 'statistical', 'distribution', 'performance'
    timestamp: str
    severity: str  # 'low', 'medium', 'high'

class DriftDetector(ABC):
    """Abstract base class for drift detection methods."""
    
    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Detect drift between reference and current data."""
        pass

class KSTestDriftDetector(DriftDetector):
    """Kolmogorov-Smirnov test for drift detection."""
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Perform KS test for drift detection."""
        try:
            ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
            return ks_statistic
        except Exception:
            return 0.0

class PSIDriftDetector(DriftDetector):
    """Population Stability Index (PSI) for drift detection."""
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Calculate PSI between reference and current data."""
        try:
            # Create bins based on reference data
            bins = np.histogram_bin_edges(reference_data, bins=10)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference_data, bins=bins)
            curr_hist, _ = np.histogram(current_data, bins=bins)
            
            # Normalize to get proportions
            ref_prop = ref_hist / len(reference_data)
            curr_prop = curr_hist / len(current_data)
            
            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)
            
            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            return psi
        except Exception:
            return 0.0

class ChiSquareDriftDetector(DriftDetector):
    """Chi-square test for categorical drift detection."""
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Perform chi-square test for categorical drift."""
        try:
            # Get unique categories
            all_categories = np.unique(np.concatenate([reference_data, current_data]))
            
            # Count occurrences
            ref_counts = pd.Series(reference_data).value_counts().reindex(all_categories, fill_value=0)
            curr_counts = pd.Series(current_data).value_counts().reindex(all_categories, fill_value=0)
            
            # Chi-square test
            chi2_stat, p_value = stats.chisquare(curr_counts, ref_counts)
            return chi2_stat
        except Exception:
            return 0.0

class PerformanceMonitor:
    """Monitors model performance metrics over time."""
    
    def __init__(self, model_id: str, performance_threshold: float = 0.8):
        self.model_id = model_id
        self.performance_threshold = performance_threshold
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)
    
    def log_prediction_batch(self, 
                           predictions: np.ndarray, 
                           actuals: np.ndarray,
                           prediction_probabilities: Optional[np.ndarray] = None):
        """Log a batch of predictions for monitoring."""
        try:
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, average='weighted')
            recall = recall_score(actuals, predictions, average='weighted')
            f1 = f1_score(actuals, predictions, average='weighted')
            
            auc_roc = None
            if prediction_probabilities is not None and len(np.unique(actuals)) == 2:
                auc_roc = roc_auc_score(actuals, prediction_probabilities[:, 1])
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc
            )
            
            self.metrics_history.append(metrics)
            
            # Check for performance degradation
            if accuracy < self.performance_threshold:
                self.logger.warning(f"Model {self.model_id} performance below threshold: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error logging prediction batch: {e}")
    
    def get_performance_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Get performance trend over recent predictions."""
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_metrics = self.metrics_history[-window_size:]
        
        # Calculate trends
        accuracies = [m.accuracy for m in recent_metrics]
        f1_scores = [m.f1_score for m in recent_metrics]
        
        acc_trend = "stable"
        if len(accuracies) > 1:
            acc_slope = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            if acc_slope > 0.01:
                acc_trend = "improving"
            elif acc_slope < -0.01:
                acc_trend = "degrading"
        
        return {
            "accuracy_trend": acc_trend,
            "avg_accuracy": np.mean(accuracies),
            "latest_accuracy": accuracies[-1],
            "accuracy_std": np.std(accuracies),
            "avg_f1": np.mean(f1_scores),
            "num_samples": len(recent_metrics)
        }

class DataDriftMonitor:
    """Monitors data drift across features."""
    
    def __init__(self, 
                 reference_data: pd.DataFrame,
                 drift_thresholds: Optional[Dict[str, float]] = None):
        self.reference_data = reference_data
        self.drift_thresholds = drift_thresholds or {}
        self.default_threshold = 0.1
        
        # Initialize drift detectors
        self.detectors = {
            'numerical': [KSTestDriftDetector(), PSIDriftDetector()],
            'categorical': [ChiSquareDriftDetector()]
        }
        
        self.drift_history = []
        self.logger = logging.getLogger(__name__)
    
    def detect_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """Detect drift between reference and current data."""
        alerts = []
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
            
            ref_values = self.reference_data[column].dropna()
            curr_values = current_data[column].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Determine data type and select appropriate detector
            if pd.api.types.is_numeric_dtype(ref_values):
                detectors = self.detectors['numerical']
                drift_type = 'numerical'
            else:
                detectors = self.detectors['categorical']
                drift_type = 'categorical'
            
            # Run drift detection
            for detector in detectors:
                try:
                    drift_score = detector.detect_drift(ref_values.values, curr_values.values)
                    threshold = self.drift_thresholds.get(column, self.default_threshold)
                    
                    if drift_score > threshold:
                        severity = self._determine_severity(drift_score, threshold)
                        
                        alert = DriftAlert(
                            feature_name=column,
                            drift_score=drift_score,
                            threshold=threshold,
                            drift_type=f"{drift_type}_{detector.__class__.__name__}",
                            timestamp=datetime.now().isoformat(),
                            severity=severity
                        )
                        
                        alerts.append(alert)
                        self.logger.warning(f"Drift detected in {column}: {drift_score:.4f} > {threshold}")
                
                except Exception as e:
                    self.logger.error(f"Error detecting drift in {column}: {e}")
        
        self.drift_history.extend(alerts)
        return alerts
    
    def _determine_severity(self, drift_score: float, threshold: float) -> str:
        """Determine alert severity based on drift score."""
        ratio = drift_score / threshold
        if ratio > 3:
            return "high"
        elif ratio > 2:
            return "medium"
        else:
            return "low"
    
    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get drift summary for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_alerts = [
            alert for alert in self.drift_history
            if datetime.fromisoformat(alert.timestamp) > cutoff_date
        ]
        
        summary = {
            "total_alerts": len(recent_alerts),
            "high_severity": len([a for a in recent_alerts if a.severity == "high"]),
            "medium_severity": len([a for a in recent_alerts if a.severity == "medium"]),
            "low_severity": len([a for a in recent_alerts if a.severity == "low"]),
            "affected_features": list(set([a.feature_name for a in recent_alerts])),
            "drift_types": list(set([a.drift_type for a in recent_alerts]))
        }
        
        return summary

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, alert_config: Dict[str, Any]):
        self.alert_config = alert_config
        self.alert_history = []
        self.logger = logging.getLogger(__name__)
    
    def send_alert(self, alert: DriftAlert):
        """Send drift alert through configured channels."""
        try:
            # Log alert
            self.alert_history.append(alert)
            
            # Send notifications based on configuration
            if self.alert_config.get('email_enabled', False):
                self._send_email_alert(alert)
            
            if self.alert_config.get('slack_enabled', False):
                self._send_slack_alert(alert)
            
            if self.alert_config.get('webhook_enabled', False):
                self._send_webhook_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def _send_email_alert(self, alert: DriftAlert):
        """Send email alert (placeholder implementation)."""
        self.logger.info(f"EMAIL ALERT: {alert.feature_name} drift detected (severity: {alert.severity})")
    
    def _send_slack_alert(self, alert: DriftAlert):
        """Send Slack alert (placeholder implementation)."""
        self.logger.info(f"SLACK ALERT: {alert.feature_name} drift detected (severity: {alert.severity})")
    
    def _send_webhook_alert(self, alert: DriftAlert):
        """Send webhook alert (placeholder implementation)."""
        self.logger.info(f"WEBHOOK ALERT: {alert.feature_name} drift detected (severity: {alert.severity})")

class ModelMonitoringSuite:
    """Complete model monitoring suite."""
    
    def __init__(self, 
                 model_id: str,
                 reference_data: pd.DataFrame,
                 monitoring_config: Dict[str, Any]):
        
        self.model_id = model_id
        self.monitoring_config = monitoring_config
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(
            model_id, 
            monitoring_config.get('performance_threshold', 0.8)
        )
        
        self.drift_monitor = DataDriftMonitor(
            reference_data,
            monitoring_config.get('drift_thresholds', {})
        )
        
        self.alert_manager = AlertManager(
            monitoring_config.get('alert_config', {})
        )
        
        self.logger = logging.getLogger(__name__)
    
    def monitor_batch(self, 
                     input_data: pd.DataFrame,
                     predictions: np.ndarray,
                     actuals: Optional[np.ndarray] = None,
                     prediction_probabilities: Optional[np.ndarray] = None):
        """Monitor a batch of predictions."""
        
        # Performance monitoring (if ground truth available)
        if actuals is not None:
            self.performance_monitor.log_prediction_batch(
                predictions, actuals, prediction_probabilities
            )
        
        # Drift detection
        drift_alerts = self.drift_monitor.detect_drift(input_data)
        
        # Send alerts for high-severity drifts
        for alert in drift_alerts:
            if alert.severity in ['high', 'medium']:
                self.alert_manager.send_alert(alert)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        performance_trend = self.performance_monitor.get_performance_trend()
        drift_summary = self.drift_monitor.get_drift_summary()
        
        dashboard = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "performance": performance_trend,
            "drift_summary": drift_summary,
            "recent_alerts": len(self.alert_manager.alert_history[-10:]),
            "health_status": self._calculate_health_status(performance_trend, drift_summary)
        }
        
        return dashboard
    
    def _calculate_health_status(self, 
                               performance_trend: Dict[str, Any], 
                               drift_summary: Dict[str, Any]) -> str:
        """Calculate overall model health status."""
        
        # Performance check
        if performance_trend.get("accuracy_trend") == "degrading":
            return "critical"
        
        if performance_trend.get("latest_accuracy", 0) < 0.7:
            return "critical"
        
        # Drift check
        if drift_summary.get("high_severity", 0) > 0:
            return "warning"
        
        if drift_summary.get("medium_severity", 0) > 5:
            return "warning"
        
        return "healthy"
    
    def export_monitoring_report(self, filepath: str):
        """Export comprehensive monitoring report."""
        report = {
            "model_id": self.model_id,
            "generated_at": datetime.now().isoformat(),
            "performance_history": [asdict(m) for m in self.performance_monitor.metrics_history],
            "drift_alerts": [asdict(a) for a in self.drift_monitor.drift_history],
            "dashboard": self.get_monitoring_dashboard()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Monitoring report exported to {filepath}")

# Example usage
if __name__ == "__main__":
    print("Testing Advanced Model Monitoring Suite...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Reference data (training)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.beta(2, 5, n_samples)
    })
    
    # Current data (with some drift)
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1, 500),  # Mean shift
        'feature2': np.random.exponential(1.5, 500),  # Scale change
        'feature3': np.random.choice(['A', 'B', 'C', 'D'], 500),  # New category
        'feature4': np.random.beta(2, 5, 500)  # No drift
    })
    
    # Sample predictions and actuals
    predictions = np.random.choice([0, 1], 500)
    actuals = np.random.choice([0, 1], 500)
    pred_probabilities = np.random.rand(500, 2)
    pred_probabilities = pred_probabilities / pred_probabilities.sum(axis=1, keepdims=True)
    
    # Initialize monitoring suite
    monitoring_config = {
        'performance_threshold': 0.8,
        'drift_thresholds': {
            'feature1': 0.1,
            'feature2': 0.15,
            'feature3': 0.2
        },
        'alert_config': {
            'email_enabled': True,
            'slack_enabled': True
        }
    }
    
    monitor = ModelMonitoringSuite(
        model_id="customer_churn_v1",
        reference_data=reference_data,
        monitoring_config=monitoring_config
    )
    
    # Monitor batch
    print("Monitoring prediction batch...")
    monitor.monitor_batch(
        input_data=current_data,
        predictions=predictions,
        actuals=actuals,
        prediction_probabilities=pred_probabilities
    )
    
    # Get dashboard
    dashboard = monitor.get_monitoring_dashboard()
    print(f"\nMonitoring Dashboard:")
    print(f"Model Health: {dashboard['health_status']}")
    print(f"Performance Trend: {dashboard['performance']['accuracy_trend']}")
    print(f"Total Drift Alerts: {dashboard['drift_summary']['total_alerts']}")
    print(f"High Severity Alerts: {dashboard['drift_summary']['high_severity']}")
    
    # Export report
    monitor.export_monitoring_report("monitoring_report.json")
    
    print("\nAdvanced Model Monitoring Suite implemented successfully! 🚀")
