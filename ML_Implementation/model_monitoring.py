"""
Model Performance Monitoring and Alerting System
Comprehensive monitoring for ML models in production
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from scipy import stats
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import time
from collections import deque
import warnings

warnings.filterwarnings('ignore')

@dataclass
class Alert:
    """Alert configuration and metadata"""
    id: str
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: datetime
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_count: int
    avg_response_time: float
    error_rate: float
    data_drift_score: float = 0.0
    model_drift_score: float = 0.0

class DataDriftDetector:
    """Detect data drift using statistical methods"""
    
    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize drift detector with reference data
        
        Args:
            reference_data: Training/reference dataset
            significance_level: Statistical significance threshold
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.reference_stats = self._compute_statistics(reference_data)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute statistical summaries for each feature"""
        stats_dict = {}
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                stats_dict[column] = {
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'median': data[column].median(),
                    'q25': data[column].quantile(0.25),
                    'q75': data[column].quantile(0.75)
                }
            elif data[column].dtype == 'object' or data[column].dtype.name == 'category':
                stats_dict[column] = {
                    'value_counts': data[column].value_counts().to_dict(),
                    'unique_count': data[column].nunique(),
                    'mode': data[column].mode().iloc[0] if not data[column].mode().empty else None
                }
        
        return stats_dict
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and new data
        
        Args:
            new_data: New data to compare against reference
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'feature_drift': {},
            'timestamp': datetime.now()
        }
        
        feature_drift_scores = []
        
        for column in self.reference_data.columns:
            if column not in new_data.columns:
                continue
                
            if self.reference_data[column].dtype in ['int64', 'float64']:
                # Numerical feature drift detection using KS test
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    new_data[column].dropna()
                )
                
                feature_drift = {
                    'drift_detected': p_value < self.significance_level,
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'drift_magnitude': ks_statistic
                }
                
                feature_drift_scores.append(ks_statistic)
                
            else:
                # Categorical feature drift detection using Chi-square test
                ref_counts = self.reference_data[column].value_counts()
                new_counts = new_data[column].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(new_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                new_aligned = [new_counts.get(cat, 0) for cat in all_categories]
                
                try:
                    chi2_statistic, p_value = stats.chisquare(new_aligned, ref_aligned)
                    drift_magnitude = chi2_statistic / sum(ref_aligned)
                except (ValueError, ZeroDivisionError):
                    p_value = 1.0
                    drift_magnitude = 0.0
                
                feature_drift = {
                    'drift_detected': p_value < self.significance_level,
                    'chi2_statistic': chi2_statistic if 'chi2_statistic' in locals() else 0,
                    'p_value': p_value,
                    'drift_magnitude': drift_magnitude
                }
                
                feature_drift_scores.append(drift_magnitude)
            
            drift_results['feature_drift'][column] = feature_drift
        
        # Overall drift score and detection
        drift_results['drift_score'] = np.mean(feature_drift_scores) if feature_drift_scores else 0.0
        drift_results['overall_drift_detected'] = drift_results['drift_score'] > 0.1  # Threshold
        
        return drift_results

class ModelDriftDetector:
    """Detect model performance drift"""
    
    def __init__(self, baseline_metrics: Dict[str, float], window_size: int = 100):
        """
        Initialize model drift detector
        
        Args:
            baseline_metrics: Baseline performance metrics
            window_size: Window size for moving statistics
        """
        self.baseline_metrics = baseline_metrics
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_performance_data(self, metrics: Dict[str, float]):
        """Add new performance metrics to history"""
        self.performance_history.append(metrics)
    
    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect model performance drift
        
        Returns:
            Dictionary with drift detection results
        """
        if len(self.performance_history) < self.window_size // 2:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'reason': 'Insufficient data for drift detection'
            }
        
        # Calculate current performance statistics
        current_metrics = {}
        for metric in self.baseline_metrics.keys():
            values = [perf.get(metric, 0) for perf in self.performance_history if metric in perf]
            if values:
                current_metrics[metric] = np.mean(values)
        
        # Calculate drift scores for each metric
        drift_scores = {}
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                # Relative change
                if baseline_value != 0:
                    drift_score = abs(current_value - baseline_value) / baseline_value
                else:
                    drift_score = abs(current_value - baseline_value)
                drift_scores[metric] = drift_score
        
        overall_drift_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        
        return {
            'drift_detected': overall_drift_score > 0.1,  # 10% threshold
            'drift_score': overall_drift_score,
            'metric_drift_scores': drift_scores,
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics,
            'timestamp': datetime.now()
        }

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, notification_config: Dict[str, Any] = None):
        """
        Initialize alert manager
        
        Args:
            notification_config: Configuration for notifications (email, Slack, etc.)
        """
        self.alerts = {}
        self.notification_config = notification_config or {}
        self.alert_history = deque(maxlen=1000)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_alert(self, alert: Alert):
        """Add a new alert configuration"""
        self.alerts[alert.id] = alert
        self.logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_id: str):
        """Remove an alert configuration"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            self.logger.info(f"Removed alert: {alert_id}")
    
    def check_alerts(self, metrics: ModelMetrics) -> List[Dict[str, Any]]:
        """
        Check if any alerts should be triggered
        
        Args:
            metrics: Current model metrics
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for alert_id, alert in self.alerts.items():
            if not alert.enabled:
                continue
            
            # Get metric value
            metric_value = getattr(metrics, alert.metric, None)
            if metric_value is None:
                continue
            
            # Check threshold
            triggered = False
            if alert.comparison == 'gt' and metric_value > alert.threshold:
                triggered = True
            elif alert.comparison == 'lt' and metric_value < alert.threshold:
                triggered = True
            elif alert.comparison == 'eq' and metric_value == alert.threshold:
                triggered = True
            
            if triggered:
                alert_info = {
                    'alert_id': alert_id,
                    'alert_name': alert.name,
                    'metric': alert.metric,
                    'current_value': metric_value,
                    'threshold': alert.threshold,
                    'severity': alert.severity,
                    'timestamp': datetime.now()
                }
                
                triggered_alerts.append(alert_info)
                self.alert_history.append(alert_info)
                
                # Send notification
                self._send_notification(alert_info)
        
        return triggered_alerts
    
    def _send_notification(self, alert_info: Dict[str, Any]):
        """Send notification for triggered alert"""
        try:
            if 'email' in self.notification_config:
                self._send_email_notification(alert_info)
            
            if 'slack' in self.notification_config:
                self._send_slack_notification(alert_info)
                
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def _send_email_notification(self, alert_info: Dict[str, Any]):
        """Send email notification"""
        email_config = self.notification_config['email']
        
        msg = MimeMultipart()
        msg['From'] = email_config['from_address']
        msg['To'] = ', '.join(email_config['to_addresses'])
        msg['Subject'] = f"ML Model Alert: {alert_info['alert_name']}"
        
        body = f"""
        Alert: {alert_info['alert_name']}
        Severity: {alert_info['severity']}
        Metric: {alert_info['metric']}
        Current Value: {alert_info['current_value']}
        Threshold: {alert_info['threshold']}
        Time: {alert_info['timestamp']}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        if email_config.get('use_tls', True):
            server.starttls()
        if 'username' in email_config:
            server.login(email_config['username'], email_config['password'])
        
        server.send_message(msg)
        server.quit()
    
    def _send_slack_notification(self, alert_info: Dict[str, Any]):
        """Send Slack notification"""
        slack_config = self.notification_config['slack']
        
        message = {
            "text": f"🚨 ML Model Alert: {alert_info['alert_name']}",
            "attachments": [
                {
                    "color": "danger" if alert_info['severity'] in ['high', 'critical'] else "warning",
                    "fields": [
                        {"title": "Metric", "value": alert_info['metric'], "short": True},
                        {"title": "Current Value", "value": str(alert_info['current_value']), "short": True},
                        {"title": "Threshold", "value": str(alert_info['threshold']), "short": True},
                        {"title": "Severity", "value": alert_info['severity'], "short": True}
                    ]
                }
            ]
        }
        
        response = requests.post(slack_config['webhook_url'], json=message)
        response.raise_for_status()

class ModelMonitor:
    """Main model monitoring system"""
    
    def __init__(self, 
                 model_id: str,
                 reference_data: pd.DataFrame = None,
                 baseline_metrics: Dict[str, float] = None,
                 notification_config: Dict[str, Any] = None):
        """
        Initialize model monitor
        
        Args:
            model_id: Unique identifier for the model
            reference_data: Reference data for drift detection
            baseline_metrics: Baseline performance metrics
            notification_config: Configuration for notifications
        """
        self.model_id = model_id
        self.metrics_history = deque(maxlen=10000)
        
        # Initialize drift detectors
        self.data_drift_detector = DataDriftDetector(reference_data) if reference_data is not None else None
        self.model_drift_detector = ModelDriftDetector(baseline_metrics) if baseline_metrics else None
        
        # Initialize alert manager
        self.alert_manager = AlertManager(notification_config)
        
        # Database for persistence
        self.db_path = f"model_monitor_{model_id}.db"
        self._init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for storing metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_id TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                prediction_count INTEGER,
                avg_response_time REAL,
                error_rate REAL,
                data_drift_score REAL,
                model_drift_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_id TEXT,
                alert_name TEXT,
                metric TEXT,
                current_value REAL,
                threshold REAL,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_metrics(self, metrics: ModelMetrics):
        """Add new metrics and check for alerts"""
        # Add to history
        self.metrics_history.append(metrics)
        
        # Check for drift
        if self.model_drift_detector:
            self.model_drift_detector.add_performance_data({
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score
            })
        
        # Check alerts
        triggered_alerts = self.alert_manager.check_alerts(metrics)
        
        # Store in database
        self._store_metrics(metrics)
        for alert in triggered_alerts:
            self._store_alert(alert)
        
        # Log summary
        self.logger.info(f"Added metrics for model {self.model_id}. "
                        f"Accuracy: {metrics.accuracy:.4f}, "
                        f"Alerts triggered: {len(triggered_alerts)}")
        
        return triggered_alerts
    
    def check_data_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift"""
        if self.data_drift_detector is None:
            return {'error': 'Data drift detector not initialized'}
        
        return self.data_drift_detector.detect_drift(new_data)
    
    def check_model_drift(self) -> Dict[str, Any]:
        """Check for model drift"""
        if self.model_drift_detector is None:
            return {'error': 'Model drift detector not initialized'}
        
        return self.model_drift_detector.detect_drift()
    
    def _store_metrics(self, metrics: ModelMetrics):
        """Store metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (
                timestamp, model_id, accuracy, precision_score, recall, f1_score,
                prediction_count, avg_response_time, error_rate, 
                data_drift_score, model_drift_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.model_id,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
            metrics.prediction_count,
            metrics.avg_response_time,
            metrics.error_rate,
            metrics.data_drift_score,
            metrics.model_drift_score
        ))
        
        conn.commit()
        conn.close()
    
    def _store_alert(self, alert_info: Dict[str, Any]):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (
                timestamp, alert_id, alert_name, metric, 
                current_value, threshold, severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert_info['timestamp'].isoformat(),
            alert_info['alert_id'],
            alert_info['alert_name'],
            alert_info['metric'],
            alert_info['current_value'],
            alert_info['threshold'],
            alert_info['severity']
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No recent metrics available'}
        
        return {
            'total_predictions': sum(m.prediction_count for m in recent_metrics),
            'avg_accuracy': np.mean([m.accuracy for m in recent_metrics]),
            'avg_response_time': np.mean([m.avg_response_time for m in recent_metrics]),
            'avg_error_rate': np.mean([m.error_rate for m in recent_metrics]),
            'metrics_count': len(recent_metrics),
            'time_range': f"Last {hours} hours"
        }

# Example usage
if __name__ == "__main__":
    # Create sample reference data
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Baseline metrics
    baseline_metrics = {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'f1_score': 0.77
    }
    
    # Initialize monitor
    monitor = ModelMonitor(
        model_id="test_model_v1",
        reference_data=reference_data,
        baseline_metrics=baseline_metrics
    )
    
    # Add some alerts
    monitor.alert_manager.add_alert(Alert(
        id="low_accuracy",
        name="Low Accuracy Alert",
        metric="accuracy",
        threshold=0.70,
        comparison="lt",
        severity="high"
    ))
    
    # Simulate some metrics
    for i in range(10):
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_id="test_model_v1",
            accuracy=0.85 + np.random.normal(0, 0.05),
            precision=0.80 + np.random.normal(0, 0.05),
            recall=0.75 + np.random.normal(0, 0.05),
            f1_score=0.77 + np.random.normal(0, 0.05),
            prediction_count=np.random.randint(100, 1000),
            avg_response_time=np.random.uniform(0.1, 0.5),
            error_rate=np.random.uniform(0.0, 0.05)
        )
        
        triggered_alerts = monitor.add_metrics(metrics)
        time.sleep(1)  # Small delay
    
    # Get summary
    summary = monitor.get_metrics_summary(hours=1)
    print(f"Metrics summary: {summary}")
    
    # Check for drift
    new_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1, 500),  # Slight drift
        'feature2': np.random.normal(5, 2, 500),
        'feature3': np.random.choice(['A', 'B', 'C'], 500)
    })
    
    drift_results = monitor.check_data_drift(new_data)
    print(f"Data drift detected: {drift_results['overall_drift_detected']}")
    print(f"Drift score: {drift_results['drift_score']:.4f}")
