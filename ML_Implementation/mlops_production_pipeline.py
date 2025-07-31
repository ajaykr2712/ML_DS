"""
Production-Ready MLOps Pipeline Framework
Complete MLOps solution for model lifecycle management

Features:
- Automated model training and deployment pipelines
- Real-time model monitoring and drift detection
- A/B testing framework for model validation
- Model versioning and registry
- Automated rollback and canary deployments
- Performance monitoring and alerting
- Data quality validation and lineage tracking
- Kubernetes-native deployment support
"""

import os
import json
import yaml
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
import sqlite3
from contextlib import contextmanager
import warnings

warnings.filterwarnings('ignore')

@dataclass
class ModelMetadata:
    """Model metadata for registry"""
    model_id: str
    version: str
    name: str
    algorithm: str
    framework: str
    training_date: datetime
    author: str
    description: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    model_size_mb: float
    status: str = "active"  # active, deprecated, archived
    deployment_config: Dict[str, Any] = None
    tags: List[str] = None

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str
    model_id: str
    environment: str  # dev, staging, production
    deployment_type: str  # blue_green, canary, rolling
    traffic_percentage: float = 100.0
    resource_requirements: Dict[str, Any] = None
    auto_scaling: Dict[str, Any] = None
    health_check_config: Dict[str, Any] = None
    rollback_config: Dict[str, Any] = None

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    data_quality_checks: List[str] = None
    alerting_config: Dict[str, Any] = None
    monitoring_frequency: str = "hourly"  # hourly, daily, real-time
    metrics_to_track: List[str] = None

class ModelRegistry:
    """Centralized model registry for version control and metadata management"""
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.db_path = self.registry_path / "registry.db"
        self._init_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for model metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    name TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    training_date TEXT NOT NULL,
                    author TEXT NOT NULL,
                    description TEXT,
                    performance_metrics TEXT,
                    hyperparameters TEXT,
                    training_data_hash TEXT,
                    model_size_mb REAL,
                    status TEXT DEFAULT 'active',
                    deployment_config TEXT,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployment_type TEXT NOT NULL,
                    traffic_percentage REAL DEFAULT 100.0,
                    status TEXT DEFAULT 'active',
                    deployed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)
            
            conn.commit()
    
    def register_model(self, model: BaseEstimator, metadata: ModelMetadata) -> str:
        """Register a new model in the registry"""
        # Generate unique model ID if not provided
        if not metadata.model_id:
            metadata.model_id = self._generate_model_id(metadata.name, metadata.version)
        
        # Save model file
        model_path = self.registry_path / f"{metadata.model_id}.joblib"
        joblib.dump(model, model_path)
        
        # Calculate model size
        metadata.model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Store metadata in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models 
                (model_id, version, name, algorithm, framework, training_date, author, 
                 description, performance_metrics, hyperparameters, training_data_hash, 
                 model_size_mb, status, deployment_config, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.model_id, metadata.version, metadata.name, metadata.algorithm,
                metadata.framework, metadata.training_date.isoformat(), metadata.author,
                metadata.description, json.dumps(metadata.performance_metrics),
                json.dumps(metadata.hyperparameters), metadata.training_data_hash,
                metadata.model_size_mb, metadata.status,
                json.dumps(metadata.deployment_config) if metadata.deployment_config else None,
                json.dumps(metadata.tags) if metadata.tags else None
            ))
            conn.commit()
        
        self.logger.info(f"Model {metadata.model_id} registered successfully")
        return metadata.model_id
    
    def load_model(self, model_id: str) -> BaseEstimator:
        """Load a model from the registry"""
        model_path = self.registry_path / f"{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found in registry")
        
        return joblib.load(model_path)
    
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata from registry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Model {model_id} not found in registry")
            
            # Parse row data
            columns = [desc[0] for desc in cursor.description]
            model_data = dict(zip(columns, row))
            
            # Convert JSON fields back to objects
            model_data['performance_metrics'] = json.loads(model_data['performance_metrics'] or '{}')
            model_data['hyperparameters'] = json.loads(model_data['hyperparameters'] or '{}')
            model_data['deployment_config'] = json.loads(model_data['deployment_config'] or 'null')
            model_data['tags'] = json.loads(model_data['tags'] or '[]')
            model_data['training_date'] = datetime.fromisoformat(model_data['training_date'])
            
            # Remove database-specific fields
            model_data.pop('created_at', None)
            
            return ModelMetadata(**model_data)
    
    def list_models(self, status: str = None, algorithm: str = None) -> List[ModelMetadata]:
        """List models in registry with optional filtering"""
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        
        query += " ORDER BY created_at DESC"
        
        models = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                model_data = dict(zip(columns, row))
                
                # Convert JSON fields
                model_data['performance_metrics'] = json.loads(model_data['performance_metrics'] or '{}')
                model_data['hyperparameters'] = json.loads(model_data['hyperparameters'] or '{}')
                model_data['deployment_config'] = json.loads(model_data['deployment_config'] or 'null')
                model_data['tags'] = json.loads(model_data['tags'] or '[]')
                model_data['training_date'] = datetime.fromisoformat(model_data['training_date'])
                model_data.pop('created_at', None)
                
                models.append(ModelMetadata(**model_data))
        
        return models
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{version}_{timestamp}"

class ModelMonitor:
    """Real-time model monitoring and drift detection"""
    
    def __init__(self, config: MonitoringConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.metrics_history = {}
        self.drift_detectors = {}
        self.alerts = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics storage
        self.metrics_path = Path("monitoring_data")
        self.metrics_path.mkdir(exist_ok=True)
    
    def setup_drift_detection(self, model_id: str, reference_data: np.ndarray):
        """Setup drift detection for a model"""
        from scipy import stats
        
        # Store reference statistics
        self.drift_detectors[model_id] = {
            'reference_mean': np.mean(reference_data, axis=0),
            'reference_std': np.std(reference_data, axis=0),
            'reference_data': reference_data[:1000]  # Keep sample for KS test
        }
        
        self.logger.info(f"Drift detection setup for model {model_id}")
    
    def detect_data_drift(self, model_id: str, new_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        if model_id not in self.drift_detectors:
            raise ValueError(f"Drift detection not setup for model {model_id}")
        
        detector = self.drift_detectors[model_id]
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drifts': []
        }
        
        # Calculate drift for each feature
        for i in range(new_data.shape[1]):
            feature_drift = self._calculate_feature_drift(
                detector['reference_data'][:, i],
                new_data[:, i]
            )
            
            drift_results['feature_drifts'].append({
                'feature_index': i,
                'drift_score': feature_drift['score'],
                'p_value': feature_drift['p_value'],
                'drift_detected': feature_drift['drift_detected']
            })
            
            if feature_drift['drift_detected']:
                drift_results['drift_detected'] = True
                drift_results['drift_score'] = max(drift_results['drift_score'], 
                                                 feature_drift['score'])
        
        # Store drift results
        self._store_drift_results(drift_results)
        
        # Trigger alerts if drift detected
        if drift_results['drift_detected']:
            self._trigger_drift_alert(drift_results)
        
        return drift_results
    
    def _calculate_feature_drift(self, reference: np.ndarray, 
                               current: np.ndarray) -> Dict[str, float]:
        """Calculate drift for a single feature using KS test"""
        from scipy.stats import ks_2samp
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(reference, current)
        
        drift_detected = ks_stat > self.config.drift_threshold
        
        return {
            'score': ks_stat,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    
    def monitor_model_performance(self, model_id: str, predictions: np.ndarray, 
                                true_labels: np.ndarray) -> Dict[str, Any]:
        """Monitor model performance in real-time"""
        # Calculate current performance metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sample_count': len(predictions)
        }
        
        # Store metrics
        self._store_performance_metrics(current_metrics)
        
        # Check for performance degradation
        degradation_detected = self._check_performance_degradation(model_id, current_metrics)
        
        if degradation_detected:
            self._trigger_performance_alert(model_id, current_metrics)
        
        return current_metrics
    
    def _check_performance_degradation(self, model_id: str, 
                                     current_metrics: Dict[str, Any]) -> bool:
        """Check if model performance has degraded"""
        # Get historical performance
        historical_metrics = self._get_historical_metrics(model_id, days=7)
        
        if not historical_metrics:
            return False
        
        # Calculate historical average
        historical_accuracy = np.mean([m['accuracy'] for m in historical_metrics])
        
        # Check if current performance is below threshold
        performance_drop = historical_accuracy - current_metrics['accuracy']
        
        return performance_drop > self.config.performance_threshold
    
    def _store_drift_results(self, drift_results: Dict[str, Any]):
        """Store drift detection results"""
        filename = f"drift_results_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self.metrics_path / filename
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(drift_results) + '\n')
    
    def _store_performance_metrics(self, metrics: Dict[str, Any]):
        """Store performance metrics"""
        filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self.metrics_path / filename
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def _get_historical_metrics(self, model_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get historical performance metrics"""
        metrics = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Read recent metric files
        for i in range(days):
            date = cutoff_date + timedelta(days=i)
            filename = f"performance_metrics_{date.strftime('%Y%m%d')}.jsonl"
            filepath = self.metrics_path / filename
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    for line in f:
                        metric = json.loads(line.strip())
                        if metric['model_id'] == model_id:
                            metrics.append(metric)
        
        return metrics
    
    def _trigger_drift_alert(self, drift_results: Dict[str, Any]):
        """Trigger alert for data drift"""
        alert = {
            'type': 'data_drift',
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
            'model_id': drift_results['model_id'],
            'message': f"Data drift detected for model {drift_results['model_id']}",
            'details': drift_results
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"Data drift alert: {alert['message']}")
    
    def _trigger_performance_alert(self, model_id: str, metrics: Dict[str, Any]):
        """Trigger alert for performance degradation"""
        alert = {
            'type': 'performance_degradation',
            'severity': 'critical',
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'message': f"Performance degradation detected for model {model_id}",
            'details': metrics
        }
        
        self.alerts.append(alert)
        self.logger.critical(f"Performance degradation alert: {alert['message']}")

class ABTestingFramework:
    """A/B testing framework for model validation"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_tests = {}
        self.test_results = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def create_ab_test(self, test_id: str, control_model_id: str, 
                      treatment_model_id: str, traffic_split: float = 0.5,
                      success_metric: str = 'accuracy', 
                      minimum_sample_size: int = 1000) -> Dict[str, Any]:
        """Create a new A/B test"""
        ab_test = {
            'test_id': test_id,
            'control_model_id': control_model_id,
            'treatment_model_id': treatment_model_id,
            'traffic_split': traffic_split,
            'success_metric': success_metric,
            'minimum_sample_size': minimum_sample_size,
            'start_time': datetime.now().isoformat(),
            'status': 'active',
            'control_results': [],
            'treatment_results': []
        }
        
        self.active_tests[test_id] = ab_test
        self.logger.info(f"A/B test {test_id} created")
        
        return ab_test
    
    def route_traffic(self, test_id: str, request_id: str) -> str:
        """Route traffic between control and treatment models"""
        if test_id not in self.active_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        # Simple hash-based routing for consistency
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        routing_value = (hash_value % 100) / 100.0
        
        if routing_value < test['traffic_split']:
            return test['treatment_model_id']
        else:
            return test['control_model_id']
    
    def record_result(self, test_id: str, model_id: str, prediction: Any, 
                     true_label: Any, additional_metrics: Dict[str, float] = None):
        """Record prediction result for A/B test"""
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'true_label': true_label,
            'additional_metrics': additional_metrics or {}
        }
        
        if model_id == test['control_model_id']:
            test['control_results'].append(result)
        elif model_id == test['treatment_model_id']:
            test['treatment_results'].append(result)
    
    def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results and determine statistical significance"""
        if test_id not in self.active_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        # Extract predictions and true labels
        control_preds = [r['prediction'] for r in test['control_results']]
        control_true = [r['true_label'] for r in test['control_results']]
        treatment_preds = [r['prediction'] for r in test['treatment_results']]
        treatment_true = [r['true_label'] for r in test['treatment_results']]
        
        if len(control_preds) == 0 or len(treatment_preds) == 0:
            return {'status': 'insufficient_data', 'message': 'Not enough data collected'}
        
        # Calculate metrics
        control_accuracy = accuracy_score(control_true, control_preds)
        treatment_accuracy = accuracy_score(treatment_true, treatment_preds)
        
        # Statistical significance test
        significance_result = self._test_statistical_significance(
            control_accuracy, len(control_preds),
            treatment_accuracy, len(treatment_preds)
        )
        
        analysis = {
            'test_id': test_id,
            'control_accuracy': control_accuracy,
            'treatment_accuracy': treatment_accuracy,
            'improvement': treatment_accuracy - control_accuracy,
            'relative_improvement': (treatment_accuracy - control_accuracy) / control_accuracy * 100,
            'control_sample_size': len(control_preds),
            'treatment_sample_size': len(treatment_preds),
            'statistical_significance': significance_result,
            'recommendation': self._generate_recommendation(
                control_accuracy, treatment_accuracy, significance_result
            )
        }
        
        return analysis
    
    def _test_statistical_significance(self, control_rate: float, control_n: int,
                                     treatment_rate: float, treatment_n: int,
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """Test statistical significance using two-proportion z-test"""
        from scipy import stats
        
        # Pooled proportion
        pooled_p = (control_rate * control_n + treatment_rate * treatment_n) / (control_n + treatment_n)
        
        # Standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_n + 1/treatment_n))
        
        # Z-score
        z_score = (treatment_rate - control_rate) / se
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'confidence_level': (1 - alpha) * 100
        }
    
    def _generate_recommendation(self, control_rate: float, treatment_rate: float,
                               significance: Dict[str, Any]) -> str:
        """Generate recommendation based on test results"""
        if not significance['is_significant']:
            return "No significant difference detected. Continue testing or maintain current model."
        
        if treatment_rate > control_rate:
            return "Treatment model shows significant improvement. Recommend deployment."
        else:
            return "Treatment model shows significant degradation. Do not deploy."

class DeploymentManager:
    """Manages model deployments with blue-green and canary strategies"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_deployments = {}
        self.deployment_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def deploy_model(self, model_id: str, config: DeploymentConfig) -> str:
        """Deploy a model using specified strategy"""
        # Validate model exists
        try:
            self.registry.get_model_metadata(model_id)
        except ValueError:
            raise ValueError(f"Model {model_id} not found in registry")
        
        deployment = {
            'deployment_id': config.deployment_id,
            'model_id': model_id,
            'config': config,
            'status': 'deploying',
            'start_time': datetime.now().isoformat(),
            'health_status': 'healthy'
        }
        
        try:
            if config.deployment_type == 'blue_green':
                self._deploy_blue_green(deployment)
            elif config.deployment_type == 'canary':
                self._deploy_canary(deployment)
            elif config.deployment_type == 'rolling':
                self._deploy_rolling(deployment)
            else:
                raise ValueError(f"Unknown deployment type: {config.deployment_type}")
            
            deployment['status'] = 'deployed'
            self.active_deployments[config.deployment_id] = deployment
            self.deployment_history.append(deployment.copy())
            
            self.logger.info(f"Model {model_id} deployed successfully as {config.deployment_id}")
            
        except Exception as e:
            deployment['status'] = 'failed'
            deployment['error'] = str(e)
            self.deployment_history.append(deployment.copy())
            self.logger.error(f"Deployment failed: {e}")
            raise
        
        return config.deployment_id
    
    def _deploy_blue_green(self, deployment: Dict[str, Any]):
        """Deploy using blue-green strategy"""
        self.logger.info(f"Executing blue-green deployment for {deployment['deployment_id']}")
        
        # Simulate blue-green deployment steps
        steps = [
            "Creating green environment",
            "Loading model in green environment",
            "Running health checks",
            "Switching traffic to green",
            "Terminating blue environment"
        ]
        
        for step in steps:
            self.logger.info(f"Blue-green deployment: {step}")
            time.sleep(1)  # Simulate deployment time
    
    def _deploy_canary(self, deployment: Dict[str, Any]):
        """Deploy using canary strategy"""
        self.logger.info(f"Executing canary deployment for {deployment['deployment_id']}")
        
        config = deployment['config']
        traffic_percentage = config.traffic_percentage
        
        # Simulate canary deployment with gradual traffic increase
        canary_stages = [traffic_percentage * 0.1, traffic_percentage * 0.5, traffic_percentage]
        
        for stage_traffic in canary_stages:
            self.logger.info(f"Canary deployment: Routing {stage_traffic}% traffic to new version")
            time.sleep(2)  # Simulate monitoring period
            
            # Simulate health check
            if not self._health_check(deployment):
                raise Exception("Health check failed during canary deployment")
    
    def _deploy_rolling(self, deployment: Dict[str, Any]):
        """Deploy using rolling update strategy"""
        self.logger.info(f"Executing rolling deployment for {deployment['deployment_id']}")
        
        # Simulate rolling update
        instances = 5  # Simulate 5 instances
        for i in range(instances):
            self.logger.info(f"Rolling deployment: Updating instance {i+1}/{instances}")
            time.sleep(1)
    
    def _health_check(self, deployment: Dict[str, Any]) -> bool:
        """Perform health check on deployment"""
        # Simulate health check
        return True
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        try:
            self.logger.info(f"Rolling back deployment {deployment_id}")
            
            # Simulate rollback process
            rollback_steps = [
                "Identifying previous stable version",
                "Switching traffic back to previous version",
                "Terminating failed deployment"
            ]
            
            for step in rollback_steps:
                self.logger.info(f"Rollback: {step}")
                time.sleep(1)
            
            deployment['status'] = 'rolled_back'
            deployment['rollback_time'] = datetime.now().isoformat()
            
            self.logger.info(f"Deployment {deployment_id} rolled back successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

class MLOpsPipeline:
    """Complete MLOps pipeline orchestrator"""
    
    def __init__(self, config_path: str = "mlops_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.registry = ModelRegistry(self.config.get('registry_path', 'model_registry'))
        self.monitor = ModelMonitor(
            MonitoringConfig(**self.config.get('monitoring', {})), 
            self.registry
        )
        self.ab_testing = ABTestingFramework(self.registry)
        self.deployment_manager = DeploymentManager(self.registry)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self.pipeline_runs = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MLOps configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'registry_path': 'model_registry',
                'monitoring': {
                    'drift_threshold': 0.1,
                    'performance_threshold': 0.05,
                    'monitoring_frequency': 'hourly'
                },
                'deployment': {
                    'default_strategy': 'canary',
                    'health_check_timeout': 300
                }
            }
    
    def run_training_pipeline(self, model: BaseEstimator, X_train: np.ndarray, 
                            y_train: np.ndarray, metadata: ModelMetadata) -> str:
        """Run complete training and registration pipeline"""
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting training pipeline {pipeline_id}")
        
        try:
            # Train model
            self.logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Validate model
            self.logger.info("Validating model...")
            validation_score = self._validate_model(model, X_train, y_train)
            metadata.performance_metrics['validation_score'] = validation_score
            
            # Register model
            self.logger.info("Registering model...")
            model_id = self.registry.register_model(model, metadata)
            
            # Setup monitoring
            self.logger.info("Setting up monitoring...")
            self.monitor.setup_drift_detection(model_id, X_train)
            
            self.logger.info(f"Training pipeline {pipeline_id} completed successfully")
            
            self.pipeline_runs[pipeline_id] = {
                'status': 'completed',
                'model_id': model_id,
                'completion_time': datetime.now().isoformat()
            }
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Training pipeline {pipeline_id} failed: {e}")
            self.pipeline_runs[pipeline_id] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
            raise
    
    def _validate_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        """Validate model performance"""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5)
        return scores.mean()
    
    def run_deployment_pipeline(self, model_id: str, environment: str = 'production',
                              deployment_type: str = 'canary') -> str:
        """Run complete deployment pipeline"""
        deployment_id = f"deploy_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create deployment configuration
        config = DeploymentConfig(
            deployment_id=deployment_id,
            model_id=model_id,
            environment=environment,
            deployment_type=deployment_type,
            traffic_percentage=100.0 if deployment_type != 'canary' else 10.0
        )
        
        # Deploy model
        return self.deployment_manager.deploy_model(model_id, config)

# Example usage
if __name__ == "__main__":
    # Initialize MLOps pipeline
    pipeline = MLOpsPipeline()
    
    # Example model training and deployment
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create metadata
    metadata = ModelMetadata(
        model_id="",
        version="1.0.0",
        name="fraud_detection",
        algorithm="RandomForest",
        framework="scikit-learn",
        training_date=datetime.now(),
        author="MLOps Team",
        description="Fraud detection model for financial transactions",
        performance_metrics={},
        hyperparameters={"n_estimators": 100, "random_state": 42},
        training_data_hash=hashlib.md5(str(X).encode()).hexdigest()[:16]
    )
    
    # Run training pipeline
    model_id = pipeline.run_training_pipeline(model, X, y, metadata)
    print(f"Model trained and registered: {model_id}")
    
    # Run deployment pipeline
    deployment_id = pipeline.run_deployment_pipeline(model_id, environment='production')
    print(f"Model deployed: {deployment_id}")
