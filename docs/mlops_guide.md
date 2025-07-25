# MLOps Best Practices and Implementation Guide

## Table of Contents
1. [Model Development Lifecycle](#model-development-lifecycle)
2. [Version Control for ML](#version-control-for-ml)
3. [Automated Testing](#automated-testing)
4. [Model Deployment Strategies](#model-deployment-strategies)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Security and Compliance](#security-and-compliance)

## Model Development Lifecycle

### 1. Experimentation Phase
- **Jupyter Notebooks**: For initial exploration and prototyping
- **MLflow Tracking**: Log experiments, parameters, and metrics
- **Data Versioning**: Use DVC for dataset version control
- **Feature Stores**: Centralized feature management

### 2. Development Phase
- **Code Organization**: Modular, testable Python packages
- **Configuration Management**: YAML/JSON config files
- **Environment Management**: Docker containers, virtual environments
- **Code Quality**: Linting, formatting, type checking

### 3. Training Phase
- **Pipeline Orchestration**: Kubeflow, Apache Airflow
- **Resource Management**: GPU scheduling, auto-scaling
- **Hyperparameter Tuning**: Optuna, Ray Tune
- **Model Validation**: Cross-validation, holdout sets

### 4. Deployment Phase
- **Model Packaging**: Docker images, Python packages
- **Inference Endpoints**: REST APIs, gRPC services
- **Batch Processing**: Scheduled model runs
- **Edge Deployment**: Mobile, IoT devices

## Version Control for ML

### Git Best Practices
```bash
# Feature branch workflow
git checkout -b feature/new-algorithm
git add src/models/new_algorithm.py
git commit -m "feat: implement XGBoost classifier with custom features"
git push origin feature/new-algorithm

# Create pull request for review
```

### DVC (Data Version Control)
```yaml
# dvc.yaml - Pipeline definition
stages:
  data_preprocessing:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw/
      - src/data/preprocess.py
    outs:
      - data/processed/

  train:
    cmd: python src/models/train.py
    deps:
      - data/processed/
      - src/models/train.py
    outs:
      - models/
    metrics:
      - metrics/train_metrics.json
```

### Model Registry
- **MLflow Model Registry**: Centralized model storage
- **Model Versioning**: Semantic versioning (v1.0.0)
- **Stage Management**: Development → Staging → Production
- **Model Lineage**: Track data and code dependencies

## Automated Testing

### Unit Tests
```python
import pytest
from src.models.classifier import XGBoostClassifier

def test_model_training():
    """Test model training with sample data."""
    model = XGBoostClassifier()
    X, y = generate_sample_data()
    
    model.fit(X, y)
    
    assert model.is_fitted
    assert model.feature_importance_ is not None

def test_model_prediction():
    """Test model prediction functionality."""
    model = XGBoostClassifier()
    X, y = generate_sample_data()
    model.fit(X, y)
    
    predictions = model.predict(X[:10])
    
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)
```

### Integration Tests
```python
def test_full_pipeline():
    """Test complete ML pipeline end-to-end."""
    # Load data
    data = load_test_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(data)
    
    # Train model
    model = XGBoostClassifier()
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X[:100])
    
    # Validate output
    assert len(predictions) == 100
    assert model.score(X, y) > 0.8
```

### Data Quality Tests
```python
def test_data_schema():
    """Test data schema compliance."""
    data = load_training_data()
    
    # Check required columns
    required_cols = ['feature1', 'feature2', 'target']
    assert all(col in data.columns for col in required_cols)
    
    # Check data types
    assert data['feature1'].dtype == 'float64'
    assert data['target'].dtype == 'int64'
    
    # Check for missing values
    assert data.isnull().sum().sum() == 0
```

### Model Performance Tests
```python
def test_model_performance():
    """Test model meets performance requirements."""
    model = load_production_model()
    test_data = load_test_data()
    
    accuracy = model.score(test_data['X'], test_data['y'])
    precision = precision_score(test_data['y'], model.predict(test_data['X']))
    
    assert accuracy >= 0.85, f"Accuracy {accuracy} below threshold"
    assert precision >= 0.80, f"Precision {precision} below threshold"
```

## Model Deployment Strategies

### Blue-Green Deployment
```yaml
# kubernetes/blue-green-deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
    version: blue  # Switch to 'green' for deployment
  ports:
    - port: 80
      targetPort: 8080
```

### Canary Deployment
```yaml
# Istio VirtualService for canary deployment
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: ml-model-canary
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: ml-model-v2
  - route:
    - destination:
        host: ml-model-v1
      weight: 90
    - destination:
        host: ml-model-v2
      weight: 10
```

### A/B Testing
```python
class ABTestingController:
    def __init__(self):
        self.model_a = load_model('model_v1')
        self.model_b = load_model('model_v2')
        
    def predict(self, X, user_id):
        # Route based on user ID hash
        if hash(user_id) % 2 == 0:
            return self.model_a.predict(X), 'model_a'
        else:
            return self.model_b.predict(X), 'model_b'
```

## Monitoring and Observability

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')

@PREDICTION_LATENCY.time()
def predict(model, X):
    PREDICTION_COUNTER.inc()
    return model.predict(X)
```

### Logging Strategy
```python
import logging
import structlog

# Structured logging configuration
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def log_prediction(model_id, input_features, prediction, confidence):
    logger.info(
        "model_prediction",
        model_id=model_id,
        input_features=input_features,
        prediction=prediction,
        confidence=confidence
    )
```

### Alerting Rules
```yaml
# Prometheus alerting rules
groups:
- name: ml_model_alerts
  rules:
  - alert: HighPredictionLatency
    expr: ml_prediction_duration_seconds > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High prediction latency detected"
      
  - alert: LowModelAccuracy
    expr: ml_model_accuracy < 0.85
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Model accuracy below threshold"
```

## Security and Compliance

### Model Security
- **Input Validation**: Sanitize and validate all inputs
- **Access Control**: RBAC for model endpoints
- **Encryption**: TLS for data in transit, encryption at rest
- **Audit Logging**: Track all model access and predictions

### Privacy Protection
- **Data Anonymization**: Remove or hash PII
- **Differential Privacy**: Add noise to protect individual privacy
- **GDPR Compliance**: Right to explanation, data deletion
- **Federated Learning**: Train without centralizing data

### Model Governance
```python
class ModelGovernance:
    def __init__(self):
        self.approved_models = set()
        self.model_metadata = {}
    
    def register_model(self, model_id, metadata):
        """Register model with governance metadata."""
        required_fields = ['owner', 'purpose', 'data_sources', 'approval_date']
        
        if not all(field in metadata for field in required_fields):
            raise ValueError("Missing required metadata fields")
        
        self.model_metadata[model_id] = metadata
        self.approved_models.add(model_id)
    
    def validate_deployment(self, model_id):
        """Validate model is approved for deployment."""
        if model_id not in self.approved_models:
            raise PermissionError("Model not approved for deployment")
        
        return True
```

## Tools and Technologies

### Core MLOps Stack
- **Orchestration**: Kubeflow, Apache Airflow, Prefect
- **Experiment Tracking**: MLflow, Weights & Biases, Neptune
- **Model Serving**: Seldon, KFServing, TorchServe
- **Feature Stores**: Feast, Tecton, Hopsworks
- **Monitoring**: Prometheus, Grafana, DataDog

### Cloud Platforms
- **AWS**: SageMaker, EKS, Lambda
- **GCP**: Vertex AI, GKE, Cloud Functions
- **Azure**: ML Studio, AKS, Functions

### Infrastructure as Code
```hcl
# Terraform configuration for ML infrastructure
resource "aws_sagemaker_notebook_instance" "ml_notebook" {
  name          = "ml-development"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t3.medium"
  
  tags = {
    Environment = "development"
    Team        = "ml-engineering"
  }
}

resource "aws_eks_cluster" "ml_cluster" {
  name     = "ml-training-cluster"
  role_arn = aws_iam_role.eks_role.arn
  version  = "1.21"
  
  vpc_config {
    subnet_ids = var.subnet_ids
  }
}
```

This guide provides a comprehensive foundation for implementing MLOps practices in any machine learning project.
