# Machine Learning Project Templates and Patterns

## Overview
This document provides standardized templates and design patterns for machine learning projects, ensuring consistency, maintainability, and best practices across different ML applications.

## Project Structure Templates

### 1. Standard ML Project Structure
```
ml_project/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── logging.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── interim/
│   └── external/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── validate.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── select_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── evaluate_model.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       └── helpers.py
├── models/
│   ├── trained/
│   ├── serialized/
│   └── checkpoints/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── test_utils/
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── deploy.py
├── docs/
│   ├── index.md
│   ├── data_dictionary.md
│   ├── model_card.md
│   └── api_reference.md
└── deployment/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── kubernetes/
    └── terraform/
```

### 2. Deep Learning Project Structure
```
dl_project/
├── README.md
├── requirements.txt
├── environment.yml
├── config/
│   ├── model_configs/
│   ├── training_configs/
│   └── data_configs/
├── data/
│   ├── datasets/
│   ├── annotations/
│   └── augmented/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── architectures/
│   │   ├── losses/
│   │   ├── metrics/
│   │   └── optimizers/
│   ├── training/
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── schedulers.py
│   └── inference/
│       ├── predictor.py
│       └── postprocess.py
├── experiments/
│   ├── baselines/
│   ├── ablations/
│   └── hyperopt/
├── checkpoints/
├── logs/
│   ├── tensorboard/
│   └── wandb/
└── deployment/
    ├── model_server/
    ├── batch_inference/
    └── edge_deployment/
```

## Design Patterns

### 1. Data Pipeline Pattern

#### Abstract Data Pipeline
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class DataPipeline(ABC):
    """Abstract base class for data pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.steps = []
    
    @abstractmethod
    def extract(self) -> Any:
        """Extract data from source."""
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform the data."""
        pass
    
    @abstractmethod
    def load(self, data: Any) -> None:
        """Load data to destination."""
        pass
    
    def run(self) -> None:
        """Run the complete pipeline."""
        data = self.extract()
        transformed_data = self.transform(data)
        self.load(transformed_data)
        
class BatchDataPipeline(DataPipeline):
    """Concrete implementation for batch processing."""
    
    def extract(self) -> pd.DataFrame:
        # Implementation for batch data extraction
        pass
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implementation for batch data transformation
        pass
    
    def load(self, data: pd.DataFrame) -> None:
        # Implementation for batch data loading
        pass
```

### 2. Model Factory Pattern

#### Model Factory
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelFactory(ABC):
    """Abstract factory for creating models."""
    
    @abstractmethod
    def create_model(self, model_type: str, **kwargs) -> Any:
        pass

class SklearnModelFactory(ModelFactory):
    """Factory for sklearn models."""
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**kwargs)
        elif model_type == "svm":
            from sklearn.svm import SVC
            return SVC(**kwargs)
        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class PyTorchModelFactory(ModelFactory):
    """Factory for PyTorch models."""
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        if model_type == "resnet":
            from torchvision.models import resnet50
            return resnet50(**kwargs)
        elif model_type == "bert":
            from transformers import BertModel
            return BertModel.from_pretrained(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

### 3. Strategy Pattern for Model Training

#### Training Strategy
```python
from abc import ABC, abstractmethod

class TrainingStrategy(ABC):
    """Abstract strategy for model training."""
    
    @abstractmethod
    def train(self, model, train_data, val_data, **kwargs):
        pass

class StandardTraining(TrainingStrategy):
    """Standard training strategy."""
    
    def train(self, model, train_data, val_data, **kwargs):
        # Standard training implementation
        pass

class AdversarialTraining(TrainingStrategy):
    """Adversarial training strategy."""
    
    def train(self, model, train_data, val_data, **kwargs):
        # Adversarial training implementation
        pass

class FederatedTraining(TrainingStrategy):
    """Federated learning training strategy."""
    
    def train(self, model, train_data, val_data, **kwargs):
        # Federated training implementation
        pass

class ModelTrainer:
    """Context class that uses training strategies."""
    
    def __init__(self, strategy: TrainingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: TrainingStrategy):
        self.strategy = strategy
    
    def train_model(self, model, train_data, val_data, **kwargs):
        return self.strategy.train(model, train_data, val_data, **kwargs)
```

### 4. Observer Pattern for Monitoring

#### Training Monitor
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class Observer(ABC):
    """Abstract observer for monitoring."""
    
    @abstractmethod
    def update(self, event: str, data: Dict[str, Any]):
        pass

class MetricsLogger(Observer):
    """Observer that logs training metrics."""
    
    def update(self, event: str, data: Dict[str, Any]):
        if event == "epoch_end":
            print(f"Epoch {data['epoch']}: Loss = {data['loss']}, Accuracy = {data['accuracy']}")

class ModelCheckpointer(Observer):
    """Observer that saves model checkpoints."""
    
    def update(self, event: str, data: Dict[str, Any]):
        if event == "best_model":
            # Save model checkpoint
            pass

class EarlyStopper(Observer):
    """Observer that implements early stopping."""
    
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
    
    def update(self, event: str, data: Dict[str, Any]):
        if event == "epoch_end":
            if data['val_loss'] < self.best_loss:
                self.best_loss = data['val_loss']
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    data['stop_training'] = True

class TrainingSubject:
    """Subject that notifies observers."""
    
    def __init__(self):
        self.observers = []
    
    def attach(self, observer: Observer):
        self.observers.append(observer)
    
    def detach(self, observer: Observer):
        self.observers.remove(observer)
    
    def notify(self, event: str, data: Dict[str, Any]):
        for observer in self.observers:
            observer.update(event, data)
```

## Configuration Management Patterns

### 1. Hierarchical Configuration

#### Configuration Hierarchy
```yaml
# base_config.yaml
model:
  type: "neural_network"
  hidden_layers: [128, 64, 32]
  dropout: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

# experiment_config.yaml (inherits from base)
inherits: "base_config.yaml"

model:
  hidden_layers: [256, 128, 64]  # Override
  regularization: "l2"           # Add new

training:
  epochs: 200                    # Override
```

### 2. Environment-Specific Configuration

#### Environment Configuration
```python
import os
from typing import Dict, Any

class ConfigManager:
    """Manages environment-specific configurations."""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv('ML_ENV', 'development')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        base_config = self._load_base_config()
        env_config = self._load_env_config()
        
        # Merge configurations
        return self._merge_configs(base_config, env_config)
    
    def _load_base_config(self) -> Dict[str, Any]:
        # Load base configuration
        pass
    
    def _load_env_config(self) -> Dict[str, Any]:
        # Load environment-specific configuration
        pass
    
    def _merge_configs(self, base: Dict, env: Dict) -> Dict[str, Any]:
        # Deep merge configuration dictionaries
        pass
```

## Testing Patterns

### 1. Model Testing Framework

#### Model Test Suite
```python
import unittest
from abc import ABC, abstractmethod

class ModelTestCase(unittest.TestCase, ABC):
    """Abstract test case for ML models."""
    
    def setUp(self):
        self.model = self.create_model()
        self.sample_data = self.create_sample_data()
    
    @abstractmethod
    def create_model(self):
        pass
    
    @abstractmethod
    def create_sample_data(self):
        pass
    
    def test_model_creation(self):
        """Test that model can be created."""
        self.assertIsNotNone(self.model)
    
    def test_model_fit(self):
        """Test that model can be trained."""
        X, y = self.sample_data
        self.model.fit(X, y)
        self.assertTrue(hasattr(self.model, 'is_fitted'))
    
    def test_model_predict(self):
        """Test that model can make predictions."""
        X, y = self.sample_data
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.assertEqual(len(predictions), len(y))
    
    def test_model_serialization(self):
        """Test that model can be serialized."""
        import pickle
        X, y = self.sample_data
        self.model.fit(X, y)
        
        # Test serialization
        serialized = pickle.dumps(self.model)
        deserialized = pickle.loads(serialized)
        
        # Test that deserialized model works
        predictions = deserialized.predict(X)
        self.assertEqual(len(predictions), len(y))

class RandomForestTestCase(ModelTestCase):
    """Concrete test case for Random Forest."""
    
    def create_model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=42)
    
    def create_sample_data(self):
        from sklearn.datasets import make_classification
        return make_classification(n_samples=100, n_features=10, random_state=42)
```

### 2. Data Quality Testing

#### Data Quality Tests
```python
import pandas as pd
from typing import List, Dict, Any

class DataQualityTest:
    """Framework for data quality testing."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
    
    def test_missing_values(self, threshold: float = 0.1) -> bool:
        """Test for acceptable missing value ratio."""
        missing_ratio = self.data.isnull().sum() / len(self.data)
        max_missing = missing_ratio.max()
        
        self.results['missing_values'] = {
            'max_missing_ratio': max_missing,
            'threshold': threshold,
            'passed': max_missing <= threshold
        }
        
        return max_missing <= threshold
    
    def test_data_types(self, expected_types: Dict[str, str]) -> bool:
        """Test that columns have expected data types."""
        type_check = {}
        all_passed = True
        
        for column, expected_type in expected_types.items():
            if column in self.data.columns:
                actual_type = str(self.data[column].dtype)
                passed = actual_type == expected_type
                type_check[column] = {
                    'expected': expected_type,
                    'actual': actual_type,
                    'passed': passed
                }
                all_passed &= passed
        
        self.results['data_types'] = type_check
        return all_passed
    
    def test_value_ranges(self, ranges: Dict[str, tuple]) -> bool:
        """Test that numeric columns are within expected ranges."""
        range_check = {}
        all_passed = True
        
        for column, (min_val, max_val) in ranges.items():
            if column in self.data.columns:
                col_min = self.data[column].min()
                col_max = self.data[column].max()
                passed = min_val <= col_min and col_max <= max_val
                
                range_check[column] = {
                    'expected_range': (min_val, max_val),
                    'actual_range': (col_min, col_max),
                    'passed': passed
                }
                all_passed &= passed
        
        self.results['value_ranges'] = range_check
        return all_passed
    
    def run_all_tests(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run all configured tests."""
        if 'missing_values' in config:
            self.test_missing_values(**config['missing_values'])
        
        if 'data_types' in config:
            self.test_data_types(config['data_types'])
        
        if 'value_ranges' in config:
            self.test_value_ranges(config['value_ranges'])
        
        return self.results
```

## Deployment Patterns

### 1. Model Serving Pattern

#### Model Server Template
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ModelServer(ABC):
    """Abstract model server interface."""
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        pass
    
    @abstractmethod
    def postprocess(self, predictions: Any) -> Any:
        pass
    
    def serve(self, input_data: Any) -> Any:
        """Main serving pipeline."""
        preprocessed = self.preprocess(input_data)
        predictions = self.predict(preprocessed)
        return self.postprocess(predictions)

class SklearnModelServer(ModelServer):
    """Model server for sklearn models."""
    
    def load_model(self, model_path: str) -> Any:
        import joblib
        return joblib.load(model_path)
    
    def preprocess(self, input_data: Dict) -> Any:
        import pandas as pd
        return pd.DataFrame([input_data])
    
    def predict(self, input_data: Any) -> Any:
        return self.model.predict(input_data)
    
    def postprocess(self, predictions: Any) -> Dict:
        return {'prediction': predictions[0]}
```

### 2. Batch Inference Pattern

#### Batch Processing Template
```python
from typing import Iterator, List, Any
import pandas as pd

class BatchProcessor:
    """Template for batch inference processing."""
    
    def __init__(self, model, batch_size: int = 1000):
        self.model = model
        self.batch_size = batch_size
    
    def process_file(self, input_file: str, output_file: str):
        """Process a file in batches."""
        batch_generator = self.read_batches(input_file)
        
        with open(output_file, 'w') as f:
            for batch_num, batch in enumerate(batch_generator):
                predictions = self.model.predict(batch)
                self.write_predictions(f, predictions, batch_num)
    
    def read_batches(self, input_file: str) -> Iterator[pd.DataFrame]:
        """Read input file in batches."""
        for chunk in pd.read_csv(input_file, chunksize=self.batch_size):
            yield chunk
    
    def write_predictions(self, file_handle, predictions: Any, batch_num: int):
        """Write predictions to output file."""
        # Implementation depends on output format
        pass
```

## Documentation Templates

### 1. Model Card Template

#### Model Card Structure
```markdown
# Model Card: [Model Name]

## Model Details
- **Model Type**: [Classification/Regression/etc.]
- **Model Architecture**: [Brief description]
- **Version**: [Version number]
- **Date**: [Creation date]
- **Authors**: [Author names]

## Intended Use
- **Primary Use Cases**: [Describe intended applications]
- **Primary Users**: [Target user groups]
- **Out-of-Scope Uses**: [Uses not recommended]

## Training Data
- **Dataset**: [Dataset name and source]
- **Size**: [Number of samples, features]
- **Collection Period**: [When data was collected]
- **Preprocessing**: [Data preprocessing steps]

## Performance
- **Metrics**: [Primary evaluation metrics]
- **Test Results**: [Performance on test set]
- **Cross-Validation**: [CV results if applicable]

## Limitations
- **Known Limitations**: [Model limitations]
- **Bias Considerations**: [Potential biases]
- **Fairness Assessment**: [Fairness evaluation results]

## Ethical Considerations
- **Potential Risks**: [Identified risks]
- **Mitigation Strategies**: [Risk mitigation approaches]
- **Use Guidelines**: [Responsible use guidelines]
```

### 2. API Documentation Template

#### API Reference Structure
```markdown
# API Reference

## Overview
Brief description of the API functionality.

## Authentication
Description of authentication requirements.

## Endpoints

### POST /predict
Make predictions using the trained model.

**Request Body:**
```json
{
  "features": [1.0, 2.0, 3.0],
  "model_version": "v1.0"
}
```

**Response:**
```json
{
  "prediction": 0.85,
  "confidence": 0.92,
  "model_version": "v1.0",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Error Responses:**
- 400: Bad Request - Invalid input format
- 404: Not Found - Model not found
- 500: Internal Server Error

## Rate Limiting
API rate limits and quotas.

## Examples
Code examples in different programming languages.
```

This template collection provides a comprehensive foundation for building robust, maintainable ML systems following industry best practices and design patterns.
